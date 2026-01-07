import functools
import logging
import pprint
import traceback
from argparse import ArgumentParser, Namespace, FileType
import copy
import os
from functools import partial
import warnings
from typing import Mapping, Optional

import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# 系统资源管理（Unix系统专用）
try:
    import resource  # 系统资源管理
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)  # 获取文件描述符限制
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))  # 提高文件描述符上限，避免大数据集加载问题
except ImportError:
    # Windows系统没有resource模块，跳过文件描述符限制设置
    pass

# Ignore pandas deprecation warning around pyarrow
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message="(?s).*Pyarrow will become a required dependency of pandas.*")

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from rdkit import RDLogger
from rdkit.Chem import RemoveAllHs

# TODO imports are a little odd, utils seems to shadow things
from utils.logging_utils import configure_logger, get_logger
import utils.utils
from datasets.process_mols import write_mol_with_coords
from utils.download import download_and_extract
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, set_time
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.molecules_utils import get_symmetry_rmsd
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning,
                        message="The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`")

# Prody logging is very verbose by default
prody_logger = logging.getLogger(".prody")
prody_logger.setLevel(logging.ERROR)

REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")
REMOTE_URLS = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
               f"{REPOSITORY_URL}/releases/download/v1.1/diffdock_models.zip"]


def build_nci_edges(complex_graph, cutoff=10.0):
    lig_pos = complex_graph['ligand'].pos
    rec_pos = complex_graph['receptor'].pos
    dist = torch.cdist(lig_pos, rec_pos)
    mask = dist <= cutoff
    if mask.any():
        lig_idx, rec_idx = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([lig_idx, rec_idx], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=lig_pos.device)
    complex_graph['ligand', 'nci_cand', 'receptor'].edge_index = edge_index
    return edge_index


def compute_pose_rmsds(mol, ref_pos, pose_positions):
    rmsds = []
    for pose in pose_positions:
        rmsds.append(get_symmetry_rmsd(mol, ref_pos, pose))
    return np.asarray(rmsds)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default='default_inference_args.yaml')
    parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
    parser.add_argument('--complex_name', type=str, default=None, help='Name that the complex will be saved with')
    parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
    parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
    parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')
    
    # PDBBind数据集推理的新参数
    parser.add_argument('--use_pdbbind_test', action='store_true', default=False, help='Use PDBBind test set for inference. If enabled, will read complex names from test.txt and load data from pdbbind directory, ignoring --protein_ligand_csv, --protein_path, --protein_sequence and --ligand_description parameters')
    parser.add_argument('--test_split_file', type=str, default='data/splits/test.txt', help='Path to test split file containing complex names (default: data/splits/test.txt)')
    parser.add_argument('--pdbbind_dir', type=str, default='data/pdbbind', help='Path to pdbbind directory containing complex subdirectories (default: data/pdbbind)')
    parser.add_argument('--esm_embeddings_path', type=str, default='data/pdb.pt', help='Path to ESM embeddings file (default: data/pdb.pt)')

    parser.add_argument('-l', '--log', '--loglevel', type=str, default='WARNING', dest="loglevel",
                        help='Log level. Default %(default)s')

    parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
    parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
    parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

    parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
    parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--confidence_ckpt', type=str, default='best_model.pt', help='Checkpoint to use for the confidence model')

    parser.add_argument('--batch_size', type=int, default=10, help='')
    parser.add_argument('--no_final_step_noise', action='store_true', default=True, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')

    parser.add_argument('--old_score_model', action='store_true', default=False, help='')
    parser.add_argument('--old_confidence_model', action='store_true', default=True, help='')
    parser.add_argument('--initial_noise_std_proportion', type=float, default=-1.0, help='Initial noise std proportion')
    parser.add_argument('--choose_residue', action='store_true', default=False, help='')

    parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
    parser.add_argument('--temp_psi_tr', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tr', type=float, default=0.5)
    parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
    parser.add_argument('--temp_psi_rot', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_rot', type=float, default=0.5)
    parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
    parser.add_argument('--temp_psi_tor', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tor', type=float, default=0.5)

    parser.add_argument('--gnina_minimize', action='store_true', default=False, help='')
    parser.add_argument('--gnina_path', type=str, default='gnina', help='')
    parser.add_argument('--gnina_log_file', type=str, default='gnina_log.txt', help='')  # To redirect gnina subprocesses stdouts from the terminal window
    parser.add_argument('--gnina_full_dock', action='store_true', default=False, help='')
    parser.add_argument('--gnina_autobox_add', type=float, default=4.0)
    parser.add_argument('--gnina_poses_to_optimize', type=int, default=1)

    return parser


def main(args):

    configure_logger(args.loglevel)
    logger = get_logger()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    # Download models if they don't exist locally
    if not os.path.exists(args.model_dir):
        logger.info(f"Models not found. Downloading")
        remote_urls = REMOTE_URLS
        downloaded_successfully = False
        for remote_url in remote_urls:
            try:
                logger.info(f"Attempting download from {remote_url}")
                files_downloaded = download_and_extract(remote_url, os.path.dirname(args.model_dir))
                if not files_downloaded:
                    logger.info(f"Download from {remote_url} failed.")
                    continue
                logger.info(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
                downloaded_successfully = True
                # Once we have downloaded the models, we can break the loop
                break
            except Exception as e:
                pass

        if not downloaded_successfully:
            raise Exception(f"Models not found locally and failed to download them from {remote_urls}")

    print('########正在加载最佳模型########')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    if args.confidence_model_dir is not None:
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            confidence_args = Namespace(**yaml.full_load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"DiffDock will run on {device}")

    # 根据输入方式选择数据加载策略
    if args.use_pdbbind_test:
        # 使用PDBBind测试集
        print(f"Using PDBBind test set for inference")
        test_dataset = InferenceDataset.from_pdbbind(
            out_dir=args.out_dir,
            test_split_file=args.test_split_file,
            pdbbind_dir=args.pdbbind_dir,
            esm_embeddings_path=args.esm_embeddings_path,
            receptor_radius=score_model_args.receptor_radius,
            remove_hs=score_model_args.remove_hs,
            c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
            all_atoms=score_model_args.all_atoms,
            atom_radius=score_model_args.atom_radius,
            atom_max_neighbors=score_model_args.atom_max_neighbors,
            knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph
        )
        complex_name_list = test_dataset.complex_names
        # 为每个复合物创建输出目录
        for name in complex_name_list:
            write_dir = f'{args.out_dir}/{name}'
            os.makedirs(write_dir, exist_ok=True)
            
    elif args.protein_ligand_csv is not None:
        # 使用CSV文件
        df = pd.read_csv(args.protein_ligand_csv)
        complex_name_list = set_nones(df['complex_name'].tolist())
        protein_path_list = set_nones(df['protein_path'].tolist())
        protein_sequence_list = set_nones(df['protein_sequence'].tolist())
        ligand_description_list = set_nones(df['ligand_description'].tolist())
        
        complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
        for name in complex_name_list:
            write_dir = f'{args.out_dir}/{name}'
            os.makedirs(write_dir, exist_ok=True)

        # preprocessing of complexes into geometric graphs
        test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                        ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                        lm_embeddings=True,
                                        receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                        all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                        atom_max_neighbors=score_model_args.atom_max_neighbors,
                                        knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
    else:
        # 使用单独的复合物
        complex_name_list = [args.complex_name if args.complex_name else f"complex_0"]
        protein_path_list = [args.protein_path]
        protein_sequence_list = [args.protein_sequence]
        ligand_description_list = [args.ligand_description]
        
        complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
        for name in complex_name_list:
            write_dir = f'{args.out_dir}/{name}'
            os.makedirs(write_dir, exist_ok=True)

        # preprocessing of complexes into geometric graphs
        test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                        ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                        lm_embeddings=True,
                                        receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                        all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                        atom_max_neighbors=score_model_args.atom_max_neighbors,
                                        knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
   
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args.confidence_model_dir is not None and not confidence_args.use_original_model_cache:
        logger.info('Confidence model uses different type of graphs than the score model. '
                    'Loading (or creating if not existing) the data for the confidence model now.')
        
        if args.use_pdbbind_test:
            # 使用PDBBind测试集时，置信度模型使用相同的数据
            confidence_test_dataset = \
                InferenceDataset.from_pdbbind(
                    out_dir=args.out_dir,
                    test_split_file=args.test_split_file,
                    pdbbind_dir=args.pdbbind_dir,
                    esm_embeddings_path=args.esm_embeddings_path,
                    receptor_radius=confidence_args.receptor_radius,
                    remove_hs=confidence_args.remove_hs,
                    c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                    all_atoms=confidence_args.all_atoms,
                    atom_radius=confidence_args.atom_radius,
                    atom_max_neighbors=confidence_args.atom_max_neighbors,
                    knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
        else:
            # 原有的置信度模型数据加载逻辑
            confidence_test_dataset = \
                InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                 ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                 lm_embeddings=True,
                                 receptor_radius=confidence_args.receptor_radius, remove_hs=confidence_args.remove_hs,
                                 c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                                 all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
                                 atom_max_neighbors=confidence_args.atom_max_neighbors,
                                 precomputed_lm_embeddings=test_dataset.lm_embeddings,
                                 knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
    else:
        confidence_test_dataset = None

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args.old_score_model)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.confidence_model_dir is not None:
        confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                     confidence_mode=True, old=args.old_confidence_model)
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None

    tr_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')

    failures, skipped = 0, 0
    N = args.samples_per_complex
    test_ds_size = len(test_dataset)
    logger.info(f'Size of test dataset: {test_ds_size}')
    rerank_reported = 0
    for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
        if not orig_complex_graph.success[0]:
            skipped += 1
            logger.warning(f"The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
            continue
        try:
            if confidence_test_dataset is not None:
                confidence_complex_graph = confidence_test_dataset[idx]
                if not confidence_complex_graph.success:
                    skipped += 1
                    logger.warning(f"The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                    continue
                confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
            else:
                confidence_data_list = None
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
            randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max,
                               initial_noise_std_proportion=args.initial_noise_std_proportion,
                               choose_residue=args.choose_residue)

            lig = orig_complex_graph.mol[0]

            # initialize visualisation
            pdb = None
            if args.save_visualisation:
                visualization_list = []
                for graph in data_list:
                    pdb = PDBFile(lig)
                    pdb.add(lig, 0, 0)
                    pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                    pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                    visualization_list.append(pdb)
            else:
                visualization_list = None

            # run reverse diffusion
            data_list, confidence = sampling(data_list=data_list, model=model,
                                             inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                             tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                             device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                             visualization_list=visualization_list, confidence_model=confidence_model,
                                             confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                             batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise,
                                             temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot,
                                                            args.temp_sampling_tor],
                                             temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                             temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot,
                                                              args.temp_sigma_data_tor])

            ligand_pos_rel = np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in data_list])
            ligand_pos = np.asarray(
                [complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                 for complex_graph in data_list]
            )
            ligand_pos_rel_raw = ligand_pos_rel.copy()
            ligand_pos_raw = ligand_pos.copy()
            rmsds_raw = None
            mol_for_rmsd = None
            ref_positions = None
            if hasattr(orig_complex_graph['ligand'], 'orig_pos'):
                ref_positions = np.asarray(orig_complex_graph['ligand'].orig_pos)
                if ref_positions.ndim == 2:
                    ref_positions = ref_positions[None, :, :]
                mol_for_rmsd = copy.deepcopy(lig)
                if score_model_args.remove_hs:
                    mol_for_rmsd = RemoveAllHs(mol_for_rmsd)
                filter_hs = torch.not_equal(data_list[0]['ligand'].x[:, 0], 0).cpu().numpy()
                try:
                    ligand_pos_rel_raw = ligand_pos_rel_raw[:, filter_hs]
                    ref_positions = ref_positions[:, filter_hs] - orig_complex_graph.original_center.cpu().numpy()
                    complex_rmsds = []
                    for i in range(ref_positions.shape[0]):
                        complex_rmsds.append(compute_pose_rmsds(mol_for_rmsd, ref_positions[i], ligand_pos_rel_raw))
                    rmsds_raw = np.min(np.asarray(complex_rmsds), axis=0)
                except Exception as e:
                    logger.warning(f"Failed to compute RMSDs for {orig_complex_graph['name']}: {e}")

            # reorder predictions based on confidence output
            nci_scores = None
            if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                confidence = confidence[:, 0]
            if confidence is not None:
                confidence = confidence.cpu().numpy()
                re_order = np.argsort(confidence)[::-1]
                confidence = confidence[re_order]
                ligand_pos = ligand_pos[re_order]
                ligand_pos_rel = ligand_pos_rel[re_order]
                data_list = [data_list[i] for i in re_order]
                if confidence_data_list is not None:
                    confidence_data_list = [confidence_data_list[i] for i in re_order]

                if confidence_model is not None:
                    nci_scores = []
                    include_misc = bool(getattr(confidence_args, 'include_miscellaneous_atoms', False))
                    base_graphs = confidence_data_list if confidence_data_list is not None else data_list
                    with torch.no_grad():
                        for pose_idx, base_graph in enumerate(base_graphs):
                            pose_graph = copy.deepcopy(base_graph)
                            pose_graph['ligand'].pos = data_list[pose_idx]['ligand'].pos
                            pose_graph = Batch.from_data_list([pose_graph]).to(device)
                            set_time(pose_graph, 0, 0, 0, 0, 1, confidence_args.all_atoms, device, include_misc)
                            edge_index = build_nci_edges(pose_graph, cutoff=10.0)
                            if edge_index.size(1) == 0:
                                nci_scores.append(torch.tensor(-20.0, device=device))
                                continue
                            out = confidence_model(pose_graph, return_nci=True)
                            nci_logits = None
                            if isinstance(out, tuple):
                                if len(out) >= 3:
                                    nci_logits = out[2]
                                elif len(out) == 2:
                                    nci_logits = out[1]
                            if nci_logits is None or nci_logits.numel() == 0:
                                nci_scores.append(torch.tensor(-20.0, device=device))
                                continue
                            probs = torch.softmax(nci_logits, dim=-1)
                            conf_edge = probs[:, 1:].max(dim=-1).values
                            if conf_edge.numel() == 0:
                                nci_scores.append(torch.tensor(-20.0, device=device))
                                continue
                            top_k = min(50, conf_edge.numel())
                            top_conf = torch.topk(conf_edge, top_k).values
                            nci_scores.append(torch.log(top_conf + 1e-6).mean())

                    nci_scores = torch.stack(nci_scores).cpu().numpy()
                    score_final = confidence + 0.1 * nci_scores
                    final_order = np.argsort(score_final)[::-1]
                    confidence = confidence[final_order]
                    ligand_pos = ligand_pos[final_order]
                    ligand_pos_rel = ligand_pos_rel[final_order]
                    nci_scores = nci_scores[final_order]
                    data_list = [data_list[i] for i in final_order]
                    if confidence_data_list is not None:
                        confidence_data_list = [confidence_data_list[i] for i in final_order]
                    re_order = re_order[final_order]

            if rmsds_raw is not None and confidence is not None and rerank_reported < 10:
                rmsds_rerank = rmsds_raw[re_order]
                top1_raw = rmsds_raw[0]
                top5_raw = np.min(rmsds_raw[:min(5, rmsds_raw.shape[0])])
                top1_rerank = rmsds_rerank[0]
                top5_rerank = np.min(rmsds_rerank[:min(5, rmsds_rerank.shape[0])])
                print(
                    f"[Rerank RMSD] {orig_complex_graph['name']}: "
                    f"raw_top1={top1_raw:.3f}, raw_top5={top5_raw:.3f}, "
                    f"rerank_top1={top1_rerank:.3f}, rerank_top5={top5_rerank:.3f}"
                )
                rerank_reported += 1

            # save predictions
            write_dir = f'{args.out_dir}/{complex_name_list[idx]}'
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                if score_model_args.remove_hs: mol_pred = RemoveAllHs(mol_pred)
                if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
                if nci_scores is not None:
                    write_mol_with_coords(
                        mol_pred,
                        pos,
                        os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}_nci{nci_scores[rank]:.3f}.sdf')
                    )
                else:
                    write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))

            # save visualisation frames
            if args.save_visualisation:
                if confidence is not None:
                    for rank, batch_idx in enumerate(re_order):
                        visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
                else:
                    for rank, batch_idx in enumerate(ligand_pos):
                        visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

        except Exception as e:
            logger.warning("Failed on", orig_complex_graph["name"], e)
            failures += 1

    result_msg = f"""
    Failed for {failures} / {test_ds_size} complexes.
    Skipped {skipped} / {test_ds_size} complexes.
"""
    if failures or skipped:
        logger.warning(result_msg)
    else:
        logger.info(result_msg)
    logger.info(f"Results saved in {args.out_dir}")


if __name__ == "__main__":
    _args = get_parser().parse_args()
    main(_args)
