import binascii
import glob
import json
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from rdkit.Chem import RemoveAllHs
from scipy.spatial import cKDTree

from datasets.process_mols import read_molecule, get_lig_graph_with_matching, generate_conformer, moad_extract_receptor_structure
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt, crop_beyond
from utils import so3, torus

 
class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom, alpha=1, beta=1,
                 include_miscellaneous_atoms=False, crop_beyond_cutoff=None, time_independent=False, rmsd_cutoff=0,
                 minimum_t=0, sampling_mixing_coeff=0):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.minimum_t = minimum_t
        self.mixing_coeff = sampling_mixing_coeff
        self.alpha = alpha
        self.beta = beta
        self.crop_beyond_cutoff = crop_beyond_cutoff
        self.rmsd_cutoff = rmsd_cutoff
        self.time_independent = time_independent

    def forward(self, data):
        """
        数据变换的主入口函数（PyTorch Geometric兼容）
        步骤：
        1. 采样时间参数
        2. 应用噪声变换
        3. t=0: 原始数据 .t=1:纯噪声
        """
        t_tr, t_rot, t_tor, t = self.get_time()
        return self.apply_noise(data, t_tr, t_rot, t_tor, t)

    def __call__(self, data):
        t_tr, t_rot, t_tor, t = self.get_time()
        return self.apply_noise(data, t_tr, t_rot, t_tor, t)

    def get_time(self):
        if self.time_independent:
            t = np.random.beta(self.alpha, self.beta)
            t_tr, t_rot, t_tor = t,t,t
        else:
            t = None
            if self.mixing_coeff == 0:
                t = np.random.beta(self.alpha, self.beta)
                t = self.minimum_t + t * (1 - self.minimum_t)
            else:
                choice = np.random.binomial(1, self.mixing_coeff)
                t1 = np.random.beta(self.alpha, self.beta)
                t1 = t1 * self.minimum_t
                t2 = np.random.beta(self.alpha, self.beta)
                t2 = self.minimum_t + t2 * (1 - self.minimum_t)
                t = choice * t1 + (1 - choice) * t2 

            t_tr, t_rot, t_tor = t,t,t
        return t_tr, t_rot, t_tor, t

    def apply_noise(self, data, t_tr, t_rot, t_tor, t, tr_update = None, rot_update=None, torsion_updates=None):
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        if self.time_independent:
            orig_complex_graph = copy.deepcopy(data)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)

        if self.time_independent:
            set_time(data, 0, 0, 0, 0, 1, self.all_atom, device=None, include_miscellaneous_atoms=self.include_miscellaneous_atoms)
        else:
            set_time(data, t, t_tr, t_rot, t_tor, 1, self.all_atom, device=None, include_miscellaneous_atoms=self.include_miscellaneous_atoms)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
        torsion_updates = None if self.no_torsion else torsion_updates
        try:
            modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)
        except Exception as e:
            print("failed modify conformer")
            print(e)

        if self.time_independent:
            if self.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

            filterHs = torch.not_equal(data['ligand'].x[:, 0], 0).cpu().numpy()
            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
            ligand_pos = data['ligand'].pos.cpu().numpy()[filterHs]
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy()
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=1).mean(axis=0))
            data.y = torch.tensor(rmsd < self.rmsd_cutoff).float().unsqueeze(0)
            data.atom_y = data.y
            return data

        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma

        if data['ligand'].pos.shape[0] == 1:
            # if the ligand is a single atom, the rotational score is always 0
            data.rot_score = data.rot_score * 0

        if self.crop_beyond_cutoff is not None:
            crop_beyond(data, tr_sigma * 3 + self.crop_beyond_cutoff, self.all_atom)
        set_time(data, t, t_tr, t_rot, t_tor, 1, self.all_atom, device=None, include_miscellaneous_atoms=self.include_miscellaneous_atoms)
        return data


class PDBBind(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0, chain_cutoff=10,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 include_miscellaneous_atoms=False,
                 protein_path_list=None, ligand_descriptions=None, keep_local_structures=False,
                 protein_file="protein_processed", ligand_file="ligand",
                 knn_only_graph=False, matching_tries=1, dataset='PDBBind', nci_cache_path=None, plip_dir="data/plip"):

        super(PDBBind, self).__init__(root, transform)
        self.pdbbind_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.chain_cutoff = chain_cutoff
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.use_old_wrong_embedding_order = False
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.protein_file = protein_file
        self.fixed_knn_radius_graph = True
        self.knn_only_graph = knn_only_graph
        self.matching_tries = matching_tries
        self.ligand_file = ligand_file
        self.dataset = dataset
        self.nci_cache_path = nci_cache_path
        self.plip_dir = plip_dir
        assert knn_only_graph or (not all_atoms)
        self.all_atoms = all_atoms
        if matching or protein_path_list is not None and ligand_descriptions is not None:
            cache_path += '_torsion'
        if all_atoms:
            cache_path += '_allatoms'
        if self.nci_cache_path is not None:
            cache_path += '_nci'
        self.full_cache_path = os.path.join(cache_path, f'{dataset}3_limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                                        f'_chainCutoff{self.chain_cutoff if self.chain_cutoff is None else int(self.chain_cutoff)}'
                                            + (''if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + (''if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + '_full'
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode())))
                                            + ('' if protein_file == "protein_processed" else '_' + protein_file)
                                            + ('' if not self.fixed_knn_radius_graph else (f'_fixedKNN' if not self.knn_only_graph else '_fixedKNNonly'))
                                            + ('' if not self.include_miscellaneous_atoms else '_miscAtoms')
                                            + ('' if self.use_old_wrong_embedding_order else '_chainOrd')
                                            + ('' if self.matching_tries == 1 else f'_tries{matching_tries}'))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        if not self.check_all_complexes():
            os.makedirs(self.full_cache_path, exist_ok=True)
            if protein_path_list is None or ligand_descriptions is None:
                self.preprocessing()
            else:
                self.inference_preprocessing()

        self.complex_graphs, self.rdkit_ligands = self.collect_all_complexes()
        print_statistics(self.complex_graphs)
        list_names = [complex['name'] for complex in self.complex_graphs]
        with open(os.path.join(self.full_cache_path, f'pdbbind_{os.path.splitext(os.path.basename(self.split_path))[0][:3]}_names.txt'), 'w') as f:
            f.write('\n'.join(list_names))

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graphs[idx])
        if self.require_ligand:
            complex_graph.mol = RemoveAllHs(copy.deepcopy(self.rdkit_ligands[idx]))

        for a in ['random_coords', 'coords', 'seq', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq', 'to_keep', 'chain_ids']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)

        nci_edge = None
        try:
            nci_edge = complex_graph['ligand', 'nci_cand', 'receptor']
        except KeyError:
            nci_edge = None
        has_cached_nci = (
            nci_edge is not None
            and hasattr(nci_edge, 'edge_type_y')
            and nci_edge.edge_type_y is not None
        )
        if not has_cached_nci:
            self._add_nci_edges(complex_graph)
        return complex_graph

    def _add_nci_edges(self, complex_graph):
        if self.nci_cache_path is None:
            return
        complex_name = complex_graph['name'] if 'name' in complex_graph else None
        if complex_name is None:
            return
        nci_labels = self._load_nci_labels(str(complex_name))
        if nci_labels is None:
            return
        def _get_label(labels, *keys):
            for key in keys:
                if key in labels:
                    value = labels[key]
                    if value is not None:
                        return value
            return None

        edge_index = _get_label(nci_labels, "cand_edge_index", "edge_index")
        if edge_index is None:
            return
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t()
        edge_type = _get_label(nci_labels, "cand_edge_y_type", "edge_type_y")
        edge_dist = _get_label(nci_labels, "edge_y_dist", "edge_dist_y")
        cache_res_pos = nci_labels.get("res_pos")
        cache_res_chain_ids = nci_labels.get("res_chain_ids")
        cache_resnums = nci_labels.get("resnums")
        nci_edge = complex_graph['ligand', 'nci_cand', 'receptor']
        if cache_res_chain_ids is not None and cache_resnums is not None:
            if isinstance(cache_res_chain_ids, np.ndarray):
                cache_res_chain_ids = cache_res_chain_ids.tolist()
            if isinstance(cache_resnums, np.ndarray):
                cache_resnums = cache_resnums.tolist()
            cache_res_chain_ids = [str(chain) for chain in cache_res_chain_ids]
            cache_resnums = [int(resnum) for resnum in cache_resnums]

            curr_resnums = complex_graph['receptor'].resnums
            curr_res_chain_ids = complex_graph['receptor'].res_chain_ids
            if torch.is_tensor(curr_resnums):
                curr_resnums = curr_resnums.cpu().numpy()
            if torch.is_tensor(curr_res_chain_ids):
                curr_res_chain_ids = curr_res_chain_ids.cpu().numpy()
            curr_res_chain_ids = [str(chain) for chain in curr_res_chain_ids]
            curr_resnums = [int(resnum) for resnum in curr_resnums]

            current_map = {
                (chain, resnum): idx
                for idx, (chain, resnum) in enumerate(zip(curr_res_chain_ids, curr_resnums))
            }
            remap = torch.full((len(cache_resnums),), -1, dtype=torch.long)
            for idx, (chain, resnum) in enumerate(zip(cache_res_chain_ids, cache_resnums)):
                mapped = current_map.get((chain, resnum))
                if mapped is not None:
                    remap[idx] = mapped
            remapped_rec = remap[edge_index[1]]
            valid_edges = remapped_rec >= 0
            edge_index = torch.stack([edge_index[0], remapped_rec], dim=0)[:, valid_edges]
            if edge_type is not None:
                edge_type = torch.as_tensor(edge_type, dtype=torch.long)[valid_edges]
            if edge_dist is not None:
                edge_dist = torch.as_tensor(edge_dist, dtype=torch.float32)[valid_edges]
        elif cache_res_pos is not None:
            cache_res_pos = torch.as_tensor(cache_res_pos, dtype=torch.float32)
            curr_res_pos = complex_graph['receptor'].pos
            if cache_res_pos.numel() > 0 and curr_res_pos.numel() > 0:
                dist = torch.cdist(curr_res_pos, cache_res_pos)
                min_dist, nearest = torch.min(dist, dim=1)
                max_dist = 1.0
                valid_nodes = min_dist <= max_dist
                remap = torch.full((cache_res_pos.size(0),), -1, dtype=torch.long)
                remap[nearest[valid_nodes]] = torch.nonzero(valid_nodes, as_tuple=False).view(-1)
                remapped_rec = remap[edge_index[1]]
                valid_edges = remapped_rec >= 0
                edge_index = torch.stack([edge_index[0], remapped_rec], dim=0)[:, valid_edges]
                if edge_type is not None:
                    edge_type = torch.as_tensor(edge_type, dtype=torch.long)[valid_edges]
                if edge_dist is not None:
                    edge_dist = torch.as_tensor(edge_dist, dtype=torch.float32)[valid_edges]
        num_lig = complex_graph['ligand'].num_nodes
        num_rec = complex_graph['receptor'].num_nodes
        valid_edges = (
            (edge_index[0] >= 0)
            & (edge_index[0] < num_lig)
            & (edge_index[1] >= 0)
            & (edge_index[1] < num_rec)
        )
        if not torch.all(valid_edges):
            edge_index = edge_index[:, valid_edges]
            if edge_type is not None:
                edge_type = torch.as_tensor(edge_type, dtype=torch.long)[valid_edges]
            if edge_dist is not None:
                edge_dist = torch.as_tensor(edge_dist, dtype=torch.float32)[valid_edges]
        nci_edge.edge_index = edge_index
        if edge_type is not None:
            nci_edge.edge_type_y = torch.as_tensor(edge_type, dtype=torch.long)
        if edge_dist is not None:
            nci_edge.edge_dist_y = torch.as_tensor(edge_dist, dtype=torch.float32)

    def _add_nci_labels_from_plip(self, complex_graph, complex_name):
        if self.plip_dir is None or complex_name is None:
            return
        report_path = os.path.join(self.plip_dir, complex_name, "report.json")
        if not os.path.exists(report_path):
            return
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except json.JSONDecodeError:
            return

        lig_pos = complex_graph['ligand'].pos
        if not torch.is_tensor(lig_pos):
            lig_pos = lig_pos[0]
        res_pos = complex_graph['receptor'].pos
        if lig_pos is None or res_pos is None:
            return

        center = None
        if hasattr(complex_graph, "original_center"):
            center = complex_graph.original_center
            if torch.is_tensor(center):
                center = center.detach().cpu().numpy()
            center = np.asarray(center, dtype=np.float32).reshape(-1)

        lig_pos_np = lig_pos.detach().cpu().numpy() if torch.is_tensor(lig_pos) else np.asarray(lig_pos)
        res_pos_np = res_pos.detach().cpu().numpy() if torch.is_tensor(res_pos) else np.asarray(res_pos)

        try:
            pos_map, pos_dist, total_records, failed_records = parse_plip_records(
                report,
                lig_pos_np,
                res_pos_np,
                center=center,
            )
        except ValueError:
            return
        bad_ratio = 0.3
        if total_records > 0 and failed_records / total_records > bad_ratio:
            return

        lig_idx, res_idx = build_candidate_edges(lig_pos_np, res_pos_np, cutoff=10.0)
        edge_index = negative_sample_edges(
            lig_idx,
            res_idx,
            pos_map,
            num_residues=res_pos_np.shape[0],
            neg_per_pos=20,
            neg_min=10,
            neg_max=200,
        )
        y_type, y_dist = build_edge_labels(edge_index, pos_map, pos_dist)
        nci_edge = complex_graph['ligand', 'nci_cand', 'receptor']
        nci_edge.edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        nci_edge.edge_type_y = torch.as_tensor(y_type, dtype=torch.long)
        nci_edge.edge_dist_y = torch.as_tensor(y_dist, dtype=torch.float32)

    def _load_nci_labels(self, complex_name):
        if self.nci_cache_path is None:
            return None
        candidates = []
        if complex_name:
            candidates.extend([complex_name, complex_name[:4], complex_name[:6]])
        for candidate in dict.fromkeys(candidates):
            for ext in (".pt", ".npz"):
                path = os.path.join(self.nci_cache_path, f"{candidate}{ext}")
                if os.path.exists(path):
                    if ext == ".pt":
                        return torch.load(path, map_location="cpu")
                    data = np.load(path, allow_pickle=True)
                    return {key: data[key] for key in data.files}
        return None

    def _has_nci_labels(self, complex_name):
        if self.nci_cache_path is None:
            return False
        candidates = []
        if complex_name:
            candidates.extend([complex_name, complex_name[:4], complex_name[:6]])
        for candidate in dict.fromkeys(candidates):
            for ext in (".pt", ".npz"):
                path = os.path.join(self.nci_cache_path, f"{candidate}{ext}")
                if os.path.exists(path):
                    return True
        return False

    def _report_nci_stats_once(self, complex_names):
        if self.nci_cache_path is None:
            return
        if getattr(self, "_nci_stats_reported", False):
            return
        self._nci_stats_reported = True
        total = len(complex_names)
        found = sum(1 for name in complex_names if self._has_nci_labels(name))
        print(f'NCI labels available for {found}/{total} complexes in {self.nci_cache_path}')

    def preprocessing(self):
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')
        stats_enabled = os.environ.get("DIFFDOCK_PREPROCESS_STATS", "").lower() in {"1", "true", "yes", "on"}
        log_every = int(os.environ.get("DIFFDOCK_PREPROCESS_LOG_EVERY", "200"))
        total_start_time = None
        if stats_enabled:
            import time
            total_start_time = time.perf_counter()
        processed = 0
        success = 0
        failed = 0

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} complexes.')
        self._report_nci_stats_once(complex_names_all)

        if self.esm_embeddings_path is not None:
            id_to_embeddings = torch.load(self.esm_embeddings_path)
            chain_embeddings_dictlist = defaultdict(list)
            chain_indices_dictlist = defaultdict(list)
            for key, embedding in id_to_embeddings.items():
                key_name = key.split('_chain_')[0]
                if key_name in complex_names_all:
                    chain_embeddings_dictlist[key_name].append(embedding)
                    chain_indices_dictlist[key_name].append(int(key.split('_chain_')[1]))
            lm_embeddings_chains_all = []
            for name in complex_names_all:
                complex_chains_embeddings = chain_embeddings_dictlist[name]
                complex_chains_indices = chain_indices_dictlist[name]
                chain_reorder_idx = np.argsort(complex_chains_indices)
                reordered_chains = [complex_chains_embeddings[i] for i in chain_reorder_idx]
                lm_embeddings_chains_all.append(reordered_chains)
        else:
            lm_embeddings_chains_all = [None] * len(complex_names_all)

        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(complex_names_all)//1000+1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                continue
            import time
            batch_start = time.perf_counter()
            batch_processed = 0
            batch_failed = 0
            complex_names = complex_names_all[1000*i:1000*(i+1)]
            lm_embeddings_chains = lm_embeddings_chains_all[1000*i:1000*(i+1)]
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(complex_names), desc=f'loading complexes {i}/{len(complex_names_all)//1000+1}') as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(self.get_complex, zip(complex_names, lm_embeddings_chains, [None] * len(complex_names), [None] * len(complex_names))):
                    complex_graphs.extend(t[0])
                    rdkit_ligands.extend(t[1])
                    batch_processed += 1
                    processed += 1
                    if len(t[0]) == 0:
                        batch_failed += 1
                        failed += 1
                    else:
                        success += 1
                    if stats_enabled and processed % log_every == 0:
                        total_elapsed_s = time.perf_counter() - total_start_time
                        print(
                            f"[preprocess] processed={processed} success={success} failed={failed} "
                            f"elapsed={total_elapsed_s:.1f}s"
                        )
                    pbar.update()
            if self.num_workers > 1: p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'wb') as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands), f)
            if stats_enabled:
                batch_elapsed = time.perf_counter() - batch_start
                batch_fail_rate = (batch_failed / max(batch_processed, 1)) * 100.0
                print(
                    f"[preprocess] batch={i} wrote {len(complex_graphs)} graphs "
                    f"in {batch_elapsed:.1f}s fail_rate={batch_fail_rate:.2f}%"
                )

    def inference_preprocessing(self):
        ligands_list = []
        print('Reading molecules and generating local structures with RDKit')
        for ligand_description in tqdm(self.ligand_descriptions):
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol)
                ligands_list.append(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if not self.keep_local_structures:
                    mol.RemoveAllConformers()
                    mol = AddHs(mol)
                    generate_conformer(mol)
                ligands_list.append(mol)

        if self.esm_embeddings_path is not None:
            print('Reading language model embeddings.')
            lm_embeddings_chains_all = []
            if not os.path.exists(self.esm_embeddings_path): raise Exception('ESM embeddings path does not exist: ',self.esm_embeddings_path)
            for protein_path in self.protein_path_list:
                embeddings_paths = sorted(glob.glob(os.path.join(self.esm_embeddings_path, os.path.basename(protein_path)) + '*'))
                lm_embeddings_chains = []
                for embeddings_path in embeddings_paths:
                    lm_embeddings_chains.append(torch.load(embeddings_path)['representations'][33])
                lm_embeddings_chains_all.append(lm_embeddings_chains)
        else:
            lm_embeddings_chains_all = [None] * len(self.protein_path_list)

        print('Generating graphs for ligands and proteins')
        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(self.protein_path_list)//1000+1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                continue
            protein_paths_chunk = self.protein_path_list[1000*i:1000*(i+1)]
            ligand_description_chunk = self.ligand_descriptions[1000*i:1000*(i+1)]
            ligands_chunk = ligands_list[1000 * i:1000 * (i + 1)]
            lm_embeddings_chains = lm_embeddings_chains_all[1000*i:1000*(i+1)]
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(protein_paths_chunk), desc=f'loading complexes {i}/{len(protein_paths_chunk)//1000+1}') as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(self.get_complex, zip(protein_paths_chunk, lm_embeddings_chains, ligands_chunk,ligand_description_chunk)):
                    complex_graphs.extend(t[0])
                    rdkit_ligands.extend(t[1])
                    pbar.update()
            if self.num_workers > 1: p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'wb') as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands), f)

    def check_all_complexes(self):
        if os.path.exists(os.path.join(self.full_cache_path, f"heterographs.pkl")):
            return True

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        for i in range(len(complex_names_all) // 1000 + 1):
            if not os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                return False
        return True

    def collect_all_complexes(self):
        print('Collecting all complexes from cache', self.full_cache_path)
        if os.path.exists(os.path.join(self.full_cache_path, f"heterographs.pkl")):
            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'rb') as f:
                complex_graphs = pickle.load(f)
            if self.require_ligand:
                with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
                    rdkit_ligands = pickle.load(f)
            else:
                rdkit_ligands = None
            return complex_graphs, rdkit_ligands

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        complex_graphs_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'rb') as f:
                print(i)
                l = pickle.load(f)
                complex_graphs_all.extend(l)

        rdkit_ligands_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'rb') as f:
                l = pickle.load(f)
                rdkit_ligands_all.extend(l)

        return complex_graphs_all, rdkit_ligands_all

    def get_complex(self, par):
        name, lm_embedding_chains, ligand, ligand_description = par
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
            print("Folder not found", name)
            return [], []

        try:

            lig = read_mol(self.pdbbind_dir, name, suffix=self.ligand_file, remove_hs=False)
            if self.max_lig_size != None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                return [], []

            complex_graph = HeteroData()
            complex_graph['name'] = name
            get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                        self.num_conformers, remove_hs=self.remove_hs, tries=self.matching_tries)

            moad_extract_receptor_structure(path=os.path.join(self.pdbbind_dir, name, f'{name}_{self.protein_file}.pdb'),
                                            complex_graph=complex_graph,
                                            neighbor_cutoff=self.receptor_radius,
                                            max_neighbors=self.c_alpha_max_neighbors,
                                            lm_embeddings=lm_embedding_chains,
                                            knn_only_graph=self.knn_only_graph,
                                            all_atoms=self.all_atoms,
                                            atom_cutoff=self.atom_radius,
                                            atom_max_neighbors=self.atom_max_neighbors)

        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            return [], []

        if self.dataset == 'posebusters':
            other_positions = []
            all_mol_file = os.path.join(self.pdbbind_dir, name, f'{name}_ligands.sdf')
            supplier = Chem.SDMolSupplier(all_mol_file, sanitize=False, removeHs=False)
            for mol in supplier:
                Chem.SanitizeMol(mol)
                all_mol = RemoveAllHs(mol)
                for conf in all_mol.GetConformers():
                    other_positions.append(conf.GetPositions())

            print(f'Found {len(other_positions)} alternative poses for {name}')
            complex_graph['ligand'].orig_pos = np.asarray(other_positions)

        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        if (not self.matching) or self.num_conformers == 1:
            complex_graph['ligand'].pos -= protein_center
        else:
            for p in complex_graph['ligand'].pos:
                p -= protein_center

        complex_graph.original_center = protein_center
        complex_graph['receptor_name'] = name
        self._add_nci_labels_from_plip(complex_graph, name)
        return [complex_graph], [lig]


def print_statistics(complex_graphs):
    statistics = ([], [], [], [], [], [])
    receptor_sizes = []

    for complex_graph in complex_graphs:
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        receptor_sizes.append(complex_graph['receptor'].pos.shape[0])
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)
        statistics[4].append(int(complex_graph.random_coords) if "random_coords" in complex_graph else -1)
        if "random_coords" in complex_graph and complex_graph.random_coords and "rmsd_matching" in complex_graph:
            statistics[5].append(complex_graph.rmsd_matching)

    if len(statistics[5]) == 0:
        statistics[5].append(-1)
    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching', 'random coordinates', 'random rmsd matching']
    print('Number of complexes: ', len(complex_graphs))
    for i in range(len(name)):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")

    return


def read_mol(pdbbind_dir, name, suffix='ligand', remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_{suffix}.sdf'), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_{suffix}.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs


PI_LIGAND_MATCH_THRESHOLDS = (1.2, 1.6)


def parse_lig_idx_list(lig_idx_list, num_atoms):
    if isinstance(lig_idx_list, str):
        tokens = [t.strip() for t in lig_idx_list.split(",") if t.strip()]
        indices = [int(tok) for tok in tokens]
    elif isinstance(lig_idx_list, (list, tuple, np.ndarray)):
        indices = [int(v) for v in lig_idx_list]
    else:
        return []
    if not indices:
        return []
    max_idx = max(indices)
    if max_idx >= num_atoms:
        indices = [idx - 1 for idx in indices]
    return [idx for idx in indices if 0 <= idx < num_atoms]


def map_ligand_coord(lig_tree, coord, thresholds=(0.2, 0.4)):
    coord = np.asarray(coord, dtype=np.float32)
    dist, idx = lig_tree.query(coord, k=1)
    for threshold in thresholds:
        if dist <= threshold:
            return int(idx), float(dist)
    return None, float(dist)


def parse_plip_records(report, lig_pos, res_pos, center=None):
    type_to_idx = report.get("type_to_idx", {})
    if not type_to_idx:
        raise ValueError("Missing type_to_idx in PLIP report")
    lig_tree = cKDTree(lig_pos)
    res_tree = cKDTree(res_pos)
    pos_map = {}
    pos_dist = {}
    total_records = 0
    failed_records = 0

    for site in report.get("binding_sites", {}).values():
        interactions = site.get("interactions", {})
        for interaction_type, payload in interactions.items():
            records = payload.get("records", []) if isinstance(payload, dict) else []
            for record in records:
                total_records += 1
                prot_coord = record.get("PROTCOO")
                if prot_coord is None:
                    failed_records += 1
                    continue
                if center is not None:
                    prot_coord = np.asarray(prot_coord, dtype=np.float32) - center
                res_idx = int(res_tree.query(prot_coord, k=1)[1])

                type_id = type_to_idx.get(interaction_type)
                if type_id is None:
                    failed_records += 1
                    continue
                type_id = int(type_id) + 1

                dist_value = record.get("DIST")
                if dist_value is None:
                    dist_value = record.get("CENTDIST")
                dist_value = float(dist_value) if dist_value is not None else 0.0

                lig_indices = None
                if interaction_type in {"pistacking", "pication"}:
                    lig_idx_list = record.get("LIG_IDX_LIST")
                    if lig_idx_list:
                        lig_indices = parse_lig_idx_list(lig_idx_list, lig_pos.shape[0])
                        if not lig_indices:
                            lig_indices = None

                if lig_indices is None:
                    lig_coord = record.get("LIGCOO")
                    if lig_coord is None:
                        failed_records += 1
                        continue
                    if center is not None:
                        lig_coord = np.asarray(lig_coord, dtype=np.float32) - center
                    thresholds = PI_LIGAND_MATCH_THRESHOLDS if interaction_type in {"pistacking", "pication"} else (0.2, 0.4)
                    lig_idx, _ = map_ligand_coord(lig_tree, lig_coord, thresholds=thresholds)
                    if lig_idx is None:
                        failed_records += 1
                        continue
                    lig_indices = [lig_idx]

                for lig_idx in lig_indices:
                    key = (int(lig_idx), int(res_idx))
                    prev_dist = pos_dist.get(key)
                    if prev_dist is None or dist_value < prev_dist:
                        pos_map[key] = type_id
                        pos_dist[key] = dist_value

    return pos_map, pos_dist, total_records, failed_records


def build_candidate_edges(lig_pos, res_pos, cutoff=10.0):
    diff = lig_pos[:, None, :] - res_pos[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    lig_idx, res_idx = np.where(distances <= cutoff)
    return lig_idx, res_idx


def negative_sample_edges(lig_idx, res_idx, pos_map, num_residues, neg_per_pos=20, neg_min=10, neg_max=200):
    edge_set = {(int(l), int(r)) for l, r in zip(lig_idx, res_idx)}
    pos_by_lig = {}
    for (l, r) in pos_map.keys():
        if (l, r) in edge_set:
            pos_by_lig.setdefault(l, set()).add(r)

    sampled_edges = []
    rng = np.random.default_rng()
    for lig_atom in np.unique(lig_idx):
        lig_atom = int(lig_atom)
        pos_res = pos_by_lig.get(lig_atom, set())
        cand_res = [r for (l, r) in edge_set if l == lig_atom]
        pos_edges = [(lig_atom, r) for r in cand_res if r in pos_res]
        neg_pool = [r for r in cand_res if r not in pos_res]
        num_pos = len(pos_res)
        num_neg = min(neg_max, neg_per_pos * num_pos + neg_min)
        if len(neg_pool) <= num_neg:
            neg_res = neg_pool
        else:
            neg_res = rng.choice(neg_pool, size=num_neg, replace=False).tolist()
        sampled_edges.extend(pos_edges + [(lig_atom, r) for r in neg_res])
    if not sampled_edges:
        return np.zeros((2, 0), dtype=np.int64)
    sampled_edges = np.array(sampled_edges, dtype=np.int64)
    return sampled_edges.T


def build_edge_labels(edge_index, pos_map, pos_dist):
    num_edges = edge_index.shape[1]
    y_type = np.zeros(num_edges, dtype=np.int64)
    y_dist = np.zeros(num_edges, dtype=np.float32)
    for idx in range(num_edges):
        lig_idx = int(edge_index[0, idx])
        res_idx = int(edge_index[1, idx])
        key = (lig_idx, res_idx)
        if key in pos_map:
            y_type[idx] = pos_map[key]
            y_dist[idx] = pos_dist.get(key, 0.0)
    return y_type, y_dist
