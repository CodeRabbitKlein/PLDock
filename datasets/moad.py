import os
import pickle
from multiprocessing import Pool
import random
import copy
from torch_geometric.data import Batch

import numpy as np
import torch
from prody import confProDy
from rdkit import Chem
from rdkit.Chem import RemoveHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import subgraph
from tqdm import tqdm
confProDy(verbosity='none')
from datasets.process_mols import get_lig_graph_with_matching, moad_extract_receptor_structure
from utils.utils import read_strings_from_txt

class MOAD(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split='train', limit_complexes=0, chain_cutoff=None,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, esm_embeddings_sequences_path=None, require_ligand=False,
                 include_miscellaneous_atoms=False, keep_local_structures=False,
                 min_ligand_size=0, knn_only_graph=False, matching_tries=1, multiplicity=1,
                 max_receptor_size=None, remove_promiscuous_targets=None, unroll_clusters=False, remove_pdbbind=False,
                 enforce_timesplit=False, no_randomness=False, single_cluster_name=None, total_dataset_size=None, skip_matching=False,
                 nci_cache_path=None):

        super(MOAD, self).__init__(root, transform)
        self.moad_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split = split
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.require_ligand = require_ligand
        self.esm_embeddings_path = esm_embeddings_path
        self.esm_embeddings_sequences_path = esm_embeddings_sequences_path
        self.keep_local_structures = keep_local_structures
        self.knn_only_graph = knn_only_graph
        self.matching_tries = matching_tries
        self.all_atoms = all_atoms
        self.multiplicity = multiplicity
        self.chain_cutoff = chain_cutoff
        self.no_randomness = no_randomness
        self.total_dataset_size = total_dataset_size
        self.skip_matching = skip_matching
        self.nci_cache_path = nci_cache_path

        self.prot_cache_path = os.path.join(cache_path, f'MOAD12_limit{self.limit_complexes}_INDEX{self.split}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                            + (''if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + ('' if not self.include_miscellaneous_atoms else '_miscAtoms')
                                            + ('' if not self.knn_only_graph else '_knnOnly')
                                            + ('' if self.nci_cache_path is None else '_nci'))

        self.lig_cache_path = os.path.join(cache_path, f'MOAD12_limit{self.limit_complexes}_INDEX{self.split}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                            + ('' if not matching else f'_matching')
                                            + ('' if not skip_matching else f'skip')
                                            + (''if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if self.matching_tries == 1 else f'_tries{matching_tries}')
                                            + ('' if self.nci_cache_path is None else '_nci'))

        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.single_cluster_name = single_cluster_name
        if split == 'train':
            split = 'PDBBind'

        with open("./data/splits/MOAD_generalisation_splits.pkl", "rb") as f:
            self.split_clusters = pickle.load(f)[split]

        clustes_path = os.path.join(self.moad_dir, "new_cluster_to_ligands.pkl")
        with open(clustes_path, "rb") as f:
            self.cluster_to_ligands = pickle.load(f)
            #self.cluster_to_ligands = {k: [s.split('.')[0] for s in v] for k, v in self.cluster_to_ligands.items()}

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        if not self.check_all_receptors():
            os.makedirs(self.prot_cache_path, exist_ok=True)
            self.preprocessing_receptors()

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        if not os.path.exists(os.path.join(self.lig_cache_path, "ligands.pkl")):
            os.makedirs(self.lig_cache_path, exist_ok=True)
            self.preprocessing_ligands()

        print('loading ligands from memory: ', os.path.join(self.lig_cache_path, "ligands.pkl"))
        with open(os.path.join(self.lig_cache_path, "ligands.pkl"), 'rb') as f:
            self.ligands = pickle.load(f)

        if require_ligand:
            with open(os.path.join(self.lig_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
                self.rdkit_ligands = pickle.load(f)
                self.rdkit_ligands = {lig.name:mol for mol, lig in zip(self.rdkit_ligands, self.ligands)}

        len_before = len(self.ligands)
        if not self.single_cluster_name is None:
            self.ligands = [lig for lig in self.ligands if lig.name in self.cluster_to_ligands[self.single_cluster_name]]
        print('Kept', len(self.ligands), f'ligands in {self.single_cluster_name} out of', len_before)

        len_before = len(self.ligands)
        self.ligands = {lig.name: lig for lig in self.ligands if min_ligand_size == 0 or lig['ligand'].x.shape[0] >= min_ligand_size}
        print('removed', len_before - len(self.ligands), 'ligands below minimum size out of', len_before)

        receptors_names = set([lig.name[:6] for lig in self.ligands.values()])
        self.collect_receptors(receptors_names, max_receptor_size, remove_promiscuous_targets)

        # filter ligands for which the receptor failed
        tot_before = len(self.ligands)
        self.ligands = {k:v for k, v in self.ligands.items() if k[:6] in self.receptors}
        print('removed', tot_before - len(self.ligands), 'ligands with no receptor out of', tot_before)

        if remove_pdbbind:
            complexes_pdbbind = read_strings_from_txt('data/splits/timesplit_no_lig_overlap_train') + read_strings_from_txt('data/splits/timesplit_no_lig_overlap_val')
            with open('data/BindingMOAD_2020_ab_processed_biounit/ecod_t_group_binding_site_assignment_dict_major_domain.pkl', 'rb') as f:
                pdbbind_to_cluster = pickle.load(f)
            clusters_pdbbind = set([pdbbind_to_cluster[c] for c in complexes_pdbbind])
            self.split_clusters = [c for c in self.split_clusters if c not in clusters_pdbbind]
            self.cluster_to_ligands = {k: v for k, v in self.cluster_to_ligands.items() if k not in clusters_pdbbind}
            ligand_accepted = []
            for c, ligands in self.cluster_to_ligands.items():
                ligand_accepted += ligands
            ligand_accepted = set(ligand_accepted)
            tot_before = len(self.ligands)
            self.ligands = {k: v for k, v in self.ligands.items() if k in ligand_accepted}
            print('removed', tot_before - len(self.ligands), 'ligands in overlap with PDBBind out of', tot_before)

        if enforce_timesplit:
            with open("data/splits/pdbids_2019", "r") as f:
                lines = f.readlines()
            pdbids_from2019 = []
            for i in range(6, len(lines), 4):
                pdbids_from2019.append(lines[i][18:22])

            pdbids_from2019 = set(pdbids_from2019)
            len_before = len(self.ligands)
            self.ligands = {k: v for k, v in self.ligands.items() if k[:4].upper() not in pdbids_from2019}
            print('removed', len_before - len(self.ligands), 'ligands from 2019 out of', len_before)

        if unroll_clusters:
            rec_keys = set([k[:6] for k in self.ligands.keys()])
            self.cluster_to_ligands = {k:[k2 for k2 in self.ligands.keys() if k2[:6] == k] for k in rec_keys}
            self.split_clusters = list(rec_keys)
        else:
            for c in self.cluster_to_ligands.keys():
                 self.cluster_to_ligands[c] = [v for v in self.cluster_to_ligands[c] if v in self.ligands]
            self.split_clusters = [c for c in self.split_clusters if len(self.cluster_to_ligands[c])>0]

        print_statistics(self)
        list_names = [name for cluster in self.split_clusters for name in self.cluster_to_ligands[cluster]]
        with open(os.path.join(self.prot_cache_path, f'moad_{self.split}_names.txt'), 'w') as f:
            f.write('\n'.join(list_names))

    def len(self):
        return len(self.split_clusters) * self.multiplicity if self.total_dataset_size is None else self.total_dataset_size

    def get_by_name(self, ligand_name, cluster):
        ligand_graph = copy.deepcopy(self.ligands[ligand_name])
        complex_graph = copy.deepcopy(self.receptors[ligand_name[:6]])

        if False and self.keep_original and hasattr(ligand_graph['ligand'], 'orig_pos'):
            lig_path = os.path.join(self.moad_dir, 'pdb_superligand', ligand_name + '.pdb')
            lig = Chem.MolFromPDBFile(lig_path)
            formula = np.asarray([atom.GetSymbol() for atom in lig.GetAtoms()])

            # check for same receptor/ligand pair with a different binding position
            for ligand_comp in self.cluster_to_ligands[cluster]:
                if ligand_comp == ligand_name or ligand_comp[:6] != ligand_name[:6]:
                    continue

                lig_path_comp = os.path.join(self.moad_dir, 'pdb_superligand', ligand_comp + '.pdb')
                if not os.path.exists(lig_path_comp):
                    continue

                lig_comp = Chem.MolFromPDBFile(lig_path_comp)
                formula_comp = np.asarray([atom.GetSymbol() for atom in lig_comp.GetAtoms()])

                if formula.shape == formula_comp.shape and np.all(formula == formula_comp) and hasattr(
                        self.ligands[ligand_comp], 'orig_pos'):
                    print(f'Found complex {ligand_comp} to have the same complex/ligand pair, adding it into orig_pos')
                    # add the orig_pos of the binding position
                    if not isinstance(ligand_graph['ligand'].orig_pos, list):
                        ligand_graph['ligand'].orig_pos = [ligand_graph['ligand'].orig_pos]
                    ligand_graph['ligand'].orig_pos.append(self.ligands[ligand_comp].orig_pos)

        for type in ligand_graph.node_types + ligand_graph.edge_types:
            for key, value in ligand_graph[type].items():
                complex_graph[type][key] = value
        complex_graph.name = ligand_graph.name
        if isinstance(complex_graph['ligand'].pos, list):
            for i in range(len(complex_graph['ligand'].pos)):
                complex_graph['ligand'].pos[i] -= complex_graph.original_center
        else:
            complex_graph['ligand'].pos -= complex_graph.original_center
        if self.require_ligand:
            complex_graph.mol = copy.deepcopy(self.rdkit_ligands[ligand_name])

        if self.chain_cutoff:
            distances = torch.norm(
                (torch.from_numpy(complex_graph['ligand'].orig_pos[0]) - complex_graph.original_center).unsqueeze(1) - complex_graph['receptor'].pos.unsqueeze(0), dim=2)
            distances = distances.min(dim=0)[0]
            if torch.min(distances) >= self.chain_cutoff:
                print('minimum distance', torch.min(distances), 'too large', ligand_name,
                      'skipping and returning random. Number of chains',
                      torch.max(complex_graph['receptor'].chain_ids) + 1)
                return self.get(random.randint(0, self.len()))

            within_cutoff = distances < self.chain_cutoff
            chains_within_cutoff = torch.zeros(torch.max(complex_graph['receptor'].chain_ids) + 1)
            chains_within_cutoff.index_add_(0, complex_graph['receptor'].chain_ids, within_cutoff.float())
            chains_within_cutoff_bool = chains_within_cutoff > 0
            residues_to_keep = chains_within_cutoff_bool[complex_graph['receptor'].chain_ids]

            if self.all_atoms:
                atom_to_res_mapping = complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index[1]
                atoms_to_keep = residues_to_keep[atom_to_res_mapping]
                rec_remapper = (torch.cumsum(residues_to_keep.long(), dim=0) - 1)
                atom_to_res_new_mapping = rec_remapper[atom_to_res_mapping][atoms_to_keep]
                atom_res_edge_index = torch.stack([torch.arange(len(atom_to_res_new_mapping)), atom_to_res_new_mapping])

                complex_graph['atom'].x = complex_graph['atom'].x[atoms_to_keep]
                complex_graph['atom'].pos = complex_graph['atom'].pos[atoms_to_keep]
                complex_graph['atom', 'atom_contact', 'atom'].edge_index = \
                    subgraph(atoms_to_keep, complex_graph['atom', 'atom_contact', 'atom'].edge_index,
                             relabel_nodes=True)[0]
                complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index

            complex_graph['receptor'].pos = complex_graph['receptor'].pos[residues_to_keep]
            complex_graph['receptor'].x = complex_graph['receptor'].x[residues_to_keep]
            complex_graph['receptor'].side_chain_vecs = complex_graph['receptor'].side_chain_vecs[residues_to_keep]
            complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = \
            subgraph(residues_to_keep, complex_graph['receptor', 'rec_contact', 'receptor'].edge_index,
                     relabel_nodes=True)[0]

            extra_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
            complex_graph['receptor'].pos -= extra_center
            if isinstance(complex_graph['ligand'].pos, list):
                for i in range(len(complex_graph['ligand'].pos)):
                    complex_graph['ligand'].pos[i] -= extra_center
            else:
                complex_graph['ligand'].pos -= extra_center
            complex_graph.original_center += extra_center

        complex_graph['receptor'].pop('chain_ids')

        for a in ['random_coords', 'coords', 'seq', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq',
                  'to_keep', 'chain_ids']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)

        self._add_nci_edges(complex_graph, ligand_name)
        return complex_graph

    def get(self, idx):
        if self.total_dataset_size is not None:
            idx = random.randint(0, len(self.split_clusters) - 1)

        idx = idx % len(self.split_clusters)
        cluster = self.split_clusters[idx]

        if self.no_randomness:
            ligand_name = sorted(self.cluster_to_ligands[cluster])[0]
        else:
            ligand_name = random.choice(self.cluster_to_ligands[cluster])

        complex_graph = self.get_by_name(ligand_name, cluster)
        
        if self.total_dataset_size is not None:
            complex_graph = Batch.from_data_list([complex_graph])
            
        return complex_graph

    def get_all_complexes(self):
        complexes = {}
        for cluster in self.split_clusters:
            for ligand_name in self.cluster_to_ligands[cluster]:
                complexes[ligand_name] = self.get_by_name(ligand_name, cluster)
        return complexes

    def _add_nci_edges(self, complex_graph, ligand_name):
        if self.nci_cache_path is None:
            return
        nci_labels = self._load_nci_labels(ligand_name)
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

    def _load_nci_labels(self, ligand_name):
        if self.nci_cache_path is None:
            return None
        candidates = []
        if ligand_name:
            candidates.extend([ligand_name, ligand_name[:4], ligand_name[:6]])
        for candidate in dict.fromkeys(candidates):
            for ext in (".pt", ".npz"):
                path = os.path.join(self.nci_cache_path, f"{candidate}{ext}")
                if os.path.exists(path):
                    if ext == ".pt":
                        return torch.load(path, map_location="cpu")
                    data = np.load(path, allow_pickle=True)
                    return {key: data[key] for key in data.files}
        return None

    def preprocessing_receptors(self):
        print(f'Processing receptors from [{self.split}] and saving it to [{self.prot_cache_path}]')

        complex_names_all = sorted([l for c in self.split_clusters for l in self.cluster_to_ligands[c]])
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]

        receptor_names_all = [l[:6] for l in complex_names_all]
        receptor_names_all = sorted(list(dict.fromkeys(receptor_names_all)))
        print(f'Loading {len(receptor_names_all)} receptors.')

        if self.esm_embeddings_path is not None:
            id_to_embeddings = torch.load(self.esm_embeddings_path)
            sequences_list = read_strings_from_txt(self.esm_embeddings_sequences_path)
            sequences_to_embeddings = {}
            for i, seq in enumerate(sequences_list):
                sequences_to_embeddings[seq] = id_to_embeddings[str(i)]

        else:
            sequences_to_embeddings = None

        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(receptor_names_all)//1000+1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.prot_cache_path, f"receptors{i}.pkl")):
                continue
            receptor_names = receptor_names_all[1000*i:1000*(i+1)]
            receptor_graphs = []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(receptor_names), desc=f'loading receptors {i}/{len(receptor_names_all)//1000+1}') as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(self.get_receptor, zip(receptor_names, [sequences_to_embeddings]*len(receptor_names))):
                    if t is not None:
                        print(len(receptor_graphs))
                        receptor_graphs.append(t)
                    pbar.update()
            if self.num_workers > 1: p.__exit__(None, None, None)

            print('Number of receptors: ', len(receptor_graphs))
            with open(os.path.join(self.prot_cache_path, f"receptors{i}.pkl"), 'wb') as f:
                pickle.dump((receptor_graphs), f)
        return receptor_names_all

    def check_all_receptors(self):
        complex_names_all = sorted([l for c in self.split_clusters for l in self.cluster_to_ligands[c]])
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        receptor_names_all = [l[:6] for l in complex_names_all]
        receptor_names_all = list(dict.fromkeys(receptor_names_all))
        for i in range(len(receptor_names_all)//1000+1):
            if not os.path.exists(os.path.join(self.prot_cache_path, f"receptors{i}.pkl")):
                return False
        return True

    def collect_receptors(self, receptors_to_keep=None, max_receptor_size=None, remove_promiscuous_targets=None):
        complex_names_all = sorted([l for c in self.split_clusters for l in self.cluster_to_ligands[c]])
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        receptor_names_all = [l[:6] for l in complex_names_all]
        receptor_names_all = sorted(list(dict.fromkeys(receptor_names_all)))

        receptor_graphs_all = []
        total_recovered = 0
        print(f'Loading {len(receptor_names_all)} receptors to keep {len(receptors_to_keep)}.')
        for i in range(len(receptor_names_all)//1000+1):
            print(f'prot path: {os.path.join(self.prot_cache_path, f"receptors{i}.pkl")}')
            with open(os.path.join(self.prot_cache_path, f"receptors{i}.pkl"), 'rb') as f:
                l = pickle.load(f)
                total_recovered += len(l)
                if receptors_to_keep is not None:
                    l = [t for t in l if t['receptor_name'] in receptors_to_keep]
                receptor_graphs_all.extend(l)

        cur_len = len(receptor_graphs_all)
        print(f"Kept {len(receptor_graphs_all)} receptors out of {len(receptor_names_all)} total and recovered {total_recovered}")

        if max_receptor_size is not None:
            receptor_graphs_all = [rec for rec in receptor_graphs_all if rec["receptor"].pos.shape[0] <= max_receptor_size]
            print(f"Kept {len(receptor_graphs_all)} receptors out of {cur_len} after filtering by size")
            cur_len = len(receptor_graphs_all)

        if remove_promiscuous_targets is not None:
            promiscuous_targets = set()
            for name in complex_names_all:
                l = name.split('_')
                if int(l[3]) > remove_promiscuous_targets:
                    promiscuous_targets.add(name[:6])
            receptor_graphs_all = [rec for rec in receptor_graphs_all if rec["receptor_name"] not in promiscuous_targets]
            print(f"Kept {len(receptor_graphs_all)} receptors out of {cur_len} after removing promiscuous targets")

        self.receptors = {}
        for r in receptor_graphs_all:
            self.receptors[r['receptor_name']] = r
        return

    def get_receptor(self, par):
        name, sequences_to_embeddings = par
        rec_path = os.path.join(self.moad_dir, 'pdb_protein', name + '_protein.pdb')
        if not os.path.exists(rec_path):
            print("Receptor not found", name, rec_path)
            return None

        complex_graph = HeteroData()
        complex_graph['receptor_name'] = name
        try:
            moad_extract_receptor_structure(path=rec_path, complex_graph=complex_graph, neighbor_cutoff=self.receptor_radius,
                                            max_neighbors=self.c_alpha_max_neighbors, sequences_to_embeddings=sequences_to_embeddings,
                                            knn_only_graph=self.knn_only_graph, all_atoms=self.all_atoms, atom_cutoff=self.atom_radius,
                                            atom_max_neighbors=self.atom_max_neighbors)

        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            return None

        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center
        complex_graph.original_center = protein_center
        return complex_graph


    def preprocessing_ligands(self):
        print(f'Processing complexes from [{self.split}] and saving it to [{self.lig_cache_path}]')

        complex_names_all = sorted([l for c in self.split_clusters for l in self.cluster_to_ligands[c]])
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} ligands.')
        self._report_nci_stats_once(complex_names_all)

        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(complex_names_all)//1000+1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.lig_cache_path, f"ligands{i}.pkl")):
                continue
            complex_names = complex_names_all[1000*i:1000*(i+1)]
            ligand_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(complex_names), desc=f'loading complexes {i}/{len(complex_names_all)//1000+1}') as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(self.get_ligand, complex_names):
                    if t is not None:
                        ligand_graphs.append(t[0])
                        rdkit_ligands.append(t[1])
                    pbar.update()
            if self.num_workers > 1: p.__exit__(None, None, None)

            with open(os.path.join(self.lig_cache_path, f"ligands{i}.pkl"), 'wb') as f:
                pickle.dump((ligand_graphs), f)
            with open(os.path.join(self.lig_cache_path, f"rdkit_ligands{i}.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands), f)

        ligand_graphs_all = []
        for i in range(len(complex_names_all)//1000+1):
            with open(os.path.join(self.lig_cache_path, f"ligands{i}.pkl"), 'rb') as f:
                l = pickle.load(f)
                ligand_graphs_all.extend(l)
        with open(os.path.join(self.lig_cache_path, f"ligands.pkl"), 'wb') as f:
            pickle.dump((ligand_graphs_all), f)

        rdkit_ligands_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.lig_cache_path, f"rdkit_ligands{i}.pkl"), 'rb') as f:
                l = pickle.load(f)
                rdkit_ligands_all.extend(l)
        with open(os.path.join(self.lig_cache_path, f"rdkit_ligands.pkl"), 'wb') as f:
            pickle.dump((rdkit_ligands_all), f)

    def get_ligand(self, name):
        if self.split == 'train':
            lig_path = os.path.join(self.moad_dir, 'pdb_superligand', name + '.pdb')
        else:
            lig_path = os.path.join(self.moad_dir, 'pdb_ligand', name + '.pdb')

        if not os.path.exists(lig_path):
            print("Ligand not found", name, lig_path)
            return None

        # read pickle
        lig = Chem.MolFromPDBFile(lig_path)

        if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
            print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
            return None

        try:
            if self.matching:
                smile = Chem.MolToSmiles(lig)
                if '.' in smile:
                    print(f'Ligand {name} has multiple fragments and we are doing matching. Not including {name} in preprocessed data.')
                    return None

            complex_graph = HeteroData()
            complex_graph['name'] = name

            Chem.SanitizeMol(lig)
            get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                        self.num_conformers, remove_hs=self.remove_hs, tries=self.matching_tries, skip_matching=self.skip_matching)
        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            return None

        if self.split != 'train':
            other_positions = [complex_graph['ligand'].orig_pos]
            nsplit = name.split('_')
            for i in range(100):
                new_file = os.path.join(self.moad_dir, 'pdb_ligand', f'{nsplit[0]}_{nsplit[1]}_{nsplit[2]}_{i}.pdb')
                if os.path.exists(new_file):
                    if i != int(nsplit[3]):
                        lig = Chem.MolFromPDBFile(new_file)
                        lig = RemoveHs(lig, sanitize=True)
                        other_positions.append(lig.GetConformer().GetPositions())
                else:
                    break
            complex_graph['ligand'].orig_pos = np.asarray(other_positions)

        return complex_graph, lig


def print_statistics(dataset):
    statistics = ([], [], [], [], [], [])
    receptor_sizes = []

    for i in range(len(dataset)):
        complex_graph = dataset[i]
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
    print('Number of complexes: ', len(dataset))
    for i in range(len(name)):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")

    return
