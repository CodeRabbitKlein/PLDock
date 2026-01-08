import argparse
import json
import os

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
import prody as pr
from torch_geometric.data import HeteroData

from datasets.pdbbind import read_mol
from datasets.process_mols import moad_extract_receptor_structure


def get_ligand_positions(pdbbind_dir, pdb_id, ligand_file="ligand", remove_hs=False):
    lig = read_mol(pdbbind_dir, pdb_id, suffix=ligand_file, remove_hs=remove_hs)
    if lig is None:
        raise ValueError(f"Failed to read ligand for {pdb_id}")
    if lig.GetNumConformers() == 0:
        raise ValueError(f"Ligand has no conformer for {pdb_id}")
    conf = lig.GetConformer()
    lig_pos = np.asarray(conf.GetPositions(), dtype=np.float32)
    return lig_pos


def get_receptor_positions(pdbbind_dir, pdb_id, protein_file="protein_processed"):
    protein_path = os.path.join(pdbbind_dir, pdb_id, f"{pdb_id}_{protein_file}.pdb")
    if not os.path.exists(protein_path):
        raise FileNotFoundError(f"Missing protein file: {protein_path}")
    pdb = pr.parsePDB(protein_path)
    ca = pdb.ca
    if ca is None:
        raise ValueError(f"No CA atoms found for {pdb_id}")
    res_pos = np.asarray(ca.getCoords(), dtype=np.float32)
    resnums = np.asarray(ca.getResnums())
    chains = np.asarray(ca.getChids())
    res_keys = [(chain, int(resnum)) for chain, resnum in zip(chains, resnums)]
    res_key_to_idx = {res_key: idx for idx, res_key in enumerate(res_keys)}
    return res_pos, res_key_to_idx, res_keys


def load_training_args(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing training args file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Training args file must contain a dictionary of args.")
    return data


def resolve_training_graph_params(args):
    train_args = load_training_args(args.training_args) if args.training_args else {}
    train_defaults = {
        "receptor_radius": 30,
        "chain_cutoff": None,
        "c_alpha_max_neighbors": 10,
        "not_knn_only_graph": False,
        "all_atoms": False,
        "atom_radius": 5,
        "atom_max_neighbors": 8,
    }

    def get_value(key, default):
        value = train_args.get(key, None)
        return default if value is None else value

    if "knn_only_graph" in train_args:
        knn_only_graph = bool(train_args["knn_only_graph"])
    else:
        not_knn_only_graph = train_args.get("not_knn_only_graph", train_defaults["not_knn_only_graph"])
        knn_only_graph = not not_knn_only_graph

    return {
        "receptor_radius": get_value("receptor_radius", args.receptor_radius if args.receptor_radius is not None else train_defaults["receptor_radius"]),
        "chain_cutoff": get_value("chain_cutoff", args.chain_cutoff),
        "c_alpha_max_neighbors": get_value("c_alpha_max_neighbors", train_defaults["c_alpha_max_neighbors"]),
        "knn_only_graph": knn_only_graph,
        "all_atoms": get_value("all_atoms", train_defaults["all_atoms"]),
        "atom_radius": get_value("atom_radius", train_defaults["atom_radius"]),
        "atom_max_neighbors": get_value("atom_max_neighbors", train_defaults["atom_max_neighbors"]),
    }


def get_training_receptor_positions(pdbbind_dir, pdb_id, protein_file, training_graph_params):
    protein_path = os.path.join(pdbbind_dir, pdb_id, f"{pdb_id}_{protein_file}.pdb")
    if not os.path.exists(protein_path):
        raise FileNotFoundError(f"Missing protein file: {protein_path}")

    complex_graph = HeteroData()
    moad_extract_receptor_structure(
        path=protein_path,
        complex_graph=complex_graph,
        neighbor_cutoff=training_graph_params.get("receptor_radius"),
        max_neighbors=training_graph_params.get("c_alpha_max_neighbors"),
        knn_only_graph=training_graph_params.get("knn_only_graph", False),
        all_atoms=training_graph_params.get("all_atoms", False),
        atom_cutoff=training_graph_params.get("atom_radius"),
        atom_max_neighbors=training_graph_params.get("atom_max_neighbors"),
    )

    res_pos = complex_graph["receptor"].pos.cpu().numpy()
    res_chain_ids = getattr(complex_graph["receptor"], "res_chain_ids", None)
    resnums = getattr(complex_graph["receptor"], "resnums", None)
    if res_chain_ids is None or resnums is None:
        pdb = pr.parsePDB(protein_path)
        ca = pdb.ca
        res_chain_ids = np.asarray(ca.getChids())
        res_seg_ids = np.asarray(ca.getSegnames())
        res_chain_ids = np.asarray([s + c for s, c in zip(res_seg_ids, res_chain_ids)])
        resnums = np.asarray(ca.getResnums())

    if torch.is_tensor(resnums):
        resnums = resnums.cpu().numpy()
    res_chain_ids = np.asarray(res_chain_ids)
    res_keys = [(str(chain), int(resnum)) for chain, resnum in zip(res_chain_ids, resnums)]
    res_key_to_idx = {res_key: idx for idx, res_key in enumerate(res_keys)}
    return res_pos, res_key_to_idx, res_keys


def map_ligand_coord(lig_tree, coord, thresholds=(0.2, 0.4)):
    coord = np.asarray(coord, dtype=np.float32)
    dist, idx = lig_tree.query(coord, k=1)
    for threshold in thresholds:
        if dist <= threshold:
            return int(idx), float(dist)
    return None, float(dist)


PI_LIGAND_MATCH_THRESHOLDS = (1.2, 1.6)


def parse_plip_records(report, lig_pos, res_key_to_idx):
    type_to_idx = report.get("type_to_idx", {})
    if not type_to_idx:
        raise ValueError("Missing type_to_idx in PLIP report")
    lig_tree = cKDTree(lig_pos)
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
                res_chain = record.get("RESCHAIN")
                res_num = record.get("RESNR")
                if res_chain is None or res_num is None:
                    failed_records += 1
                    continue
                res_key = (str(res_chain), int(res_num))
                res_idx = res_key_to_idx.get(res_key)
                if res_idx is None:
                    failed_records += 1
                    continue

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


def preprocess_complex(
    pdb_id,
    pdbbind_dir,
    plip_dir,
    cache_dir,
    ligand_file="ligand",
    protein_file="protein_processed",
    remove_hs=False,
    receptor_radius=None,
    chain_cutoff=None,
    cutoff=10.0,
    neg_per_pos=20,
    neg_min=10,
    neg_max=200,
    bad_ratio=0.3,
    use_training_graph=False,
    training_graph_params=None,
):
    report_path = os.path.join(plip_dir, pdb_id, "report.json")
    if not os.path.exists(report_path):
        return False, f"Missing PLIP report for {pdb_id}"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    lig_pos = get_ligand_positions(pdbbind_dir, pdb_id, ligand_file=ligand_file, remove_hs=remove_hs)
    if use_training_graph:
        training_graph_params = training_graph_params or {}
        res_pos, res_key_to_idx, res_keys = get_training_receptor_positions(
            pdbbind_dir,
            pdb_id,
            protein_file,
            training_graph_params,
        )
        receptor_radius = training_graph_params.get("receptor_radius", receptor_radius)
        chain_cutoff = training_graph_params.get("chain_cutoff", chain_cutoff)
    else:
        res_pos, res_key_to_idx, res_keys = get_receptor_positions(pdbbind_dir, pdb_id, protein_file=protein_file)
    if receptor_radius is not None:
        diff = lig_pos[:, None, :] - res_pos[None, :, :]
        min_dist = np.linalg.norm(diff, axis=-1).min(axis=0)
        keep = min_dist < receptor_radius
        if not np.any(keep):
            return False, f"No receptor residues within receptor_radius for {pdb_id}"
        res_pos = res_pos[keep]
        res_keys = [res_keys[i] for i in np.where(keep)[0]]
        res_key_to_idx = {res_key: idx for idx, res_key in enumerate(res_keys)}
    if chain_cutoff is not None:
        diff = lig_pos[:, None, :] - res_pos[None, :, :]
        min_dist = np.linalg.norm(diff, axis=-1).min(axis=0)
        keep = min_dist < chain_cutoff
        if not np.any(keep):
            return False, f"No receptor residues within chain_cutoff for {pdb_id}"
        res_pos = res_pos[keep]
        res_keys = [res_keys[i] for i in np.where(keep)[0]]
        res_key_to_idx = {res_key: idx for idx, res_key in enumerate(res_keys)}

    pos_map, pos_dist, total_records, failed_records = parse_plip_records(report, lig_pos, res_key_to_idx)
    if total_records > 0 and failed_records / total_records > bad_ratio:
        return False, f"Bad sample {pdb_id}: mapping fail ratio {failed_records}/{total_records}"

    lig_idx, res_idx = build_candidate_edges(lig_pos, res_pos, cutoff=cutoff)
    edge_index = negative_sample_edges(
        lig_idx,
        res_idx,
        pos_map,
        num_residues=res_pos.shape[0],
        neg_per_pos=neg_per_pos,
        neg_min=neg_min,
        neg_max=neg_max,
    )
    y_type, y_dist = build_edge_labels(edge_index, pos_map, pos_dist)

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{pdb_id}.pt")
    torch.save(
        {
            "lig_pos": torch.tensor(lig_pos, dtype=torch.float32),
            "res_pos": torch.tensor(res_pos, dtype=torch.float32),
            "cand_edge_index": torch.tensor(edge_index, dtype=torch.long),
            "cand_edge_y_type": torch.tensor(y_type, dtype=torch.long),
            "edge_y_dist": torch.tensor(y_dist, dtype=torch.float32),
        },
        out_path,
    )
    return True, f"Saved {out_path}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess PLIP NCI labels for PDBBind complexes.")
    parser.add_argument("--pdbbind_dir", default="data/pdbbind", help="Path to PDBBind complexes")
    parser.add_argument("--plip_dir", default="data/plip", help="Path to PLIP report directory")
    parser.add_argument("--cache_dir", default="data/cache", help="Output cache directory")
    parser.add_argument("--split_file", default=None, help="Optional split file containing PDB IDs")
    parser.add_argument("--ligand_file", default="ligand", help="Ligand file suffix")
    parser.add_argument("--protein_file", default="protein_processed", help="Protein file suffix")
    parser.add_argument("--remove_hs", action="store_true", default=False, help="Remove hydrogens from ligand")
    parser.add_argument("--receptor_radius", type=float, default=None, help="Match training receptor radius for residues")
    parser.add_argument("--chain_cutoff", type=float, default=None, help="Match training chain cutoff for receptors")
    parser.add_argument("--use_training_graph", action="store_true", default=False, help="Build receptor graph using training settings")
    parser.add_argument("--training_args", default=None, help="Optional training args file (train.py-style YAML/JSON)")
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--neg_per_pos", type=int, default=20)
    parser.add_argument("--neg_min", type=int, default=10)
    parser.add_argument("--neg_max", type=int, default=200)
    parser.add_argument("--bad_ratio", type=float, default=0.3)
    args = parser.parse_args()

    use_training_graph = args.use_training_graph or args.training_args is not None
    training_graph_params = resolve_training_graph_params(args) if use_training_graph else None

    if args.split_file:
        with open(args.split_file, "r", encoding="utf-8") as f:
            pdb_ids = [line.strip() for line in f if line.strip()]
    else:
        pdb_ids = [name for name in os.listdir(args.pdbbind_dir) if os.path.isdir(os.path.join(args.pdbbind_dir, name))]
        pdb_ids = [name for name in pdb_ids if os.path.isdir(os.path.join(args.plip_dir, name))]

    total = 0
    processed = 0
    skipped = 0
    for pdb_id in pdb_ids:
        total += 1
        try:
            ok, msg = preprocess_complex(
                pdb_id,
                args.pdbbind_dir,
                args.plip_dir,
                args.cache_dir,
                ligand_file=args.ligand_file,
                protein_file=args.protein_file,
                remove_hs=args.remove_hs,
                receptor_radius=args.receptor_radius,
                chain_cutoff=args.chain_cutoff,
                cutoff=args.cutoff,
                neg_per_pos=args.neg_per_pos,
                neg_min=args.neg_min,
                neg_max=args.neg_max,
                bad_ratio=args.bad_ratio,
                use_training_graph=use_training_graph,
                training_graph_params=training_graph_params,
            )
        except Exception as exc:
            ok, msg = False, f"Failed {pdb_id}: {exc}"
        if ok:
            processed += 1
        else:
            skipped += 1
        print(msg)

    print(f"Processed {processed}/{total} complexes, skipped {skipped}.")


if __name__ == "__main__":
    main()
