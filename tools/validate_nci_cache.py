import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.diffusion_utils import t_to_sigma


def _load_cache_entry(cache_dir, complex_name):
    if not cache_dir or not complex_name:
        return None
    candidates = [complex_name, complex_name[:4], complex_name[:6]]
    for candidate in dict.fromkeys(candidates):
        for ext in (".pt", ".npz"):
            path = os.path.join(cache_dir, f"{candidate}{ext}")
            if not os.path.exists(path):
                continue
            if ext == ".pt":
                return torch.load(path, map_location="cpu")
            data = np.load(path, allow_pickle=True)
            return {key: data[key] for key in data.files}
    return None


def _compute_nci_mapping_stats(graph, cache_entry):
    if cache_entry is None:
        return None
    cache_res_chain_ids = cache_entry.get("res_chain_ids")
    cache_resnums = cache_entry.get("resnums")
    if cache_res_chain_ids is None or cache_resnums is None:
        return None

    if isinstance(cache_res_chain_ids, np.ndarray):
        cache_res_chain_ids = cache_res_chain_ids.tolist()
    if isinstance(cache_resnums, np.ndarray):
        cache_resnums = cache_resnums.tolist()
    cache_res_chain_ids = [str(chain) for chain in cache_res_chain_ids]
    cache_resnums = [int(resnum) for resnum in cache_resnums]
    if len(cache_resnums) == 0:
        return None

    curr_resnums = graph['receptor'].resnums
    curr_res_chain_ids = graph['receptor'].res_chain_ids
    if torch.is_tensor(curr_resnums):
        curr_resnums = curr_resnums.cpu().numpy()
    if torch.is_tensor(curr_res_chain_ids):
        curr_res_chain_ids = curr_res_chain_ids.cpu().numpy()
    curr_res_chain_ids = [str(chain) for chain in curr_res_chain_ids]
    curr_resnums = [int(resnum) for resnum in curr_resnums]
    num_rec = len(curr_resnums)
    if num_rec == 0:
        return None

    current_map = {
        (chain, resnum): idx
        for idx, (chain, resnum) in enumerate(zip(curr_res_chain_ids, curr_resnums))
    }
    remap = torch.full((len(cache_resnums),), -1, dtype=torch.long)
    for idx, (chain, resnum) in enumerate(zip(cache_res_chain_ids, cache_resnums)):
        mapped = current_map.get((chain, resnum))
        if mapped is not None:
            remap[idx] = mapped
    mapped_mask = remap >= 0
    hit_count = int(mapped_mask.sum().item())
    hit_rate = hit_count / len(cache_resnums)
    unique_mapped = int(torch.unique(remap[mapped_mask]).numel())
    coverage_rate = unique_mapped / num_rec
    return {
        "hit_rate": hit_rate,
        "hit_count": hit_count,
        "cache_res": len(cache_resnums),
        "coverage_rate": coverage_rate,
        "coverage_count": unique_mapped,
        "graph_res": num_rec,
    }


def check_graph(graph, idx=None, cache_dir=None, report_nci_mapping=False):
    if ('ligand', 'nci_cand', 'receptor') not in graph.edge_types:
        return True

    edge = graph['ligand', 'nci_cand', 'receptor']
    if not hasattr(edge, 'edge_index'):
        return True

    edge_index = edge.edge_index
    num_lig = graph['ligand'].num_nodes
    num_rec = graph['receptor'].num_nodes

    bad_lig = (edge_index[0] < 0) | (edge_index[0] >= num_lig)
    bad_rec = (edge_index[1] < 0) | (edge_index[1] >= num_rec)
    if bad_lig.any() or bad_rec.any():
        print(
            "[BAD edge_index] graph={idx} num_lig={num_lig} num_rec={num_rec} "
            "bad_lig={bad_lig} bad_rec={bad_rec}".format(
                idx=idx,
                num_lig=num_lig,
                num_rec=num_rec,
                bad_lig=bad_lig.nonzero().flatten().tolist(),
                bad_rec=bad_rec.nonzero().flatten().tolist(),
            )
        )
        return False

    if hasattr(edge, 'edge_type_y'):
        labels = edge.edge_type_y
        if (labels < 0).any() or (labels >= 8).any():
            bad = ((labels < 0) | (labels >= 8)).nonzero().flatten().tolist()
            print(
                "[BAD labels] graph={idx} bad_idx={bad} min={minv} max={maxv}".format(
                    idx=idx,
                    bad=bad,
                    minv=int(labels.min()),
                    maxv=int(labels.max()),
                )
            )
            return False

    if cache_dir is not None:
        complex_name = graph.get('name', None)
        cache_entry = _load_cache_entry(cache_dir, complex_name)
        if cache_entry is not None and 'lig_pos' in cache_entry:
            lig_pos = cache_entry['lig_pos']
            num_lig_cache = lig_pos.shape[0]
            num_lig_graph = graph['ligand'].num_nodes
            if num_lig_cache != num_lig_graph:
                print(
                    "[MISMATCH ligand size] graph={idx} name={name} cache={cache} "
                    "lig_pos={cache_lig} graph_lig={graph_lig} (remove_hs mismatch?)".format(
                        idx=idx,
                        name=complex_name,
                        cache=cache_dir,
                        cache_lig=num_lig_cache,
                        graph_lig=num_lig_graph,
                    )
                )
                return False
        if report_nci_mapping and cache_entry is not None:
            mapping_stats = _compute_nci_mapping_stats(graph, cache_entry)
            if mapping_stats is not None:
                print(
                    "[NCI mapping] graph={idx} name={name} hit_rate={hit:.3f} ({hit_count}/{cache}) "
                    "coverage={cov:.3f} ({coverage_count}/{graph_res})".format(
                        idx=idx,
                        name=complex_name,
                        hit=mapping_stats["hit_rate"],
                        hit_count=mapping_stats["hit_count"],
                        cache=mapping_stats["cache_res"],
                        cov=mapping_stats["coverage_rate"],
                        coverage_count=mapping_stats["coverage_count"],
                        graph_res=mapping_stats["graph_res"],
                    )
                )

    return True


def validate_dataset(dataset, name, cache_dir=None, report_nci_mapping=False):
    bad = 0
    total = len(dataset)
    for i in range(total):
        graph = dataset.get(i)
        if not check_graph(graph, idx=i, cache_dir=cache_dir, report_nci_mapping=report_nci_mapping):
            bad += 1
    print("{name}: {bad}/{total} graphs failed NCI cache checks.".format(
        name=name, bad=bad, total=total
    ))


def main():
    args = parse_train_args()
    device = torch.device("cpu")
    train_loader, val_loader, _ = construct_loader(args, t_to_sigma, device)
    cache_dir = getattr(args, "nci_cache_path", None)

    print("Validating train dataset (remove_hs={}).".format(args.remove_hs))
    validate_dataset(
        train_loader.dataset,
        "train",
        cache_dir=cache_dir,
        report_nci_mapping=args.report_nci_mapping,
    )

    if val_loader is not None:
        print("Validating val dataset (remove_hs={}).".format(args.remove_hs))
        validate_dataset(
            val_loader.dataset,
            "val",
            cache_dir=cache_dir,
            report_nci_mapping=args.report_nci_mapping,
        )


if __name__ == "__main__":
    main()
