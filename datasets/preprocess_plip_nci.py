import argparse
import json
import os

import numpy as np
import torch
import yaml
import prody as pr

from datasets.pdbbind import (
    read_mol,
    parse_plip_records,
    build_candidate_edges,
    negative_sample_edges,
    build_edge_labels,
)
from datasets.process_mols import moad_extract_receptor_structure
from utils.utils import remap_edge_index


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
    resnames = np.asarray(ca.getResnames())
    icodes = ca.getIcodes()
    if icodes is None:
        icodes = np.asarray([""] * len(resnums))
    else:
        icodes = np.asarray([icode if icode is not None else "" for icode in icodes])
    icodes = np.asarray([str(icode).strip() if str(icode).strip() else "" for icode in icodes])
    chains = np.asarray(ca.getChids())
    segnames = np.asarray(ca.getSegnames())
    res_chain_ids = np.asarray([str(seg) + str(chain) for seg, chain in zip(segnames, chains)])
    res_keys = [
        (str(chain), int(resnum), str(icode), str(resname))
        for chain, resnum, icode, resname in zip(res_chain_ids, resnums, icodes, resnames)
    ]
    return res_pos, res_chain_ids, resnums, res_keys


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
):
    report_path = os.path.join(plip_dir, pdb_id, "report.json")
    if not os.path.exists(report_path):
        return False, f"Missing PLIP report for {pdb_id}"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    lig_pos = get_ligand_positions(pdbbind_dir, pdb_id, ligand_file=ligand_file, remove_hs=remove_hs)
    res_pos_full, res_chain_ids_full, resnums_full, res_keys_full = get_receptor_positions(
        pdbbind_dir,
        pdb_id,
        protein_file=protein_file,
    )
    keep = np.ones(len(res_pos_full), dtype=bool)
    if receptor_radius is not None:
        diff = lig_pos[:, None, :] - res_pos_full[None, :, :]
        min_dist = np.linalg.norm(diff, axis=-1).min(axis=0)
        keep &= min_dist < receptor_radius
        if not np.any(keep):
            return False, f"No receptor residues within receptor_radius for {pdb_id}"
    if chain_cutoff is not None:
        diff = lig_pos[:, None, :] - res_pos_full[None, :, :]
        min_dist = np.linalg.norm(diff, axis=-1).min(axis=0)
        keep &= min_dist < chain_cutoff
        if not np.any(keep):
            return False, f"No receptor residues within chain_cutoff for {pdb_id}"
    res_pos = res_pos_full[keep]
    res_chain_ids = res_chain_ids_full[keep]
    resnums = resnums_full[keep]
    res_keys = [res_keys_full[idx] for idx, flag in enumerate(keep) if flag]

    protein_center = res_pos.mean(axis=0, keepdims=False).astype(np.float32)
    res_pos = res_pos - protein_center
    res_pos_full = res_pos_full - protein_center
    lig_pos = lig_pos - protein_center

    res_key_to_full_idx = {tuple(res_key): idx for idx, res_key in enumerate(res_keys_full)}
    (
        pos_map,
        pos_dist,
        total_records,
        failed_records,
        filtered_records,
        missing_icode_records,
        fallback_records,
    ) = parse_plip_records(
        report,
        lig_pos,
        res_pos_full,
        center=protein_center,
        res_keys_full=res_keys_full,
        res_key_to_full_idx=res_key_to_full_idx,
    )
    failed_total = failed_records + filtered_records
    if total_records > 0 and failed_total / total_records > bad_ratio:
        return False, f"Bad sample {pdb_id}: mapping fail ratio {failed_total}/{total_records}"

    lig_idx, res_idx = build_candidate_edges(lig_pos, res_pos_full, cutoff=cutoff)
    edge_index = negative_sample_edges(
        lig_idx,
        res_idx,
        pos_map,
        num_residues=res_pos_full.shape[0],
        neg_per_pos=neg_per_pos,
        neg_min=neg_min,
        neg_max=neg_max,
    )
    y_type, y_dist = build_edge_labels(edge_index, pos_map, pos_dist)
    if not np.all(keep):
        old_to_new = torch.full((len(keep),), -1, dtype=torch.long)
        if keep.any():
            old_to_new[torch.from_numpy(keep)] = torch.arange(int(keep.sum()))
        edge_index, valid_edges = remap_edge_index(torch.as_tensor(edge_index, dtype=torch.long), old_to_new)
        edge_index = edge_index.cpu().numpy()
        if valid_edges is not None:
            y_type = y_type[valid_edges.cpu().numpy()]
            y_dist = y_dist[valid_edges.cpu().numpy()]

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{pdb_id}.pt")
    torch.save(
        {
            "lig_pos": torch.tensor(lig_pos, dtype=torch.float32),
            "res_pos": torch.tensor(res_pos, dtype=torch.float32),
            "res_chain_ids": res_chain_ids.tolist(),
            "resnums": resnums.astype(int).tolist(),
            "res_keys": res_keys,
            "res_id_map": res_key_to_full_idx,
            "plip_total_records": int(total_records),
            "plip_failed_records": int(failed_records),
            "plip_filtered_records": int(filtered_records),
            "plip_missing_icode_records": int(missing_icode_records),
            "plip_fallback_records": int(fallback_records),
            "original_center": torch.tensor(protein_center, dtype=torch.float32),
            "cand_edge_index": torch.tensor(edge_index, dtype=torch.long),
            "cand_edge_y_type": torch.tensor(y_type, dtype=torch.long),
            "edge_y_dist": torch.tensor(y_dist, dtype=torch.float32),
        },
        out_path,
    )
    return True, (
        f"Saved {out_path} (cropped PLIP interactions: {filtered_records}/{total_records})"
    )


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
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--neg_per_pos", type=int, default=20)
    parser.add_argument("--neg_min", type=int, default=10)
    parser.add_argument("--neg_max", type=int, default=200)
    parser.add_argument("--bad_ratio", type=float, default=0.3)
    args = parser.parse_args()

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
