#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drylab_common import compute_hotspots, parse_json_list, parse_structure_atoms, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute native interface patches and ODesign-style hotspots.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--hotspot-dir", required=True)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--top-n", type=int, default=4)
    args = parser.parse_args()

    hotspot_dir = Path(args.hotspot_dir)
    hotspot_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for target in read_csv(args.manifest):
        atoms = parse_structure_atoms(target["structure_path"], target_id=target["target_id"])
        protein_chains = set(parse_json_list(target.get("protein_chains")))
        na_chains = set(parse_json_list(target.get("na_chains")))
        protein_atoms = [atom for atom in atoms if atom.molecule_type == "protein" and (not protein_chains or atom.chain_id in protein_chains)]
        na_atoms = [atom for atom in atoms if atom.molecule_type in {"rna", "dna"} and (not na_chains or atom.chain_id in na_chains)]
        if not protein_atoms:
            protein_atoms = [atom for atom in atoms if atom.molecule_type == "protein"]
        if not na_atoms:
            na_atoms = [atom for atom in atoms if atom.molecule_type in {"rna", "dna"}]
        patch, hotspots = compute_hotspots(protein_atoms, na_atoms, cutoff=args.cutoff, top_n=args.top_n)
        (hotspot_dir / f"{target['target_id']}_patch.json").write_text(json.dumps(patch, indent=2))
        (hotspot_dir / f"{target['target_id']}_hotspots.json").write_text(json.dumps(hotspots, indent=2))
        row = dict(target)
        row["interface_patch"] = json.dumps([item["residue_key"] for item in patch])
        row["selected_hotspots"] = json.dumps([item["residue_key"] for item in hotspots])
        out_rows.append(row)
    write_csv(args.output_manifest, out_rows)


if __name__ == "__main__":
    main()
