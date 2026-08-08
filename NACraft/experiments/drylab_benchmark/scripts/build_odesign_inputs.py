#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from drylab_common import parse_json_list, read_csv
except ModuleNotFoundError:
    from .drylab_common import parse_json_list, read_csv


PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def protein_residue_ranges(pdb_path: str | Path, protein_chains: list[str]) -> dict[str, tuple[int, int]]:
    ranges: dict[str, list[int]] = {chain: [] for chain in protein_chains}
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain_id = line[21].strip() or "_"
        residue_name = line[17:20].strip()
        if chain_id not in ranges or residue_name not in PROTEIN_RESIDUES:
            continue
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        ranges[chain_id].append(residue_number)
    out = {}
    for chain_id, residue_numbers in ranges.items():
        if not residue_numbers:
            continue
        out[chain_id] = (min(residue_numbers), max(residue_numbers))
    return out


def write_protein_only_pdb(input_pdb: str | Path, output_pdb: str | Path, protein_chains: list[str]) -> None:
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    keep = set(protein_chains)
    lines = []
    for line in Path(input_pdb).read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM", "ANISOU")):
            chain_id = line[21].strip() or "_"
            residue_name = line[17:20].strip()
            if chain_id in keep and residue_name in PROTEIN_RESIDUES:
                lines.append(line)
        elif line.startswith(("TER", "END")):
            continue
    lines.append("END")
    output_pdb.write_text("\n".join(lines) + "\n")


def build_odesign_input(target: dict[str, str], protein_pdb: str) -> list[dict[str, object]]:
    protein_chains = parse_json_list(target["protein_chains"])
    ranges = protein_residue_ranges(protein_pdb, protein_chains)
    chains = []
    for chain_id in protein_chains:
        if chain_id not in ranges:
            continue
        start, end = ranges[chain_id]
        chains.append({"chain_type": "proteinChain", "sequence": f"{chain_id}/{start}-{end}"})
    polymer_type = target["polymer_type"].lower()
    na_chain_type = "rnaChain" if polymer_type == "rna" else "dnaChain"
    length = int(float(target["na_length"]))
    chains.append({"chain_type": na_chain_type, "sequence": f"{length}-{length}"})
    hotspots = []
    for item in parse_json_list(target.get("selected_hotspots") or target.get("hotspots")):
        parts = item.split(":")
        if len(parts) >= 2:
            hotspots.append(f"{parts[0]}/{parts[1]}")
    return [
        {
            "name": f"{target['target_id']}_{polymer_type}_odesign",
            "ref_file": str(protein_pdb),
            "chains": chains,
            "hotspot": ",".join(hotspots),
            "center_method": "hotspot_center",
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protein-only ODesign NA input JSONs from the drylab manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--protein-pdb-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    protein_pdb_dir = Path(args.protein_pdb_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    protein_pdb_dir.mkdir(parents=True, exist_ok=True)
    for target in read_csv(args.manifest):
        target_id = target["target_id"]
        protein_chains = parse_json_list(target["protein_chains"])
        protein_pdb = protein_pdb_dir / f"{target_id}_protein_only.pdb"
        write_protein_only_pdb(target["structure_path"], protein_pdb, protein_chains)
        payload = build_odesign_input(target, str(protein_pdb))
        (input_dir / f"{target_id}.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
