#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from drylab_common import write_csv
except ModuleNotFoundError:
    from .drylab_common import write_csv


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def read_seqres(path: Path, chain_id: str) -> str:
    residues: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("SEQRES"):
            continue
        parts = line.split()
        if len(parts) < 5 or parts[2] != chain_id:
            continue
        for residue in parts[4:]:
            if residue not in AA3_TO_1:
                raise ValueError(f"Unsupported residue {residue} in {path.name} chain {chain_id}")
            residues.append(AA3_TO_1[residue])
    if not residues:
        raise ValueError(f"No SEQRES found for {path.name} chain {chain_id}")
    return "".join(residues)


def slice_sequence(sequence: str, start: int | None = None, end: int | None = None) -> str:
    if start is None and end is None:
        return sequence
    left = 0 if start is None else start - 1
    right = len(sequence) if end is None else end
    return sequence[left:right]


def add_pair(
    rows: list[dict[str, object]],
    *,
    benchmark_id: str,
    specificity_class: str,
    design_length: int,
    experiment_tier: str,
    positive_id: str,
    positive_sequence: str,
    positive_structure_path: Path,
    negative_id: str,
    negative_sequence: str,
    negative_structure_path: Path,
    notes: str,
) -> None:
    for polymer_type in ("rna", "dna"):
        target_id = f"{benchmark_id}_L{design_length}_{polymer_type}"
        rows.append(
            {
                "target_id": target_id,
                "benchmark_id": benchmark_id,
                "specificity_class": specificity_class,
                "experiment_tier": experiment_tier,
                "polymer_type": polymer_type,
                "na_length": design_length,
                "positive_target_id": positive_id,
                "negative_target_id": negative_id,
                "positive_protein_sequence": positive_sequence,
                "negative_protein_sequence": negative_sequence,
                "positive_structure_path": str(positive_structure_path),
                "negative_structure_path": str(negative_structure_path),
                "selected_hotspots": "",
                "notes": notes,
            }
        )


def build_manifest(pdb_dir: Path) -> list[dict[str, object]]:
    pdb = {
        "1IVO": pdb_dir / "1IVO.pdb",
        "1NQL": pdb_dir / "1NQL.pdb",
        "1YY9": pdb_dir / "1YY9.pdb",
        "1N8Z": pdb_dir / "1N8Z.pdb",
    }
    missing = [str(path) for path in pdb.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PDB files: " + ", ".join(missing))

    egfr_holo = slice_sequence(read_seqres(pdb["1IVO"], "A"), 1, 621)
    egfr_apo_1nql = slice_sequence(read_seqres(pdb["1NQL"], "A"), 1, 621)
    egfr_tethered_1yy9 = slice_sequence(read_seqres(pdb["1YY9"], "A"), 1, 621)
    egfr_domain_iii = slice_sequence(read_seqres(pdb["1YY9"], "A"), 310, 480)
    her2_domain_iii = slice_sequence(read_seqres(pdb["1N8Z"], "C"), 321, 488)

    rows: list[dict[str, object]] = []
    for length in (40, 50, 60):
        tier = "main" if length == 50 else "length_sensitivity"
        add_pair(
            rows,
            benchmark_id="egfr_holo_vs_apo_1nql",
            specificity_class="strict_conformation_selective",
            design_length=length,
            experiment_tier=tier,
            positive_id="1IVO_A_1_621",
            positive_sequence=egfr_holo,
            positive_structure_path=pdb["1IVO"],
            negative_id="1NQL_A_1_621",
            negative_sequence=egfr_apo_1nql,
            negative_structure_path=pdb["1NQL"],
            notes="EGF-bound EGFR ECD positive; unactivated EGFR ECD negative; both trimmed to residues 1-621 to remove construct tails.",
        )
        add_pair(
            rows,
            benchmark_id="egfr_holo_vs_tethered_1yy9",
            specificity_class="strict_conformation_selective",
            design_length=length,
            experiment_tier=tier,
            positive_id="1IVO_A_1_621",
            positive_sequence=egfr_holo,
            positive_structure_path=pdb["1IVO"],
            negative_id="1YY9_A_1_621",
            negative_sequence=egfr_tethered_1yy9,
            negative_structure_path=pdb["1YY9"],
            notes="EGF-bound EGFR ECD positive; cetuximab-bound tethered EGFR ECD negative after ignoring Fab chains.",
        )
    for length in (30, 40, 50):
        tier = "main" if length == 40 else "length_sensitivity"
        add_pair(
            rows,
            benchmark_id="egfr_domainIII_vs_her2_domainIII",
            specificity_class="paralog_specific",
            design_length=length,
            experiment_tier=tier,
            positive_id="1YY9_A_310_480",
            positive_sequence=egfr_domain_iii,
            positive_structure_path=pdb["1YY9"],
            negative_id="1N8Z_C_321_488",
            negative_sequence=her2_domain_iii,
            negative_structure_path=pdb["1N8Z"],
            notes="EGFR domain III positive; HER2 domain III negative. This is paralog-specific discrimination, not strict conformation selection.",
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated NACraft paired-state drylab manifest.")
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_manifest(Path(args.pdb_dir))
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
