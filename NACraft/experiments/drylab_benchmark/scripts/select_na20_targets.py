#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    from drylab_common import ODESIGN_VISIBLE_DNA, ODESIGN_VISIBLE_RNA, read_csv, write_csv
except ModuleNotFoundError:
    from .drylab_common import ODESIGN_VISIBLE_DNA, ODESIGN_VISIBLE_RNA, read_csv, write_csv


RNA_BINS = [(10, 30, 2), (31, 50, 2), (51, 75, 2), (76, 100, 2), (101, 150, 2)]
DNA_BINS = [(5, 20, 2), (21, 40, 2), (41, 60, 2), (61, 80, 2), (81, 100, 2)]


def protein_length(row: dict[str, str]) -> int:
    return len(row.get("protein_sequence", "").replace(":", ""))


def is_single_protein_chain(row: dict[str, str]) -> bool:
    if ":" in row.get("protein_sequence", ""):
        return False
    try:
        return len(json.loads(row.get("protein_chains", "[]"))) == 1
    except json.JSONDecodeError:
        return True


def select_bins(
    rows: list[dict[str, str]],
    bins: list[tuple[int, int, int]],
    used_target_ids: set[str],
    used_protein_sequences: set[str],
    rng: random.Random,
) -> list[dict[str, str]]:
    selected = []
    for low, high, count in bins:
        pool = [
            row
            for row in rows
            if row["target_id"] not in used_target_ids
            and row["protein_sequence"] not in used_protein_sequences
            and low <= int(float(row["na_length"])) <= high
        ]
        rng.shuffle(pool)
        picked = []
        for row in pool:
            if len(picked) >= count:
                break
            if row["target_id"] in used_target_ids or row["protein_sequence"] in used_protein_sequences:
                continue
            out = dict(row)
            out["selection_reason"] = f"length_bin_{low}_{high}"
            picked.append(out)
            used_target_ids.add(out["target_id"])
            used_protein_sequences.add(out["protein_sequence"])
        missing = count - len(picked)
        if missing > 0:
            center = (low + high) // 2
            fallback = [
                row
                for row in rows
                if row["target_id"] not in used_target_ids
                and row["protein_sequence"] not in used_protein_sequences
            ]
            fallback_by_distance: dict[int, list[dict[str, str]]] = {}
            for row in fallback:
                distance = abs(int(float(row["na_length"])) - center)
                fallback_by_distance.setdefault(distance, []).append(row)
            for distance in sorted(fallback_by_distance):
                group = fallback_by_distance[distance]
                rng.shuffle(group)
                for row in group:
                    if missing <= 0:
                        break
                    out = dict(row)
                    out["selection_reason"] = f"adjacent_fill_for_{low}_{high}"
                    picked.append(out)
                    used_target_ids.add(out["target_id"])
                    used_protein_sequences.add(out["protein_sequence"])
                    missing -= 1
                if missing <= 0:
                    break
        selected.extend(picked)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select NA-20 targets with global protein-length and sequence constraints.")
    parser.add_argument("--rna-candidates", required=True)
    parser.add_argument("--dna-candidates", required=True)
    parser.add_argument("--output-rna", required=True)
    parser.add_argument("--output-dna", required=True)
    parser.add_argument("--output-na20", required=True)
    parser.add_argument("--max-protein-length", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=20260713)
    args = parser.parse_args()

    rng = random.Random(args.random_seed)
    rna_rows = [
        row
        for row in read_csv(args.rna_candidates)
        if row["target_id"] not in ODESIGN_VISIBLE_RNA
        and protein_length(row) <= args.max_protein_length
        and is_single_protein_chain(row)
    ]
    dna_rows = [
        row
        for row in read_csv(args.dna_candidates)
        if row["target_id"] not in ODESIGN_VISIBLE_DNA
        and protein_length(row) <= args.max_protein_length
        and is_single_protein_chain(row)
    ]

    used_target_ids: set[str] = set()
    used_protein_sequences: set[str] = set()
    rna_selected = select_bins(rna_rows, RNA_BINS, used_target_ids, used_protein_sequences, rng)
    dna_selected = select_bins(dna_rows, DNA_BINS, used_target_ids, used_protein_sequences, rng)
    combined = rna_selected + dna_selected

    for path, rows in (
        (args.output_rna, rna_selected),
        (args.output_dna, dna_selected),
        (args.output_na20, combined),
    ):
        write_csv(path, rows)

    print(f"RNA targets: {len(rna_selected)}")
    print(f"DNA targets: {len(dna_selected)}")
    print(f"NA targets: {len(combined)}")
    print(f"unique protein sequences: {len({row['protein_sequence'] for row in combined})}")
    print(f"max protein length: {max(protein_length(row) for row in combined)}")
    print(f"outputs: {Path(args.output_na20)}")


if __name__ == "__main__":
    main()
