#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from drylab_common import read_csv, write_csv
except ModuleNotFoundError:
    from .drylab_common import read_csv, write_csv


def expand_manifest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        base = {
            "source_target_id": row["target_id"],
            "polymer_type": row["polymer_type"],
            "na_length": row["na_length"],
            "benchmark_id": row.get("benchmark_id", ""),
            "specificity_class": row.get("specificity_class", ""),
            "experiment_tier": row.get("experiment_tier", ""),
        }
        out.append(
            {
                **base,
                "target_id": f"{row['target_id']}__positive",
                "state": "positive",
                "protein_sequence": row["positive_protein_sequence"],
                "state_target_id": row["positive_target_id"],
                "structure_path": row.get("positive_structure_path", ""),
            }
        )
        out.append(
            {
                **base,
                "target_id": f"{row['target_id']}__negative",
                "state": "negative",
                "protein_sequence": row["negative_protein_sequence"],
                "state_target_id": row["negative_target_id"],
                "structure_path": row.get("negative_structure_path", ""),
            }
        )
    return out


def expand_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        for state in ("positive", "negative"):
            candidate = dict(row)
            candidate["source_target_id"] = row["target_id"]
            candidate["target_id"] = f"{row['target_id']}__{state}"
            candidate["candidate_id"] = f"{row['candidate_id']}__{state}"
            candidate["validation_state"] = state
            out.append(candidate)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand target-selective candidates into positive/negative AF3 inputs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-candidates", required=True)
    args = parser.parse_args()

    write_csv(args.output_manifest, expand_manifest(read_csv(args.manifest)))
    candidate_rows = expand_candidates(read_csv(args.candidates))
    fieldnames = [
        "target_id",
        "method",
        "candidate_id",
        "sequence",
        "source_design",
        "variant",
        "variant_index",
        "source_fasta",
        "source_target_id",
        "validation_state",
    ]
    Path(args.output_candidates).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_candidates).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Wrote {len(candidate_rows)} expanded candidates to {args.output_candidates}")


if __name__ == "__main__":
    main()
