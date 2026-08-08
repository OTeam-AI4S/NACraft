#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def sanitize_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AF3 presearch inputs for conf-sel paired-state targets.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique: dict[str, dict[str, str]] = {}
    aliases: dict[str, list[str]] = {}
    for row in rows:
        for role, seq_key, id_key in (
            ("positive", "positive_protein_sequence", "positive_target_id"),
            ("negative", "negative_protein_sequence", "negative_target_id"),
        ):
            seq = row[seq_key]
            alias = f"{row['target_id']}:{role}:{row[id_key]}"
            if seq not in unique:
                unique[seq] = {
                    "role": role,
                    "target_id": row[id_key],
                    "specificity_class": row["specificity_class"],
                }
                aliases[seq] = []
            aliases[seq].append(alias)

    entries = []
    manifest_rows = []
    fasta_lines = []
    for idx, (seq, meta) in enumerate(unique.items(), start=1):
        alias_text = "_".join(aliases[seq])
        name = f"confsel_{idx:02d}_{sanitize_name(meta['target_id'])}_{len(seq)}aa_presearch"
        entries.append(
            {
                "name": name,
                "sequences": [{"protein": {"id": ["A"], "sequence": seq}}],
                "modelSeeds": [1],
                "dialect": "alphafold3",
                "version": 1,
            }
        )
        manifest_rows.append(
            {
                "presearch_name": name,
                "target_aliases": ";".join(aliases[seq]),
                "protein_length": len(seq),
                "protein_sequence": seq,
                "specificity_class": meta["specificity_class"],
            }
        )
        fasta_lines.append(f">{name}|targets={alias_text}|len={len(seq)}")
        fasta_lines.append(seq)

    (out_dir / "presearch_input.json").write_text(json.dumps(entries, indent=2) + "\n")
    with (out_dir / "presearch_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "presearch_name",
                "target_aliases",
                "protein_length",
                "protein_sequence",
                "specificity_class",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_dir / "presearch_targets.fasta").write_text("\n".join(fasta_lines) + "\n")

    print(f"wrote {len(entries)} unique conf-sel protein target(s) to {out_dir}")


if __name__ == "__main__":
    main()
