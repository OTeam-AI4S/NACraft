#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterator


def sanitize_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def iter_target_sequences(row: dict[str, str]) -> Iterator[tuple[str, str]]:
    if row.get("protein_sequence"):
        yield row.get("target_id") or row.get("name") or "target", row["protein_sequence"]
    elif row.get("sequence"):
        yield row.get("target_id") or row.get("name") or "target", row["sequence"]

    if row.get("positive_protein_sequence"):
        yield (
            row.get("positive_target_id") or f"{row.get('target_id', 'target')}_positive",
            row["positive_protein_sequence"],
        )
    if row.get("negative_protein_sequence"):
        yield (
            row.get("negative_target_id") or f"{row.get('target_id', 'target')}_negative",
            row["negative_protein_sequence"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AF3 presearch inputs from a NACraft drylab manifest.")
    parser.add_argument("--manifest", required=True, action="append")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = []
    for manifest in args.manifest:
        with Path(manifest).open() as handle:
            rows.extend(csv.DictReader(handle))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique: dict[str, dict[str, str]] = {}
    aliases: dict[str, list[str]] = {}
    for row in rows:
        for target_id, seq in iter_target_sequences(row):
            seq = seq.strip().upper()
            if not seq:
                continue
            if seq not in unique:
                unique[seq] = row
                aliases[seq] = []
            if target_id not in aliases[seq]:
                aliases[seq].append(target_id)

    entries = []
    manifest_rows = []
    fasta_lines = []
    for idx, seq in enumerate(unique, start=1):
        alias = sanitize_name(aliases[seq][0])[:40]
        name = f"target_{idx:02d}_{alias}_{len(seq)}aa_presearch"
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
                "target_ids": ";".join(aliases[seq]),
                "protein_length": len(seq),
                "protein_sequence": seq,
            }
        )
        fasta_lines.append(f">{name}|targets={';'.join(aliases[seq])}|len={len(seq)}")
        fasta_lines.append(seq)

    (out_dir / "presearch_input.json").write_text(json.dumps(entries, indent=2) + "\n")
    with (out_dir / "presearch_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["presearch_name", "target_ids", "protein_length", "protein_sequence"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_dir / "presearch_targets.fasta").write_text("\n".join(fasta_lines) + "\n")

    print(f"wrote {len(entries)} unique protein target(s) to {out_dir}")


if __name__ == "__main__":
    main()
