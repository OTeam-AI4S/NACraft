#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from drylab_common import manifest_qc_summary, read_csv, write_csv


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["polymer_type"]) for row in rows)
    lines = [
        "# NACraft Dry-Lab Target QC",
        "",
        f"Targets: {len(rows)}",
        f"RNA targets: {counts.get('rna', 0)}",
        f"DNA targets: {counts.get('dna', 0)}",
        "",
        "| target | type | length | hotspots | patch | structure | selection |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        structure = "yes" if row["structure_exists"] else "no"
        lines.append(
            f"| {row['target_id']} | {row['polymer_type']} | {row['na_length']} | "
            f"{row['hotspot_count']} | {row['interface_patch_size']} | {structure} | {row['selection_reason']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize target manifest QC before dry-lab job submission.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows = manifest_qc_summary(read_csv(args.manifest))
    write_csv(args.output_csv, rows)
    write_markdown(Path(args.output_md), rows)


if __name__ == "__main__":
    main()
