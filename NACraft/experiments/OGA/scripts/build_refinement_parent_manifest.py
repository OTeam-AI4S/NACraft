#!/usr/bin/env python3
"""Build the AF3 parent manifest for an OGA refinement batch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def final_sequence(trace: Path) -> str:
    rows = [line.rstrip("\n").split("\t") for line in trace.read_text().splitlines()[1:]]
    rows = [row for row in rows if len(row) > 4 and row[3]]
    if not rows:
        raise RuntimeError(f"no sequence rows in {trace}")
    return rows[-1][3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--group-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads(Path(args.group_manifest).read_text())
    records = []
    task_index = 0
    for group in manifest["groups"]:
        for design_index in range(group["parent_count"]):
            design_dir = root / "designs" / group["group"] / f"design{design_index}"
            structures = list(design_dir.glob("state0_sample*.cif"))
            if len(structures) != 5 or not (design_dir / "state0.pkl").exists():
                raise RuntimeError(f"incomplete parent: {design_dir}")
            records.append(
                {
                    "parent_id": f"{group['group']}_design{design_index}",
                    "sequence": final_sequence(design_dir / "optimization_trace.tsv"),
                    "group": group["group"],
                    "design_index": design_index,
                    "task_index": task_index,
                    "mode": group["mode"],
                    "source_candidate_id": group.get("source_candidate_id", ""),
                    "source_rank": group.get("source_rank", ""),
                }
            )
            task_index += 1
    if len(records) != manifest["parent_count"]:
        raise RuntimeError("parent count mismatch")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} parents to {output}")


if __name__ == "__main__":
    main()
