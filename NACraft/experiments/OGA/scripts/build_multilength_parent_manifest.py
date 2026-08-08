#!/usr/bin/env python3
"""Build an AF3 parent manifest from completed multilength designs."""

import argparse
import csv
from pathlib import Path

GROUPS = ("denovo_20", "denovo_40", "a3_guided_41", "denovo_60")


def final_sequence(trace: Path) -> str:
    rows = [line.rstrip("\n").split("\t") for line in trace.read_text().splitlines()[1:]]
    rows = [row for row in rows if len(row) > 4 and row[3]]
    if not rows:
        raise RuntimeError(f"no sequence rows in {trace}")
    return rows[-1][3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--parents-per-group", type=int, default=60)
    parser.add_argument("--min-task-index", type=int, default=0)
    parser.add_argument("--max-task-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for task_index in range(args.min_task_index, args.max_task_index + 1):
        group_index, design_index = divmod(task_index, args.parents_per_group)
        group = GROUPS[group_index]
        design_dir = root / "designs" / group / f"design{design_index}"
        structures = list(design_dir.glob("state0_sample*.cif"))
        if len(structures) != 5 or not (design_dir / "state0.pkl").exists():
            raise RuntimeError(f"incomplete task {task_index}: {design_dir}")
        records.append({
            "parent_id": f"{group}_design{design_index}",
            "sequence": final_sequence(design_dir / "optimization_trace.tsv"),
            "group": group,
            "design_index": design_index,
            "task_index": task_index,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} parents to {output}")


if __name__ == "__main__":
    main()
