#!/usr/bin/env python3
from __future__ import annotations

import argparse

from drylab_common import read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach protein-aligned NA RMSD values to AF3 metric rows.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rmsd-table", help="Optional CSV with target_id,candidate_id,rmsd")
    args = parser.parse_args()

    rows = read_csv(args.metrics)
    rmsd = {}
    if args.rmsd_table:
        for row in read_csv(args.rmsd_table):
            rmsd[(row["target_id"], row["candidate_id"])] = row.get("rmsd", "")
    for row in rows:
        row["rmsd"] = rmsd.get((row["target_id"], row["candidate_id"]), row.get("rmsd", ""))
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
