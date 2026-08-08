#!/usr/bin/env python3
from __future__ import annotations

import argparse

from drylab_common import ODESIGN_VISIBLE_DNA, ODESIGN_VISIBLE_RNA, read_csv, select_length_stratified_targets, write_csv


RNA_BINS = [(10, 30, 2), (31, 50, 2), (51, 75, 2), (76, 100, 2), (101, 150, 2)]
DNA_BINS = [(5, 20, 2), (21, 40, 2), (41, 60, 2), (61, 80, 2), (81, 100, 2)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select ODesign-style length-stratified NACraft targets.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--polymer-type", choices=["rna", "dna"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-visible", action="store_true")
    parser.add_argument("--exclude-target-id", action="append", default=[])
    parser.add_argument("--max-protein-length", type=int, default=None)
    parser.add_argument("--unique-protein-sequence", action="store_true")
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    visible = ODESIGN_VISIBLE_RNA if args.polymer_type == "rna" else ODESIGN_VISIBLE_DNA
    exclude = set(args.exclude_target_id)
    if not args.include_visible:
        exclude |= visible
    rows = read_csv(args.manifest)
    selected = select_length_stratified_targets(
        rows,
        polymer_type=args.polymer_type,
        bins=RNA_BINS if args.polymer_type == "rna" else DNA_BINS,
        exclude_target_ids=exclude,
        max_protein_length=args.max_protein_length,
        unique_protein_sequence=args.unique_protein_sequence,
        random_seed=args.random_seed,
    )
    write_csv(args.output, selected)


if __name__ == "__main__":
    main()
