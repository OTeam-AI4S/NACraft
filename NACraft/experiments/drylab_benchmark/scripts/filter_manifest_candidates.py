#!/usr/bin/env python3
from __future__ import annotations

import argparse

from drylab_common import filter_manifest_candidates, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter RCSB manifest candidates for tractable NACraft dry-lab targets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-na-length", type=int, default=1)
    parser.add_argument("--max-na-length", type=int, default=150)
    parser.add_argument("--max-protein-length", type=int, default=2000)
    parser.add_argument("--max-protein-chains", type=int, default=4)
    parser.add_argument("--max-na-chains", type=int, default=1)
    parser.add_argument("--allow-ambiguous-sequence", action="store_true")
    args = parser.parse_args()

    rows = filter_manifest_candidates(
        read_csv(args.input),
        min_na_length=args.min_na_length,
        max_na_length=args.max_na_length,
        max_protein_length=args.max_protein_length,
        max_protein_chains=args.max_protein_chains,
        max_na_chains=args.max_na_chains,
        require_unambiguous_sequence=not args.allow_ambiguous_sequence,
    )
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
