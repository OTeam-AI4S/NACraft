#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from drylab_common import best_by_method_target, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare lightweight figure source tables for the NACraft manuscript.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = list(best_by_method_target(read_csv(args.metrics)).values())
    write_csv(output_dir / "figure2_best_by_target_method.csv", best)


if __name__ == "__main__":
    main()
