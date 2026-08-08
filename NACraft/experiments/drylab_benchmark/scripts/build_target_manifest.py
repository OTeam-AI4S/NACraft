#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from drylab_common import build_target_manifest_from_structures, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw target manifest from local PDB structure files.")
    parser.add_argument("--structures-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--release-date", default="")
    args = parser.parse_args()

    paths = sorted(Path(args.structures_dir).glob("*.pdb")) + sorted(Path(args.structures_dir).glob("*.ent"))
    rows = build_target_manifest_from_structures(paths, release_date=args.release_date)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
