#!/usr/bin/env python3
from __future__ import annotations

import argparse

from drylab_common import collect_af3_metrics, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AF3 summary JSON metrics into per-design CSV.")
    parser.add_argument("--af3-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    write_csv(args.output, collect_af3_metrics(args.af3_root))


if __name__ == "__main__":
    main()
