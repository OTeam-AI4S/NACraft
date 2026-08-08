#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from drylab_common import build_nacraft_config, read_csv, write_simple_yaml
except ModuleNotFoundError:
    from .drylab_common import build_nacraft_config, read_csv, write_simple_yaml


MODES = ("denovo", "similarity_guided")
ALL_MODES = (
    "denovo",
    "similarity_guided",
    "target_selective",
    "antibind_loss",
    "no_polymer_specific",
)


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.rglob("*.yaml"):
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NACraft dry-lab benchmark YAML configs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=ALL_MODES)
    parser.add_argument("--sequence-guidance-weight", type=float, default=0.1)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clean_output:
        clean_output_dir(output_dir)
    for target in read_csv(args.manifest):
        for mode in args.modes:
            config = build_nacraft_config(target, mode=mode, sequence_guidance_weight=args.sequence_guidance_weight)
            target_id = target.get("target_id") or target.get("benchmark_id")
            if not target_id:
                raise ValueError("Manifest row requires target_id or benchmark_id")
            path = output_dir / mode / f"{target_id}.yaml"
            write_simple_yaml(path, {key: value for key, value in config.items() if key != "loss_types"})


if __name__ == "__main__":
    main()
