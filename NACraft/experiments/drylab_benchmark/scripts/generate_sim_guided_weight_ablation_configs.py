#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from drylab_common import build_nacraft_config, read_csv, write_simple_yaml
except ModuleNotFoundError:
    from .drylab_common import build_nacraft_config, read_csv, write_simple_yaml


def weight_label(weight: float) -> str:
    return f"sim_guided_w{int(round(weight * 100)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NACraft similarity-guidance weight ablation configs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[round(i / 5, 1) for i in range(6)],
        help="SequenceSimilarityLoss strengths to test.",
    )
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clean_output and output_dir.exists():
        for path in output_dir.rglob("*.yaml"):
            path.unlink()

    rows = []
    for target in read_csv(args.manifest):
        target_id = target.get("target_id") or target.get("benchmark_id")
        if not target_id:
            raise ValueError("Manifest row requires target_id or benchmark_id")
        for weight in args.weights:
            config = build_nacraft_config(
                target,
                mode="similarity_guided",
                sequence_guidance_weight=weight,
            )
            config["method"] = f"NACraft-sim-guided-w{weight:.1f}"
            config["ablation"] = "sim_guided_similarity_weight"
            config["similarity_weight"] = weight
            out_path = output_dir / weight_label(weight) / f"{target_id}.yaml"
            write_simple_yaml(out_path, {key: value for key, value in config.items() if key != "loss_types"})
            rows.append(f"{weight_label(weight)},{target_id},{weight:.1f},{config['init_seq']}")

    manifest_path = output_dir / "sim_guided_weight_ablation_manifest.csv"
    manifest_path.write_text(
        "weight_label,target_id,similarity_weight,init_seq\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
