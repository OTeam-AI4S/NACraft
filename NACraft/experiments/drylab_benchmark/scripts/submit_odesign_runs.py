#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

try:
    from drylab_common import read_csv
except ModuleNotFoundError:
    from .drylab_common import read_csv


def build_odesign_command(
    target: dict[str, str],
    input_json_path: str,
    odesign_root: str,
    output_root: str,
    samples: int,
    python_env_bin: str,
    seeds: str,
) -> str:
    polymer_type = target["polymer_type"].lower()
    out_dir = f"{output_root}/{target['target_id']}"
    parts = [
        "bash",
        "scripts/run_odesign.sh",
        "--infer_model_name",
        "odesign_base_na_rigid",
        "--design_modality",
        polymer_type,
        "--data_root_dir",
        "./ODesign/data",
        "--ckpt_root_dir",
        "./ODesign/ckpt",
        "--input_json_path",
        input_json_path,
        "--exp_name",
        f"{target['target_id']}_{polymer_type}_odesign",
        "--seeds",
        seeds,
        "--N_sample",
        str(samples),
        "--num_workers",
        "4",
        "--output_dir",
        out_dir,
    ]
    command = " ".join(shlex.quote(part) for part in parts)
    return f"cd {shlex.quote(odesign_root)} && PATH={shlex.quote(python_env_bin)}:$PATH {command}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a shell command list for local ODesign NA runs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--odesign-root", default=os.environ.get("ODESIGN_ROOT", ""))
    parser.add_argument("--input-dir", default="outputs/drylab_benchmark/configs/odesign/inputs")
    parser.add_argument("--out-root", default="outputs/drylab_benchmark/candidates/odesign")
    parser.add_argument("--python-env-bin", default=str(Path(sys.executable).parent))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seeds", default="[42]")
    args = parser.parse_args()
    if not args.odesign_root:
        parser.error("set --odesign-root or ODESIGN_ROOT")

    lines = []
    for target in read_csv(args.manifest):
        lines.append(
            build_odesign_command(
                target=target,
                input_json_path=str(Path(args.input_dir) / f"{target['target_id']}.json"),
                odesign_root=args.odesign_root,
                output_root=args.out_root,
                samples=args.samples,
                python_env_bin=args.python_env_bin,
                seeds=args.seeds,
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + ("\n" if lines else ""))


if __name__ == "__main__":
    main()
