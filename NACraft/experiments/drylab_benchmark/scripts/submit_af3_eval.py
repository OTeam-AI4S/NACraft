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


AF3_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "NACRAFT_DIR": str(Path(__file__).resolve().parents[3]),
    "AF3_SANDBOX_DIR": os.environ.get("AF3_SANDBOX_DIR", ""),
    "AF3_SIF_PATH": os.environ.get("AF3_SIF_PATH", ""),
    "AF3_CODE_DIR": os.environ.get("AF3_CODE_DIR", ""),
    "AF3_MODEL_DIR": os.environ.get("AF3_MODEL_DIR", ""),
    "AF3_DB_DIR": os.environ.get("AF3_DB_DIR", ""),
    "APPTAINER_BIN": os.environ.get("APPTAINER_BIN", "apptainer"),
}


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_af3_command(
    candidate: dict[str, str],
    target: dict[str, str],
    manifest: str,
    af3_entry: str,
    python_executable: str,
    out_root: str,
    num_samples: int,
    presearch_json: str = "",
    presearch_outdir: str = "",
) -> str:
    out_dir = f"{out_root}/{candidate['method']}/{candidate['target_id']}/{candidate['candidate_id']}"
    parts = [
        python_executable,
        af3_entry,
        "--manifest",
        manifest,
        "--target-id",
        candidate["target_id"],
        "--method",
        candidate["method"],
        "--candidate-id",
        candidate["candidate_id"],
        "--sequence",
        candidate["sequence"],
        "--polymer-type",
        target["polymer_type"],
        "--output-dir",
        out_dir,
        "--num-samples",
        str(num_samples),
        "--skip-existing",
    ]
    if presearch_json:
        parts.extend(["--presearch-json", presearch_json])
    if presearch_outdir:
        parts.extend(["--presearch-outdir", presearch_outdir])
    env = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in AF3_ENV.items()
        if value
    )
    return f"{env} {shell_join(parts)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AF3 validation command list from candidate CSV.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--af3-entry", default="NACraft/experiments/drylab_benchmark/scripts/run_af3_predict_only.py")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--out-root", default="outputs/drylab_benchmark/af3_eval")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--presearch-json", default="")
    parser.add_argument("--presearch-outdir", default="")
    parser.add_argument("--limit-candidates", type=int, default=0)
    args = parser.parse_args()

    targets = {row["target_id"].upper(): row for row in read_csv(args.manifest)}
    lines = []
    candidates = read_csv(args.candidates)
    if args.limit_candidates:
        candidates = candidates[: args.limit_candidates]
    for candidate in candidates:
        target = targets.get(candidate["target_id"].upper())
        if not target:
            continue
        lines.append(
            build_af3_command(
                candidate=candidate,
                target=target,
                manifest=args.manifest,
                af3_entry=args.af3_entry,
                python_executable=args.python_executable,
                out_root=args.out_root,
                num_samples=args.num_samples,
                presearch_json=args.presearch_json,
                presearch_outdir=args.presearch_outdir,
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + ("\n" if lines else ""))


if __name__ == "__main__":
    main()
