#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from drylab_common import read_csv
except ModuleNotFoundError:
    from .drylab_common import read_csv


def load_target(manifest: str | Path, target_id: str) -> dict[str, str]:
    target_id = target_id.upper()
    for row in read_csv(manifest):
        if row["target_id"].upper() == target_id:
            return row
    raise SystemExit(f"Target {target_id} not found in {manifest}")


def load_af3_utils(nacraft_dir: str | None) -> tuple[object, object]:
    candidates = []
    if nacraft_dir:
        candidates.append(Path(nacraft_dir))
    candidates.extend(
        [
            Path(__file__).resolve().parents[3],
        ]
    )
    for root in candidates:
        if (root / "utils" / "af3_utils.py").exists():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root / "utils"))
            from af3_utils import find_run_dir, predict_complex  # type: ignore

            return find_run_dir, predict_complex
    raise SystemExit("NACraft/utils/af3_utils.py not found; pass --nacraft-dir")


def load_presearch_lookup(presearch_json: str, presearch_outdir: str | None) -> dict | None:
    if not presearch_json:
        return None
    from af3_utils import load_presearch  # type: ignore

    return load_presearch(presearch_json, out_dir=presearch_outdir)


def parse_summary_confidences(run_dir: Path) -> dict:
    for path in sorted(run_dir.glob("*summary_confidence*.json")):
        return json.loads(path.read_text())
    raise FileNotFoundError(f"No AF3 summary confidence JSON found under {run_dir}")


def af3_cfg(args: argparse.Namespace) -> dict[str, object]:
    cfg = {
        "num_samples": args.num_samples,
        "run_data_pipeline": False,
        "timeout_sec": args.timeout_sec,
        "sandbox_dir": os.environ.get("AF3_SANDBOX_DIR"),
        "af3_code_dir": os.environ.get("AF3_CODE_DIR"),
        "host_model_dir": os.environ.get("AF3_MODEL_DIR"),
        "host_db_dir": os.environ.get("AF3_DB_DIR"),
        "host_presearch_dir": args.presearch_outdir,
        "apptainer_bin": os.environ.get("APPTAINER_BIN"),
    }
    return {key: value for key, value in cfg.items() if value not in (None, "")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AF3 predict-only validation for a drylab candidate.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--polymer-type", choices=["rna", "dna"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nacraft-dir", default=os.environ.get("NACRAFT_DIR", ""))
    parser.add_argument("--presearch-json", default="")
    parser.add_argument("--presearch-outdir", default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / f"{args.target_id}__{args.method}__{args.candidate_id}_summary.json"
    if args.skip_existing and sentinel.exists():
        return

    target = load_target(args.manifest, args.target_id)
    find_run_dir, predict_complex = load_af3_utils(args.nacraft_dir)
    presearch_lookup = load_presearch_lookup(args.presearch_json, args.presearch_outdir)
    name = f"{args.target_id}_{args.method}_{args.candidate_id}".replace("/", "_")
    predict_complex(
        name=name,
        seq=args.sequence.upper(),
        ligands=[(target["protein_sequence"], "protein")],
        polymer_type=args.polymer_type,
        output_dir=str(out_dir),
        cfg=af3_cfg(args),
        exp_name=name,
        num_samples=args.num_samples,
        presearch_lookup=presearch_lookup,
    )
    run_dir = find_run_dir(str(out_dir), name)
    if run_dir is None:
        raise SystemExit(f"AF3 run directory not found under {out_dir} for {name}")
    summary = parse_summary_confidences(Path(run_dir))
    summary.update(
        {
            "target_id": args.target_id,
            "method": args.method,
            "candidate_id": args.candidate_id,
            "sequence": args.sequence.upper(),
            "polymer_type": args.polymer_type,
        }
    )
    sentinel.write_text(json.dumps(summary, indent=2) + "\n")

    raw_summary = Path(run_dir) / sentinel.name
    if not raw_summary.exists():
        shutil.copy2(sentinel, raw_summary)


if __name__ == "__main__":
    main()
