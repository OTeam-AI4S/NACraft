#!/usr/bin/env python3
"""AF3 validation for the 16 corrected OGA multilength pilot candidates."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from run_af3_validate import (
    VALIDATION_PROTOCOL,
    load_5vvo_templates,
    load_presearch,
    predict_complex_5vvo,
)


def load_candidates(root: Path) -> list[dict]:
    candidates = []
    for group in ("denovo_20", "denovo_40", "a3_guided_41", "denovo_60"):
        parent_id = f"{group}_design0"
        record = json.loads(
            (root / "nampnn_children" / parent_id / "complete.json").read_text()
        )
        children = record["children"]
        if len(children) != 3:
            raise RuntimeError(f"Expected three children for {parent_id}")
        parent_sequence = children[0]["parent_sequence"]
        candidates.append(
            {"candidate_id": parent_id, "sequence": parent_sequence, "group": group}
        )
        for child in children:
            if child["parent_id"] != parent_id or child["sequence"] == parent_sequence:
                raise RuntimeError(f"Invalid child record for {parent_id}")
            candidates.append(
                {
                    "candidate_id": child["child_id"],
                    "sequence": child["sequence"],
                    "group": group,
                }
            )
    if len(candidates) != 16 or len({row["candidate_id"] for row in candidates}) != 16:
        raise RuntimeError("Expected 16 unique pilot candidates")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--presearch-json", required=True)
    parser.add_argument("--presearch-outdir", required=True)
    parser.add_argument("--template-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--worker-id", required=True, type=int)
    args = parser.parse_args()

    candidates = load_candidates(Path(args.pilot_root))
    row = candidates[args.worker_id]
    target = json.loads(Path(args.target_manifest).read_text())
    protein = target["sequence"]
    if len(protein) != 504 or target["protein_copy_count"] != 2:
        raise SystemExit("Refusing non-OGA504x2 target")
    presearch = load_presearch(args.presearch_json, out_dir=args.presearch_outdir)
    if protein not in presearch:
        raise SystemExit("Exact OGA504 presearch entry missing")
    template_manifest = load_5vvo_templates(args.template_manifest, target)
    templates = template_manifest["protein_templates"]

    candidate_id = row["candidate_id"]
    out_dir = Path(args.output_root) / candidate_id
    sentinel = out_dir / "complete.json"
    if sentinel.exists():
        previous = json.loads(sentinel.read_text())
        if previous.get("validation_protocol") == VALIDATION_PROTOCOL:
            print(f"skip {candidate_id}", flush=True)
            return
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        replicate_records = []
        combined_metrics: dict[str, list[float | None]] = {}
        for replicate_index, seed in enumerate((101, 211, 307, 401, 503), 1):
            replicate_name = f"{candidate_id}_rep{replicate_index}_seed{seed}"
            structs, metrics = predict_complex_5vvo(
                name=replicate_name,
                sequence=row["sequence"],
                protein=protein,
                output_dir=out_dir / f"replicate_{replicate_index}",
                seed=seed,
                presearch_entry=presearch[protein],
                templates=templates,
            )
            if len(structs) != 1:
                raise RuntimeError(f"Replicate {replicate_index} returned {len(structs)} structures")
            chain_lengths = [len(chain) for chain in structs[0][0]]
            if chain_lengths != [len(row["sequence"]), 504, 504]:
                raise RuntimeError(f"Invalid chain lengths: {chain_lengths}")
            replicate_metrics = {}
            for key, value in metrics.items():
                array = np.asarray(value).reshape(-1)
                scalar = float(array[0]) if len(array) else float("nan")
                clean = None if np.isnan(scalar) else scalar
                replicate_metrics[key] = clean
                combined_metrics.setdefault(key, []).append(clean)
            replicate_records.append(
                {
                    "replicate_index": replicate_index,
                    "seed": seed,
                    "chain_lengths": chain_lengths,
                    "metrics": replicate_metrics,
                }
            )
        sentinel.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mode": "pilot",
                    "candidate_id": candidate_id,
                    "group": row["group"],
                    "sequence": row["sequence"],
                    "target_system": "OGA504x2+RNA",
                    "validation_protocol": VALIDATION_PROTOCOL,
                    "template_protocol": template_manifest["protocol"],
                    "template_source_cif_sha256": template_manifest["source_cif_sha256"],
                    "protein_template_resolved_counts": {"B": 437, "C": 429},
                    "num_models": 5,
                    "seeds": [101, 211, 307, 401, 503],
                    "replicates": replicate_records,
                    "metrics": combined_metrics,
                },
                indent=2,
            )
            + "\n"
        )
        failed_path = out_dir / "failed.json"
        if failed_path.exists():
            failed_path.unlink()
        print(f"complete {candidate_id}: 5 models", flush=True)
    except Exception as error:
        (out_dir / "failed.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "candidate_id": candidate_id,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
