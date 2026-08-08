#!/usr/bin/env python3
"""Validate OGA504x2 parent or NA-MPNN child RNAs with five genuine AF3 models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

NACRAFT_DIR = Path(
    os.environ.get("NACRAFT_DIR", Path(__file__).resolve().parents[3])
)
sys.path[:0] = [str(NACRAFT_DIR), str(NACRAFT_DIR / "utils")]

from af3_utils import (  # noqa: E402
    build_input_json,
    load_presearch,
    parse_outputs,
    run_af3,
)


VALIDATION_PROTOCOL = "five_independent_seeds_5vvo_only_v2"


def load_candidates(args: argparse.Namespace) -> list[dict]:
    with Path(args.parent_manifest).open(newline="") as handle:
        parents = list(csv.DictReader(handle))
    if args.mode == "parent":
        return [{"candidate_id": row["parent_id"], "sequence": row["sequence"]} for row in parents]
    children = []
    for parent in parents:
        path = Path(args.child_root) / parent["parent_id"] / "complete.json"
        if not path.exists():
            raise FileNotFoundError(path)
        children.extend(json.loads(path.read_text())["children"])
    expected = len(parents) * args.children_per_parent
    if len(children) != expected:
        raise RuntimeError(f"Expected {expected} children, got {len(children)}")
    return [{"candidate_id": row["child_id"], "sequence": row["sequence"]} for row in children]


def config() -> dict:
    return {
        "num_samples": 5,
        "run_data_pipeline": False,
        "timeout_sec": 7200,
        "sandbox_dir": os.environ["AF3_SANDBOX_DIR"],
        "af3_code_dir": os.environ["AF3_CODE_DIR"],
        "host_model_dir": os.environ["AF3_MODEL_DIR"],
        "host_presearch_dir": os.environ["AF3_PRESEARCH_DIR"],
        "apptainer_bin": os.environ["APPTAINER_BIN"],
    }


def load_5vvo_templates(path: str, target: dict) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("protocol") != "oga504x2_af3_5vvo_only_v1":
        raise ValueError("Invalid 5VVO template protocol")
    if manifest.get("target_sequence_sha256") != target["sequence_sha256"]:
        raise ValueError("5VVO template target hash mismatch")
    records = manifest.get("protein_templates", {})
    if set(records) != {"A", "B"}:
        raise ValueError("Expected chain A and B 5VVO templates")
    for chain_id, expected_count in (("A", 437), ("B", 429)):
        record = records[chain_id]
        if record.get("entry_id") != "5VVO":
            raise ValueError(f"Template {chain_id} is not 5VVO")
        if record.get("resolved_count") != expected_count:
            raise ValueError(f"Template {chain_id} resolved-count mismatch")
        if len(record["queryIndices"]) != expected_count:
            raise ValueError(f"Template {chain_id} query-index mismatch")
        if record["queryIndices"] != record["templateIndices"]:
            raise ValueError(f"Template {chain_id} index maps differ")
        if not Path(record["mmcifPath"]).exists():
            raise FileNotFoundError(record["mmcifPath"])
    return manifest


def build_task_5vvo(
    *, name: str, sequence: str, protein: str, seed: int,
    presearch_entry: dict, templates: dict,
) -> dict:
    msa_only = {**presearch_entry, "templates": []}
    task = build_input_json(
        name=name,
        seq=sequence,
        ligands=[(protein, "protein"), (protein, "protein")],
        polymer_type="rna",
        seeds=[seed],
        presearch_lookup={protein: msa_only},
    )
    proteins = [item["protein"] for item in task["sequences"] if "protein" in item]
    if len(proteins) != 2 or [item["id"] for item in proteins] != [["B"], ["C"]]:
        raise RuntimeError("Expected AF3 protein chains B and C")
    for protein_input, source_chain in zip(proteins, ("A", "B"), strict=True):
        record = templates[source_chain]
        protein_input["templates"] = [{
            "mmcifPath": record["mmcifPath"],
            "queryIndices": record["queryIndices"],
            "templateIndices": record["templateIndices"],
        }]
    rna = task["sequences"][0]["rna"]
    if "templates" in rna:
        raise RuntimeError("RNA must not receive a template")
    return task


def predict_complex_5vvo(
    *, name: str, sequence: str, protein: str, output_dir: Path,
    seed: int, presearch_entry: dict, templates: dict,
) -> tuple[list, dict]:
    output_dir = Path(output_dir)
    task = build_task_5vvo(
        name=name,
        sequence=sequence,
        protein=protein,
        seed=seed,
        presearch_entry=presearch_entry,
        templates=templates,
    )
    input_json = output_dir / f"{name}_input.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_json.write_text(json.dumps([task], indent=2) + "\n")
    run_af3(
        input_json_path=str(input_json),
        output_dir=str(output_dir),
        cfg={**config(), "num_samples": 1},
        exp_name=name,
        num_samples=1,
    )
    return parse_outputs(str(output_dir), name, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parent", "child"], required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--child-root", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--presearch-json", required=True)
    parser.add_argument("--presearch-outdir", required=True)
    parser.add_argument("--template-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--children-per-parent", type=int, default=10)
    args = parser.parse_args()

    target = json.loads(Path(args.target_manifest).read_text())
    protein = target["sequence"]
    if len(protein) != 504 or target["protein_copy_count"] != 2:
        raise SystemExit("Refusing non-OGA504x2 target")
    presearch = load_presearch(args.presearch_json, out_dir=args.presearch_outdir)
    if protein not in presearch:
        raise SystemExit("Exact OGA504 presearch entry missing")
    template_manifest = load_5vvo_templates(args.template_manifest, target)
    templates = template_manifest["protein_templates"]

    candidates = load_candidates(args)
    for row in candidates[args.worker_id :: args.num_workers]:
        candidate_id = row["candidate_id"]
        out_dir = Path(args.output_root) / candidate_id
        sentinel = out_dir / "complete.json"
        if sentinel.exists():
            previous = json.loads(sentinel.read_text())
            if previous.get("validation_protocol") == VALIDATION_PROTOCOL:
                print(f"skip {candidate_id}", flush=True)
                continue
            legacy = out_dir / "legacy_tiled_complete.json"
            if not legacy.exists():
                sentinel.replace(legacy)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            replicate_records = []
            combined_metrics: dict[str, list[float | None]] = {}
            for replicate_index, seed in enumerate((101, 211, 307, 401, 503), 1):
                replicate_name = f"{candidate_id}_rep{replicate_index}_seed{seed}"
                replicate_dir = out_dir / f"replicate_{replicate_index}"
                structs, metrics = predict_complex_5vvo(
                    name=replicate_name,
                    sequence=row["sequence"],
                    protein=protein,
                    output_dir=str(replicate_dir),
                    seed=seed,
                    presearch_entry=presearch[protein],
                    templates=templates,
                )
                if len(structs) != 1:
                    raise RuntimeError(
                        f"Replicate {replicate_index} returned {len(structs)} structures"
                    )
                chain_lengths = [len(chain) for chain in structs[0][0]]
                if chain_lengths != [len(row["sequence"]), 504, 504]:
                    raise RuntimeError(
                        f"Invalid chain lengths in replicate {replicate_index}: "
                        f"{chain_lengths}"
                    )
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
                        "mode": args.mode,
                        "candidate_id": candidate_id,
                        "sequence": row["sequence"],
                        "target_system": "OGA504x2+RNA",
                        "validation_protocol": VALIDATION_PROTOCOL,
                        "template_protocol": template_manifest["protocol"],
                        "template_source_cif_sha256": template_manifest[
                            "source_cif_sha256"
                        ],
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
            print(f"FAILED {candidate_id}: {error}", flush=True)


if __name__ == "__main__":
    main()
