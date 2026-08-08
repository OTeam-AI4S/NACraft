#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from analyze_nacraft_af3_results import (
    canonical_sequence,
    candidate_artifact_paths,
    design_artifact_paths,
    fmt,
    gc_fraction,
    invalid_bases,
    mean,
    offdiag,
    plddt_from_confidence,
    sequence_identity,
    summarize_group,
    to_float,
)
from drylab_common import read_csv, write_csv
from utils.target_rmsd import target_rmsd_fields


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def chain_pair_value(matrix: object, i: int, j: int) -> float | None:
    if not isinstance(matrix, list):
        return None
    try:
        return to_float(matrix[i][j])
    except (IndexError, TypeError):
        return None


def exact_paths(af3_root: Path, candidate: dict[str, str]) -> tuple[Path, Path]:
    target_id = candidate["target_id"]
    method = candidate["method"]
    candidate_id = candidate["candidate_id"]
    out_dir = af3_root / method / target_id / candidate_id
    run_dir = out_dir / f"{target_id}_{method}_{candidate_id}"
    summary = out_dir / f"{target_id}__{method}__{candidate_id}_summary.json"
    confidence = run_dir / f"{target_id}_{method}_{candidate_id}_confidences.json"
    return summary, confidence


def collect_rows(
    candidates_path: Path,
    af3_root: Path,
    manifest_path: Path,
    experiment: str,
) -> list[dict[str, object]]:
    manifest = {row["target_id"]: row for row in read_csv(manifest_path)}
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for candidate in read_csv(candidates_path):
        summary, confidence = exact_paths(af3_root, candidate)
        if not summary.exists():
            missing.append(str(summary))
            continue
        target_id = candidate["target_id"]
        method = candidate["method"]
        candidate_id = candidate["candidate_id"]
        target_meta = manifest.get(target_id, {})
        data = read_json(summary)
        pair_iptm = offdiag(data.get("chain_pair_iptm"))
        pair_pae = offdiag(data.get("chain_pair_pae_min") or data.get("chain_pair_pae"))
        sequence = candidate.get("sequence", "")
        polymer_type = str(target_meta.get("polymer_type", candidate.get("polymer_type", ""))).lower()
        bad_bases = invalid_bases(sequence, polymer_type)
        native_seq = target_meta.get("native_na_sequence", "")
        row: dict[str, object] = {
            **candidate,
            "experiment": experiment,
            "polymer_type": polymer_type,
            "na_length": target_meta.get("na_length", len(sequence)),
            "native_structure_path": target_meta.get("structure_path", ""),
            "canonical_sequence": canonical_sequence(sequence, polymer_type),
            "canonical_valid": int(not bad_bases),
            "invalid_bases": bad_bases,
            "native_sequence_identity": sequence_identity(sequence, native_seq),
            "gc_fraction": gc_fraction(sequence),
            "iptm": to_float(data.get("iptm")),
            "ptm": to_float(data.get("ptm")),
            "ranking_score": to_float(data.get("ranking_score")),
            "has_clash": to_float(data.get("has_clash")),
            "fraction_disordered": to_float(data.get("fraction_disordered")),
            "pair_iptm_mean": mean(pair_iptm),
            "pair_iptm_min": min(pair_iptm) if pair_iptm else None,
            "protein_to_aptamer_iptm": chain_pair_value(data.get("chain_pair_iptm"), 0, -1),
            "aptamer_to_protein_iptm": chain_pair_value(data.get("chain_pair_iptm"), -1, 0),
            "ipae_mean": mean(pair_pae),
            "ipae_min": min(pair_pae) if pair_pae else None,
            "protein_to_aptamer_ipae": chain_pair_value(data.get("chain_pair_pae_min"), 0, -1),
            "aptamer_to_protein_ipae": chain_pair_value(data.get("chain_pair_pae_min"), -1, 0),
            "summary_path": str(summary),
        }
        row.update(candidate_artifact_paths(summary.parent, target_id, method, candidate_id))
        row.update(design_artifact_paths(candidate.get("source_fasta", ""), candidate.get("variant", ""), candidate.get("variant_index", "")))
        row.update(plddt_from_confidence(confidence))
        row.update(target_rmsd_fields(row.get("af3_model_cif_path", ""), target_meta))
        row["pass_iptm_060"] = int((row.get("iptm") or 0) >= 0.60)
        row["pass_iptm_070"] = int((row.get("iptm") or 0) >= 0.70)
        row["pass_iptm_080"] = int((row.get("iptm") or 0) >= 0.80)
        row["pass_plddt_70"] = int((row.get("plddt_aptamer") or 0) >= 70.0)
        rows.append(row)
    if missing:
        print(f"[warn] missing summaries: {len(missing)}")
        print("\n".join(missing[:20]))
    return rows


def add_weight(rows: list[dict[str, object]]) -> None:
    for row in rows:
        method = str(row.get("method", ""))
        suffix = method.rsplit("w", 1)[-1] if "w" in method else ""
        try:
            row["similarity_weight"] = int(suffix) / 100
        except ValueError:
            row["similarity_weight"] = ""


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "sim_guided_weight_ablation_metrics.csv", [{k: fmt(v) for k, v in row.items()} for row in rows])
    for keys, name in [
        (["similarity_weight"], "summary_by_weight.csv"),
        (["similarity_weight", "target_id"], "summary_by_weight_target.csv"),
        (["similarity_weight", "variant"], "summary_by_weight_variant.csv"),
    ]:
        write_csv(output_dir / name, [{k: fmt(v) for k, v in row.items()} for row in summarize_group(rows, keys)])

    counts: dict[tuple[object, object], int] = defaultdict(int)
    for row in rows:
        counts[(row.get("similarity_weight", ""), row.get("target_id", ""))] += 1
    expected = 50
    missing = [
        {"similarity_weight": weight, "target_id": target_id, "n": n, "expected": expected, "missing": expected - n}
        for (weight, target_id), n in sorted(counts.items())
        if n != expected
    ]
    write_csv(output_dir / "qc_missing_by_weight_target.csv", missing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AF3 metrics by exact candidate CSV paths.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--af3-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", default="sim_guided_weight_ablation")
    args = parser.parse_args()

    rows = collect_rows(Path(args.candidates), Path(args.af3_root), Path(args.manifest), args.experiment)
    add_weight(rows)
    write_outputs(rows, Path(args.output_dir))
    print(f"rows {len(rows)}")
    print(f"output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
