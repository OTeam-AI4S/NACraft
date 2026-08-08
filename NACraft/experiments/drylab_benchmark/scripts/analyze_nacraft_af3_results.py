#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

NACRAFT_ROOT = Path(__file__).resolve().parents[3]
if str(NACRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(NACRAFT_ROOT))

from drylab_common import read_csv, write_csv
from utils.target_rmsd import target_rmsd_fields


def to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str) and value.lower() == "nan":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def median(values: Iterable[float | None]) -> float | None:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    return statistics.median(clean) if clean else None


def quantile(values: Iterable[float | None], q: float) -> float | None:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def fmt(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return value


def read_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def summary_path(root: Path, row: Mapping[str, str]) -> Path:
    target_id = row["target_id"]
    method = row["method"]
    candidate_id = row["candidate_id"]
    return root / method / target_id / candidate_id / f"{target_id}__{method}__{candidate_id}_summary.json"


def confidence_path(candidate_dir: Path, target_id: str, method: str, candidate_id: str) -> Path | None:
    nested = candidate_dir / f"{target_id}_{method}_{candidate_id}"
    direct = nested / f"{target_id}_{method}_{candidate_id}_confidences.json"
    if direct.exists():
        return direct
    matches = sorted(candidate_dir.glob("*/*_confidences.json"))
    return matches[0] if matches else None


def candidate_artifact_paths(candidate_dir: Path, target_id: str, method: str, candidate_id: str) -> dict[str, str]:
    nested = candidate_dir / f"{target_id}_{method}_{candidate_id}"
    model = nested / f"{target_id}_{method}_{candidate_id}_model.cif"
    summary_conf = nested / f"{target_id}_{method}_{candidate_id}_summary_confidences.json"
    confidence = nested / f"{target_id}_{method}_{candidate_id}_confidences.json"
    input_json = candidate_dir / f"{target_id}_{method}_{candidate_id}_input.json"
    return {
        "af3_candidate_dir": str(candidate_dir),
        "af3_run_dir": str(nested) if nested.exists() else "",
        "af3_model_cif_path": str(model) if model.exists() else "",
        "af3_input_json_path": str(input_json) if input_json.exists() else "",
        "af3_confidence_json_path": str(confidence) if confidence.exists() else "",
        "af3_summary_confidences_json_path": str(summary_conf) if summary_conf.exists() else "",
    }


DESIGN_DIR_CACHE: dict[Path, dict[str, str]] = {}
REDESIGN_CACHE: dict[tuple[Path, object], dict[str, str]] = {}


def semicolon_paths(paths: Iterable[Path]) -> str:
    return ";".join(str(path) for path in sorted(paths))


def design_artifact_paths(source_fasta: str, variant: str, variant_index: object) -> dict[str, str]:
    if not source_fasta:
        return {}
    fasta = Path(source_fasta)
    try:
        design_dir = fasta.parents[2]
    except IndexError:
        return {"source_fasta": source_fasta}
    nampnn_dir = design_dir / "nampnn"
    regen_dir = nampnn_dir / "boltz_regen"
    if design_dir not in DESIGN_DIR_CACHE:
        composite = nampnn_dir / "composite.pdb"
        trace = design_dir / "optimization_trace.tsv"
        DESIGN_DIR_CACHE[design_dir] = {
            "source_design_dir": str(design_dir),
            "source_composite_pdb_path": str(composite) if composite.exists() else "",
            "source_optimized_cif_paths": semicolon_paths(design_dir.glob("state*_sample*.cif")),
            "source_optimized_pdb_paths": semicolon_paths(design_dir.glob("state*_sample*.pdb")),
            "source_optimization_trace_path": str(trace) if trace.exists() else "",
        }
    out = dict(DESIGN_DIR_CACHE[design_dir])
    redesign_key = (design_dir, variant_index)
    if redesign_key not in REDESIGN_CACHE:
        prefix = f"nampnn_seq{variant_index}_state*"
        REDESIGN_CACHE[redesign_key] = {
            "source_redesign_cif_paths": semicolon_paths(regen_dir.glob(f"{prefix}_sample*.cif")),
            "source_redesign_pdb_paths": semicolon_paths(regen_dir.glob(f"{prefix}_sample*.pdb")),
        }
    out.update(REDESIGN_CACHE[redesign_key] if str(variant) == "redesign" else {"source_redesign_cif_paths": "", "source_redesign_pdb_paths": ""})
    return out


def offdiag(matrix: object) -> list[float]:
    if not isinstance(matrix, list):
        return []
    values: list[float] = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            continue
        for j, item in enumerate(row):
            if i == j:
                continue
            value = to_float(item)
            if value is not None:
                values.append(value)
    return values


def chain_pair_value(matrix: object, i: int, j: int) -> float | None:
    if not isinstance(matrix, list):
        return None
    try:
        return to_float(matrix[i][j])
    except (IndexError, TypeError):
        return None


def plddt_from_confidence(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    data = read_json(path)
    plddts = [to_float(v) for v in data.get("atom_plddts", [])] if isinstance(data.get("atom_plddts"), list) else []
    chain_ids = data.get("atom_chain_ids", [])
    if not plddts or not isinstance(chain_ids, list) or len(chain_ids) != len(plddts):
        return {"plddt": mean(plddts), "confidence_path": str(path)}

    ordered_chains: list[str] = []
    for chain_id in chain_ids:
        chain = str(chain_id)
        if chain not in ordered_chains:
            ordered_chains.append(chain)
    by_chain: dict[str, list[float]] = defaultdict(list)
    for chain_id, value in zip(chain_ids, plddts):
        if value is not None:
            by_chain[str(chain_id)].append(value)
    aptamer_chain = ordered_chains[-1] if ordered_chains else ""
    protein_chains = ordered_chains[:-1]
    return {
        "plddt": mean(plddts),
        "plddt_aptamer": mean(by_chain.get(aptamer_chain, [])),
        "plddt_protein": mean(v for chain in protein_chains for v in by_chain.get(chain, [])),
        "aptamer_chain": aptamer_chain,
        "protein_chain_count": len(protein_chains),
        "confidence_path": str(path),
    }


def sequence_identity(seq: str, ref: str) -> float | None:
    seq = seq.upper()
    ref = ref.upper()
    if not seq or not ref or len(seq) != len(ref):
        return None
    return sum(a == b for a, b in zip(seq, ref)) / len(seq)


def gc_fraction(seq: str) -> float | None:
    seq = seq.upper()
    if not seq:
        return None
    return sum(base in {"G", "C"} for base in seq) / len(seq)


def canonical_sequence(seq: str, polymer_type: str) -> str:
    seq = seq.upper()
    if polymer_type == "rna":
        return seq.replace("T", "U")
    return seq


def invalid_bases(seq: str, polymer_type: str) -> str:
    allowed = {"dna": set("ACGT"), "rna": set("ACGU")}.get(polymer_type, set())
    return "".join(sorted(set(canonical_sequence(seq, polymer_type)) - allowed))


def merge_metrics(
    candidates_path: Path,
    af3_root: Path,
    manifest_path: Path | None,
    experiment: str,
    with_confidence: bool = False,
) -> list[dict[str, object]]:
    manifest = {row["target_id"]: row for row in read_csv(manifest_path)} if manifest_path else {}
    rows: list[dict[str, object]] = []
    missing = 0
    for candidate in read_csv(candidates_path):
        path = summary_path(af3_root, candidate)
        if not path.exists():
            missing += 1
            continue
        artifact_paths = candidate_artifact_paths(path.parent, candidate["target_id"], candidate["method"], candidate["candidate_id"])
        design_paths = design_artifact_paths(
            candidate.get("source_fasta", ""),
            candidate.get("variant", ""),
            candidate.get("variant_index", ""),
        )
        data = read_json(path)
        target_id = candidate["target_id"]
        target_meta = manifest.get(target_id) or manifest.get(candidate.get("source_target_id", ""))
        pair_iptm = offdiag(data.get("chain_pair_iptm"))
        pair_pae = offdiag(data.get("chain_pair_pae_min") or data.get("chain_pair_pae"))
        conf = (
            plddt_from_confidence(confidence_path(path.parent, target_id, candidate["method"], candidate["candidate_id"]))
            if with_confidence
            else {}
        )
        sequence = candidate.get("sequence", "")
        polymer_type = (target_meta or {}).get("polymer_type", candidate.get("polymer_type", ""))
        bad_bases = invalid_bases(sequence, str(polymer_type).lower())
        native_seq = (target_meta or {}).get("native_na_sequence", "")
        row: dict[str, object] = {
            **candidate,
            "experiment": experiment,
            "polymer_type": polymer_type,
            "na_length": (target_meta or {}).get("na_length", len(sequence)),
            "native_structure_path": (target_meta or {}).get("structure_path", ""),
            "state_target_id": (target_meta or {}).get("state_target_id", ""),
            "benchmark_id": (target_meta or {}).get("benchmark_id", ""),
            "specificity_class": (target_meta or {}).get("specificity_class", ""),
            "experiment_tier": (target_meta or {}).get("experiment_tier", ""),
            "canonical_sequence": canonical_sequence(sequence, str(polymer_type).lower()),
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
            "summary_path": str(path),
        }
        row.update(artifact_paths)
        row.update(design_paths)
        row.update(conf)
        row.update(target_rmsd_fields(artifact_paths.get("af3_model_cif_path", ""), target_meta or {}))
        row["pass_iptm_060"] = int((row.get("iptm") or 0) >= 0.60)
        row["pass_iptm_070"] = int((row.get("iptm") or 0) >= 0.70)
        row["pass_iptm_080"] = int((row.get("iptm") or 0) >= 0.80)
        row["pass_plddt_70"] = int((row.get("plddt_aptamer") or 0) >= 70.0)
        rows.append(row)
    if missing:
        print(f"[warn] Missing AF3 summaries for {missing} candidates from {candidates_path}")
    return rows


def summarize_group(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for group_key, group in sorted(grouped.items(), key=lambda item: item[0]):
        iptm = [to_float(row.get("iptm")) for row in group]
        plddt = [to_float(row.get("plddt_aptamer")) for row in group]
        ipae = [to_float(row.get("ipae_mean")) for row in group]
        target_rmsd = [to_float(row.get("target_RMSD")) for row in group]
        record = {key: value for key, value in zip(keys, group_key)}
        record.update(
            {
                "n": len(group),
                "iptm_mean": mean(iptm),
                "iptm_median": median(iptm),
                "iptm_p75": quantile(iptm, 0.75),
                "iptm_p90": quantile(iptm, 0.90),
                "iptm_max": max(v for v in iptm if v is not None) if any(v is not None for v in iptm) else None,
                "plddt_aptamer_mean": mean(plddt),
                "ipae_mean": mean(ipae),
                "target_RMSD_mean": mean(target_rmsd),
                "target_RMSD_median": median(target_rmsd),
                "target_RMSD_max": max(v for v in target_rmsd if v is not None) if any(v is not None for v in target_rmsd) else None,
                "success_iptm060": mean([to_float(row.get("pass_iptm_060")) for row in group]),
                "success_iptm070": mean([to_float(row.get("pass_iptm_070")) for row in group]),
                "success_iptm080": mean([to_float(row.get("pass_iptm_080")) for row in group]),
            }
        )
        out.append(record)
    return out


def top_candidates(rows: Sequence[Mapping[str, object]], top_n: int) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("target_id", ""), row.get("method", ""))].append(row)
    out = []
    for (target_id, method), group in sorted(grouped.items()):
        ranked = sorted(group, key=lambda row: (to_float(row.get("iptm")) or -1.0), reverse=True)
        for rank, row in enumerate(ranked[:top_n], start=1):
            out.append(
                {
                    "target_id": target_id,
                    "method": method,
                    "rank": rank,
                    "candidate_id": row.get("candidate_id", ""),
                    "variant": row.get("variant", ""),
                    "sequence": row.get("sequence", ""),
                    "iptm": row.get("iptm", ""),
                    "plddt_aptamer": row.get("plddt_aptamer", ""),
                    "ipae_mean": row.get("ipae_mean", ""),
                    "target_RMSD": row.get("target_RMSD", ""),
                    "ranking_score": row.get("ranking_score", ""),
                }
            )
    return out


def confsel_delta(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        source_target_id = row.get("source_target_id") or str(row.get("target_id", "")).rsplit("__", 1)[0]
        base_candidate = str(row.get("candidate_id", "")).rsplit("__", 1)[0]
        state = row.get("validation_state") or str(row.get("target_id", "")).rsplit("__", 1)[-1]
        grouped[(source_target_id, base_candidate)][str(state)] = row
    out = []
    for (source_target_id, base_candidate), pair in sorted(grouped.items()):
        pos = pair.get("positive")
        neg = pair.get("negative")
        if not pos or not neg:
            continue
        pos_iptm = to_float(pos.get("iptm"))
        neg_iptm = to_float(neg.get("iptm"))
        pos_ipae = to_float(pos.get("ipae_mean"))
        neg_ipae = to_float(neg.get("ipae_mean"))
        pos_target_rmsd = to_float(pos.get("target_RMSD"))
        neg_target_rmsd = to_float(neg.get("target_RMSD"))
        out.append(
            {
                "source_target_id": source_target_id,
                "base_candidate_id": base_candidate,
                "variant": pos.get("variant", ""),
                "sequence": pos.get("sequence", ""),
                "positive_iptm": pos_iptm,
                "negative_iptm": neg_iptm,
                "delta_iptm_pos_minus_neg": (pos_iptm - neg_iptm) if pos_iptm is not None and neg_iptm is not None else None,
                "positive_ipae": pos_ipae,
                "negative_ipae": neg_ipae,
                "delta_ipae_neg_minus_pos": (neg_ipae - pos_ipae) if pos_ipae is not None and neg_ipae is not None else None,
                "positive_target_RMSD": pos_target_rmsd,
                "negative_target_RMSD": neg_target_rmsd,
                "max_target_RMSD": max(pos_target_rmsd, neg_target_rmsd) if pos_target_rmsd is not None and neg_target_rmsd is not None else None,
                "positive_plddt_aptamer": pos.get("plddt_aptamer", ""),
                "negative_plddt_aptamer": neg.get("plddt_aptamer", ""),
                "selective_by_iptm": int(pos_iptm is not None and neg_iptm is not None and pos_iptm > neg_iptm),
                "selective_by_ipae": int(pos_ipae is not None and neg_ipae is not None and pos_ipae < neg_ipae),
            }
        )
    return out


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]], title: str, limit: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n")
        return
    keys = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(fmt(row.get(key, ""))) for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed NACraft AF3 dry-lab outputs.")
    parser.add_argument("--data-root", default="data/drylab_benchmark")
    parser.add_argument("--output-dir", default="data/drylab_benchmark/results/nacraft_af3")
    parser.add_argument(
        "--with-confidence",
        action="store_true",
        help="Also read large AF3 confidence JSON files to compute pLDDT. Slow on network filesystems.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    na20 = merge_metrics(
        data_root / "processed/nacraft_na20_af3_candidates.csv",
        data_root / "af3_eval/nacraft_na20",
        data_root / "processed/target_manifest_primary_na20_hotspots.csv",
        "na20",
        with_confidence=args.with_confidence,
    )
    confsel = merge_metrics(
        data_root / "conf_selective/processed/nacraft_confsel_af3_candidates.csv",
        data_root / "conf_selective/af3_eval/nacraft_confsel",
        data_root / "conf_selective/processed/conf_selective_af3_manifest.csv",
        "conf_selective",
        with_confidence=args.with_confidence,
    )
    all_rows = na20 + confsel

    fieldnames = []
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(output_dir / "nacraft_af3_merged.csv", [{k: fmt(v) for k, v in row.items()} for row in all_rows], fieldnames)
    write_csv(
        output_dir / "nacraft_all_candidates_with_paths.csv",
        [{k: fmt(v) for k, v in row.items()} for row in all_rows],
        fieldnames,
    )

    summaries = {
        "na20_by_method": summarize_group(na20, ["method"]),
        "na20_by_method_polymer": summarize_group(na20, ["method", "polymer_type"]),
        "na20_by_target_method": summarize_group(na20, ["target_id", "method"]),
        "na20_by_variant": summarize_group(na20, ["method", "variant"]),
        "na20_by_method_canonical_valid": summarize_group(na20, ["method", "canonical_valid"]),
        "na20_by_variant_canonical_valid": summarize_group(na20, ["method", "variant", "polymer_type", "canonical_valid"]),
        "confsel_by_target_state": summarize_group(confsel, ["source_target_id", "validation_state"]),
        "confsel_by_variant_state": summarize_group(confsel, ["variant", "validation_state"]),
    }
    for name, rows in summaries.items():
        write_csv(output_dir / f"{name}.csv", [{k: fmt(v) for k, v in row.items()} for row in rows])
        write_markdown(output_dir / f"{name}.md", rows, name)

    top = top_candidates(na20, top_n=10)
    write_csv(output_dir / "na20_top10_per_target_method.csv", [{k: fmt(v) for k, v in row.items()} for row in top])
    write_markdown(output_dir / "na20_top10_per_target_method.md", top, "NA-20 top candidates", limit=80)

    na20_valid = [row for row in na20 if row.get("canonical_valid") == 1]
    valid_summaries = {
        "na20_canonical_only_by_method": summarize_group(na20_valid, ["method"]),
        "na20_canonical_only_by_method_polymer": summarize_group(na20_valid, ["method", "polymer_type"]),
        "na20_canonical_only_by_target_method": summarize_group(na20_valid, ["target_id", "method"]),
        "na20_canonical_only_by_variant": summarize_group(na20_valid, ["method", "variant", "polymer_type"]),
    }
    for name, rows in valid_summaries.items():
        write_csv(output_dir / f"{name}.csv", [{k: fmt(v) for k, v in row.items()} for row in rows])
        write_markdown(output_dir / f"{name}.md", rows, name)

    deltas = confsel_delta(confsel)
    write_csv(output_dir / "confsel_positive_negative_delta.csv", [{k: fmt(v) for k, v in row.items()} for row in deltas])
    write_markdown(output_dir / "confsel_positive_negative_delta.md", deltas, "Conformation-selective deltas", limit=80)
    write_csv(
        output_dir / "confsel_delta_summary.csv",
        [{k: fmt(v) for k, v in row.items()} for row in summarize_group(deltas, ["source_target_id", "variant"])],
    )

    print(f"[ok] merged_rows={len(all_rows)} na20={len(na20)} confsel={len(confsel)} output={output_dir}")


if __name__ == "__main__":
    main()
