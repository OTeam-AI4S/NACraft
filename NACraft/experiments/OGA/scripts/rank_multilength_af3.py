#!/usr/bin/env python3
"""Extract and rank multilength OGA504x2 AF3 redesign validations."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

if "NACRAFT_DIR" in os.environ:
    NACRAFT_DIR = Path(os.environ["NACRAFT_DIR"])
else:
    NACRAFT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NACRAFT_DIR))

from utils.target_rmsd import TargetRMSDSpec, calculate_target_rmsd  # noqa: E402


EVIDENCE_FIELDS = (
    "candidate_id",
    "group",
    "parent_family",
    "sequence_length",
    "sequence",
    "seed",
    "complex_plddt",
    "rna_oga_pair_iptm",
    "rna_B_pair_iptm",
    "rna_C_pair_iptm",
    "oga_dimer_pair_iptm",
    "global_iptm",
    "target_RMSD",
    "target_RMSD_matched_CA",
    "target_RMSD_sequence_identity",
    "target_RMSD_chain_pairs",
    "target_RMSD_error",
    "af3_model_cif_path",
    "source_wave",
)


def write_csv(path: Path, rows: list[dict], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def newest_summary(candidate_dir: Path, replicate_index: int) -> Path:
    matches = [
        Path(path)
        for path in glob.glob(
            str(
                candidate_dir
                / f"replicate_{replicate_index}"
                / "*"
                / "*_summary_confidences.json"
            )
        )
    ]
    if not matches:
        raise RuntimeError(
            f"missing summary for {candidate_dir.name} replicate {replicate_index}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def extract(args: argparse.Namespace) -> None:
    root = Path(args.root)
    target_spec = TargetRMSDSpec(
        structure_path=str(Path(args.native_structure).resolve()),
        chains=tuple(args.native_chains),
        residue_ranges=tuple((None, None) for _ in args.native_chains),
    )
    rows = []
    candidates = 0
    tasks = [
        (wave, complete_path)
        for wave in args.waves
        for complete_path in sorted((root / wave).glob("*/complete.json"))
    ]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least one")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    tasks = [
        task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index
    ]
    for wave, complete_path in tasks:
        complete = json.loads(complete_path.read_text())
        candidate_dir = complete_path.parent
        raw_candidate_id = complete["candidate_id"]
        candidate_id = f"{args.candidate_prefix}{raw_candidate_id}"
        sequence = complete["sequence"]
        if complete.get("target_system") != "OGA504x2+RNA":
            raise RuntimeError(f"invalid target system for {candidate_id}")
        if complete.get("num_models") != 5 or len(complete.get("replicates", [])) != 5:
            raise RuntimeError(f"expected five AF3 seeds for {candidate_id}")
        group = raw_candidate_id.rsplit("_design", 1)[0]
        parent_family = f"{args.candidate_prefix}{raw_candidate_id.rsplit('_nampnn_', 1)[0]}"
        for replicate_index, replicate in enumerate(complete["replicates"], 1):
            plddt = replicate["metrics"].get("complex_plddt")
            if plddt is None:
                raise RuntimeError(
                    f"missing complex_plddt for {candidate_id} replicate {replicate_index}"
                )
            summary_path = newest_summary(candidate_dir, replicate_index)
            summary = json.loads(summary_path.read_text())
            pair = summary["chain_pair_iptm"]
            if len(pair) != 3 or any(len(row) != 3 for row in pair):
                raise RuntimeError(f"invalid pair matrix for {candidate_id}")
            rna_b = float(pair[0][1])
            rna_c = float(pair[0][2])
            model_paths = list(summary_path.parent.glob("*_model.cif"))
            if len(model_paths) != 1:
                raise RuntimeError(
                    f"expected one AF3 model for {candidate_id} replicate "
                    f"{replicate_index}, got {len(model_paths)}"
                )
            model_path = model_paths[0]
            rmsd_fields = {
                "target_RMSD": "",
                "target_RMSD_matched_CA": "",
                "target_RMSD_sequence_identity": "",
                "target_RMSD_chain_pairs": "",
                "target_RMSD_error": "",
            }
            try:
                rmsd = calculate_target_rmsd(model_path, target_spec)
                rmsd_fields.update(
                    {
                        "target_RMSD": rmsd.rmsd,
                        "target_RMSD_matched_CA": rmsd.matched_ca_atoms,
                        "target_RMSD_sequence_identity": rmsd.mean_sequence_identity,
                        "target_RMSD_chain_pairs": rmsd.matched_chain_pairs,
                    }
                )
            except Exception as exc:
                rmsd_fields["target_RMSD_error"] = str(exc)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "group": group,
                    "parent_family": parent_family,
                    "sequence_length": len(sequence),
                    "sequence": sequence,
                    "seed": replicate["seed"],
                    "complex_plddt": float(plddt),
                    "rna_oga_pair_iptm": max(rna_b, rna_c),
                    "rna_B_pair_iptm": rna_b,
                    "rna_C_pair_iptm": rna_c,
                    "oga_dimer_pair_iptm": float(pair[1][2]),
                    "global_iptm": float(summary["iptm"]),
                    **rmsd_fields,
                    "af3_model_cif_path": str(model_path),
                    "source_wave": wave,
                }
            )
        candidates += 1
    write_csv(Path(args.output), rows, EVIDENCE_FIELDS)
    print(json.dumps({"candidates": candidates, "replicates": len(rows)}))


def sequence_identity(left: str, right: str) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def read_evidence(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def rank(args: argparse.Namespace) -> None:
    evidence = read_evidence(args.evidence)
    by_candidate: dict[str, list[dict]] = {}
    for row in evidence:
        by_candidate.setdefault(row["candidate_id"], []).append(row)

    ranked = []
    excluded = 0
    for candidate_id, replicates in by_candidate.items():
        if len(replicates) != 5:
            raise RuntimeError(
                f"expected five evidence rows for {candidate_id}, got {len(replicates)}"
            )
        eligible = []
        for row in replicates:
            try:
                target_rmsd = float(row["target_RMSD"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                float(row["complex_plddt"]) > args.plddt_threshold
                and target_rmsd < args.target_rmsd_threshold
            ):
                eligible.append(row)
        if not eligible:
            excluded += 1
            continue
        best = max(
            eligible,
            key=lambda row: (
                float(row["rna_oga_pair_iptm"]),
                -float(row["target_RMSD"]),
                float(row["complex_plddt"]),
            ),
        )
        ranked.append(
            {
                "candidate_id": candidate_id,
                "group": best["group"],
                "parent_family": best["parent_family"],
                "sequence_length": int(best["sequence_length"]),
                "sequence": best["sequence"],
                "eligible_seed_count": len(eligible),
                "seed": int(best["seed"]),
                "complex_plddt": float(best["complex_plddt"]),
                "rna_oga_pair_iptm": float(best["rna_oga_pair_iptm"]),
                "rna_B_pair_iptm": float(best["rna_B_pair_iptm"]),
                "rna_C_pair_iptm": float(best["rna_C_pair_iptm"]),
                "oga_dimer_pair_iptm": float(best["oga_dimer_pair_iptm"]),
                "global_iptm": float(best["global_iptm"]),
                "target_RMSD": float(best["target_RMSD"]),
                "target_RMSD_matched_CA": int(best["target_RMSD_matched_CA"]),
                "target_RMSD_sequence_identity": float(
                    best["target_RMSD_sequence_identity"]
                ),
                "target_RMSD_chain_pairs": best["target_RMSD_chain_pairs"],
                "af3_model_cif_path": best["af3_model_cif_path"],
                "source_wave": best["source_wave"],
            }
        )
    ranked.sort(
        key=lambda row: (
            row["rna_oga_pair_iptm"],
            -row["target_RMSD"],
            row["complex_plddt"],
        ),
        reverse=True,
    )

    output = Path(args.output_dir)
    write_csv(output / "oga_target_rmsd_seed_evidence.csv", evidence, EVIDENCE_FIELDS)
    error_rows = [row for row in evidence if row.get("target_RMSD_error")]
    write_csv(output / "oga_target_rmsd_errors.csv", error_rows, EVIDENCE_FIELDS)
    ranked_rows = [{"rank": index, **row} for index, row in enumerate(ranked, 1)]
    fields = list(ranked_rows[0])
    write_csv(output / "oga_redesign_ranked.csv", ranked_rows, fields)
    top50 = ranked_rows[: args.top_pool]
    write_csv(output / f"oga_redesign_top{args.top_pool}.csv", top50, fields)

    length_limits = {}
    for spec in args.max_count_by_length:
        length, count = spec.split(":", 1)
        length_limits[int(length)] = int(count)
    selected = []
    family_counts: Counter[str] = Counter()
    selected_length_counts: Counter[int] = Counter()
    for row in top50:
        if family_counts[row["parent_family"]] >= args.max_per_parent_family:
            continue
        if (
            row["sequence_length"] in length_limits
            and selected_length_counts[row["sequence_length"]]
            >= length_limits[row["sequence_length"]]
        ):
            continue
        if any(
            sequence_identity(row["sequence"], old["sequence"])
            >= args.max_sequence_identity
            for old in selected
        ):
            continue
        selected.append(row)
        family_counts[row["parent_family"]] += 1
        selected_length_counts[row["sequence_length"]] += 1
        if len(selected) == args.select_count:
            break
    if len(selected) != args.select_count:
        raise RuntimeError(
            f"only {len(selected)} diverse candidates in top {args.top_pool}"
        )
    selection_rows = [
        {"selection_rank": index, "raw_rank": row["rank"], **{k: v for k, v in row.items() if k != "rank"}}
        for index, row in enumerate(selected, 1)
    ]
    write_csv(
        output / f"oga_redesign_top{args.select_count}_diverse.csv",
        selection_rows,
        list(selection_rows[0]),
    )
    with (output / f"oga_redesign_top{args.select_count}_diverse.fasta").open("w") as handle:
        for row in selection_rows:
            handle.write(
                f">OGA-{row['selection_rank']:02d}|{row['candidate_id']}|raw_rank={row['raw_rank']}\n"
                f"{row['sequence']}\n"
            )
    rmsd_values = [
        float(row["target_RMSD"])
        for row in evidence
        if row.get("target_RMSD") not in (None, "")
    ]
    summary = {
        "plddt_operator": ">",
        "plddt_threshold": args.plddt_threshold,
        "target_RMSD_operator": "<",
        "target_RMSD_threshold_angstrom": args.target_rmsd_threshold,
        "evidence_candidate_count": len(by_candidate),
        "evidence_seed_count": len(evidence),
        "target_RMSD_error_seed_count": len(error_rows),
        "eligible_seed_count": sum(
            1
            for row in evidence
            if not row.get("target_RMSD_error")
            and float(row["complex_plddt"]) > args.plddt_threshold
            and float(row["target_RMSD"]) < args.target_rmsd_threshold
        ),
        "eligible_candidate_count": len(ranked),
        "excluded_candidate_count": excluded,
        "top_pool_count": len(top50),
        "diverse_selection_count": len(selected),
        "diversity": {
            "maximum_per_parent_family": args.max_per_parent_family,
            "maximum_equal_length_sequence_identity_exclusive": args.max_sequence_identity,
            "maximum_count_by_length": length_limits,
        },
        "selected_group_counts": dict(Counter(row["group"] for row in selected)),
        "target_RMSD_distribution_angstrom": {
            "minimum": min(rmsd_values),
            "median": statistics.median(rmsd_values),
            "maximum": max(rmsd_values),
        },
        "target_RMSD_matched_CA_counts": dict(
            Counter(row["target_RMSD_matched_CA"] for row in evidence)
        ),
        "target_RMSD_chain_pair_counts": dict(
            Counter(row["target_RMSD_chain_pairs"] for row in evidence)
        ),
        "evidence_files": [Path(path).name for path in args.evidence],
    }
    if args.previous_selection:
        with Path(args.previous_selection).open(newline="") as handle:
            previous = list(csv.DictReader(handle))
        previous_ids = [row["candidate_id"] for row in previous]
        current_ids = [row["candidate_id"] for row in selection_rows]
        comparison = {
            "previous_selection": str(Path(args.previous_selection)),
            "same_member_set": set(previous_ids) == set(current_ids),
            "added": sorted(set(current_ids) - set(previous_ids)),
            "removed": sorted(set(previous_ids) - set(current_ids)),
            "rank_changes": [
                {
                    "candidate_id": candidate_id,
                    "previous_rank": previous_ids.index(candidate_id) + 1,
                    "current_rank": current_ids.index(candidate_id) + 1,
                }
                for candidate_id in current_ids
                if candidate_id in previous_ids
                and previous_ids.index(candidate_id) != current_ids.index(candidate_id)
            ],
        }
        (output / "comparison_vs_previous.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        summary["previous_selection_comparison"] = comparison
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--root", required=True)
    extract_parser.add_argument("--waves", nargs="+", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--native-structure", required=True)
    extract_parser.add_argument("--native-chains", nargs="+", default=["A", "B"])
    extract_parser.add_argument("--candidate-prefix", default="")
    extract_parser.add_argument("--num-shards", type=int, default=1)
    extract_parser.add_argument("--shard-index", type=int, default=0)
    extract_parser.set_defaults(func=extract)

    rank_parser = subparsers.add_parser("rank")
    rank_parser.add_argument("--evidence", action="append", required=True)
    rank_parser.add_argument("--output-dir", required=True)
    rank_parser.add_argument("--plddt-threshold", type=float, default=0.6)
    rank_parser.add_argument("--target-rmsd-threshold", type=float, default=10.0)
    rank_parser.add_argument("--top-pool", type=int, default=50)
    rank_parser.add_argument("--select-count", type=int, default=20)
    rank_parser.add_argument("--max-sequence-identity", type=float, default=0.90)
    rank_parser.add_argument("--max-count-by-length", action="append", default=[])
    rank_parser.add_argument("--max-per-parent-family", type=int, default=1)
    rank_parser.add_argument("--previous-selection")
    rank_parser.set_defaults(func=rank)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
