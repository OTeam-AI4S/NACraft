#!/usr/bin/env python3
"""Aggregate genuine AF3 replicates and select corrected OGA wet-lab candidates."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np


PROTOCOL = "five_independent_seeds_v1"
CONTACT_CUTOFF = 6.0
# Calibrated for the RNA-versus-(OGA504)2 chain-pair metric. In the complete
# 3,600-candidate independent-seed population, the median pair-ipTM 99th
# percentile is ~0.24; 0.20 retains the reproducible high-confidence tail.
PAIR_IPTM_MIN = 0.20
DIMER_IPTM_MIN = 0.75
MIN_PASS_COUNT = 3


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_population(root: Path) -> list[dict]:
    parents = read_csv(root / "manifests/parent_manifest.csv")
    candidates = [
        {
            "candidate_id": row["parent_id"],
            "parent_id": row["parent_id"],
            "candidate_type": "parent",
            "group": row["group"],
            "sequence": row["sequence"],
            "sequence_length": int(row["sequence_length"]),
            "sequence_sha256": row["sequence_sha256"],
        }
        for row in parents
    ]
    for parent in parents:
        child_file = root / "nampnn_children" / parent["parent_id"] / "complete.json"
        for child in json.loads(child_file.read_text())["children"]:
            candidates.append(
                {
                    "candidate_id": child["child_id"],
                    "parent_id": child["parent_id"],
                    "candidate_type": "nampnn_child",
                    "group": parent["group"],
                    "sequence": child["sequence"],
                    "sequence_length": int(child["sequence_length"]),
                    "sequence_sha256": child["sequence_sha256"],
                }
            )
    if len(candidates) != 3600:
        raise RuntimeError(f"Expected 3,600 candidates, got {len(candidates)}")
    return candidates


def atom_positions(chain: gemmi.Chain) -> np.ndarray:
    return np.asarray(
        [[atom.pos.x, atom.pos.y, atom.pos.z] for residue in chain for atom in residue],
        dtype=float,
    )


def pocket_metrics(cif_path: Path, hotspot_ids: list[int]) -> dict:
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    chains = {chain.name: chain for chain in model}
    if list(chains) != ["A", "B", "C"]:
        raise RuntimeError(f"Unexpected chains in {cif_path}: {list(chains)}")
    if [len(chains[x]) for x in ("A", "B", "C")] != [
        len(chains["A"]),
        504,
        504,
    ]:
        raise RuntimeError(f"Protein chain length failure: {cif_path}")
    rna = atom_positions(chains["A"])
    per_chain = {}
    for chain_id in ("B", "C"):
        distances = []
        for residue_id in hotspot_ids:
            residues = [r for r in chains[chain_id] if r.seqid.num == residue_id]
            if len(residues) != 1:
                raise RuntimeError(
                    f"Hotspot {chain_id}:{residue_id} absent or duplicated in {cif_path}"
                )
            pocket_atoms = np.asarray(
                [[a.pos.x, a.pos.y, a.pos.z] for a in residues[0]], dtype=float
            )
            distance = float(
                np.linalg.norm(rna[:, None, :] - pocket_atoms[None, :, :], axis=-1).min()
            )
            distances.append(distance)
        per_chain[chain_id] = {
            "coverage": sum(value < CONTACT_CUTOFF for value in distances),
            "min_distance": min(distances),
        }
    best_chain = max(
        ("B", "C"),
        key=lambda x: (per_chain[x]["coverage"], -per_chain[x]["min_distance"]),
    )
    return {
        "pocket_chain": best_chain,
        "pocket_coverage": per_chain[best_chain]["coverage"],
        "pocket_min_distance": per_chain[best_chain]["min_distance"],
        "pocket_B_coverage": per_chain["B"]["coverage"],
        "pocket_C_coverage": per_chain["C"]["coverage"],
    }


def find_single(pattern: str) -> Path:
    paths = [Path(path) for path in glob.glob(pattern)]
    if len(paths) != 1:
        raise RuntimeError(f"Expected one path for {pattern}, got {len(paths)}")
    return paths[0]


def aggregate_candidate(root: Path, candidate: dict, hotspots: list[int]) -> dict:
    mode = "parent" if candidate["candidate_type"] == "parent" else "child"
    candidate_dir = root / f"af3_{mode}" / candidate["candidate_id"]
    complete = json.loads((candidate_dir / "complete.json").read_text())
    if complete.get("validation_protocol") != PROTOCOL:
        raise RuntimeError(f"Invalid AF3 protocol: {candidate['candidate_id']}")
    replicates = []
    for replicate_index in range(1, 6):
        replicate_dir = candidate_dir / f"replicate_{replicate_index}"
        summary_path = find_single(str(replicate_dir / "*" / "*_summary_confidences.json"))
        cif_path = find_single(str(replicate_dir / "*" / "*_model.cif"))
        summary = json.loads(summary_path.read_text())
        pair = summary["chain_pair_iptm"]
        if len(pair) != 3 or any(len(row) != 3 for row in pair):
            raise RuntimeError(f"Invalid chain-pair matrix: {summary_path}")
        pocket = pocket_metrics(cif_path, hotspots)
        rna_pair = max(float(pair[0][1]), float(pair[0][2]))
        dimer_pair = float(pair[1][2])
        passed = (
            rna_pair >= PAIR_IPTM_MIN
            and dimer_pair >= DIMER_IPTM_MIN
            and pocket["pocket_coverage"] >= 1
            and not bool(summary.get("has_clash", False))
        )
        replicates.append(
            {
                "seed": complete["seeds"][replicate_index - 1],
                "iptm": float(summary["iptm"]),
                "ptm": float(summary["ptm"]),
                "ranking_score": float(summary["ranking_score"]),
                "rna_oga_pair_iptm": rna_pair,
                "oga_dimer_pair_iptm": dimer_pair,
                "has_clash": bool(summary.get("has_clash", False)),
                "passed": passed,
                **pocket,
            }
        )
    pair_values = [row["rna_oga_pair_iptm"] for row in replicates]
    dimer_values = [row["oga_dimer_pair_iptm"] for row in replicates]
    coverages = [row["pocket_coverage"] for row in replicates]
    contacting = sum(value >= 1 for value in coverages)
    pass_count = sum(row["passed"] for row in replicates)
    pocket_chain_counts = Counter(row["pocket_chain"] for row in replicates)
    pocket_stability = contacting / 5.0
    median_pair = median(pair_values)
    median_dimer = median(dimer_values)
    composite = (
        0.45 * median_pair
        + 0.30 * pocket_stability
        + 0.15 * (pass_count / 5.0)
        + 0.10 * median_dimer
    )
    return {
        **candidate,
        "max_iptm": max(row["iptm"] for row in replicates),
        "median_iptm": median([row["iptm"] for row in replicates]),
        "max_rna_oga_pair_iptm": max(pair_values),
        "median_rna_oga_pair_iptm": median_pair,
        "median_oga_dimer_pair_iptm": median_dimer,
        "median_pocket_coverage": median(coverages),
        "pocket_contact_replicates": contacting,
        "pocket_stability": pocket_stability,
        "replicate_pass_count": pass_count,
        "dominant_pocket_chain": pocket_chain_counts.most_common(1)[0][0],
        "dominant_pocket_chain_count": pocket_chain_counts.most_common(1)[0][1],
        "clash_replicates": sum(row["has_clash"] for row in replicates),
        "hard_filter_pass": pass_count >= MIN_PASS_COUNT,
        "composite_score": composite,
        "replicate_json": json.dumps(replicates, separators=(",", ":")),
    }


def refresh_threshold_metrics(row: dict) -> dict:
    """Recompute threshold-dependent fields from stored per-replicate evidence."""
    replicates = json.loads(row["replicate_json"])
    for replicate in replicates:
        replicate["passed"] = (
            replicate["rna_oga_pair_iptm"] >= PAIR_IPTM_MIN
            and replicate["oga_dimer_pair_iptm"] >= DIMER_IPTM_MIN
            and replicate["pocket_coverage"] >= 1
            and not replicate["has_clash"]
        )
    pass_count = sum(replicate["passed"] for replicate in replicates)
    median_pair = median([replicate["rna_oga_pair_iptm"] for replicate in replicates])
    median_dimer = median([replicate["oga_dimer_pair_iptm"] for replicate in replicates])
    contacting = sum(replicate["pocket_coverage"] >= 1 for replicate in replicates)
    pocket_stability = contacting / 5.0
    row.update(
        {
            "sequence_length": int(row["sequence_length"]),
            "max_iptm": float(row["max_iptm"]),
            "median_iptm": float(row["median_iptm"]),
            "max_rna_oga_pair_iptm": max(
                replicate["rna_oga_pair_iptm"] for replicate in replicates
            ),
            "median_rna_oga_pair_iptm": median_pair,
            "median_oga_dimer_pair_iptm": median_dimer,
            "median_pocket_coverage": median(
                [replicate["pocket_coverage"] for replicate in replicates]
            ),
            "pocket_contact_replicates": contacting,
            "pocket_stability": pocket_stability,
            "replicate_pass_count": pass_count,
            "dominant_pocket_chain_count": int(row["dominant_pocket_chain_count"]),
            "clash_replicates": sum(replicate["has_clash"] for replicate in replicates),
            "hard_filter_pass": pass_count >= MIN_PASS_COUNT,
            "composite_score": (
                0.45 * median_pair
                + 0.30 * pocket_stability
                + 0.15 * (pass_count / 5.0)
                + 0.10 * median_dimer
            ),
            "replicate_json": json.dumps(replicates, separators=(",", ":")),
        }
    )
    return row


def sequence_identity(left: str, right: str) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def select_diverse(rows: list[dict], count: int = 40) -> list[dict]:
    selected = []
    family_counts = Counter()
    seen_sequences = set()
    for row in rows:
        if not row["hard_filter_pass"] or row["sequence"] in seen_sequences:
            continue
        if family_counts[row["parent_id"]] >= 2:
            continue
        if any(sequence_identity(row["sequence"], x["sequence"]) >= 0.90 for x in selected):
            continue
        selected.append(row)
        seen_sequences.add(row["sequence"])
        family_counts[row["parent_id"]] += 1
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} diverse passing candidates for top 40")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    target = json.loads((root / "targets/target_manifest.json").read_text())
    candidates = candidate_population(root)
    output = root / "analysis"
    cached_table = output / "all_candidates.csv"
    if cached_table.exists():
        cached = read_csv(cached_table)
        if len(cached) == 3600 and all(row.get("replicate_json") for row in cached):
            rows = [refresh_threshold_metrics(row) for row in cached]
        else:
            rows = [
                aggregate_candidate(root, row, target["label_hotspots"])
                for row in candidates
            ]
    else:
        rows = [
            aggregate_candidate(root, row, target["label_hotspots"])
            for row in candidates
        ]
    rows.sort(
        key=lambda row: (
            row["hard_filter_pass"],
            row["composite_score"],
            row["median_rna_oga_pair_iptm"],
            row["pocket_stability"],
            row["replicate_pass_count"],
        ),
        reverse=True,
    )
    write_csv(output / "all_candidates.csv", rows)

    selected = select_diverse(rows)
    selection_rows = []
    for rank, row in enumerate(selected, 1):
        selection_rows.append(
            {
                "rank": rank,
                "selection_set": "top20" if rank <= 20 else "reserve21_40",
                **{key: value for key, value in row.items() if key != "replicate_json"},
            }
        )
    write_csv(output / "top40_selection.csv", selection_rows)
    with (output / "top20.fasta").open("w") as handle:
        for row in selection_rows[:20]:
            handle.write(f">OGA-{int(row['rank']):02d}|{row['candidate_id']}\n{row['sequence']}\n")

    wetlab = read_csv(root / "manifests/wetlab_previous_audit.csv")
    by_id = {row["candidate_id"]: row for row in rows}
    comparison = []
    for old in wetlab:
        new = by_id[old["parent_id"]]
        comparison.append(
            {
                **old,
                "corrected_global_max_iptm_not_directly_comparable": new["max_iptm"],
                "corrected_global_median_iptm_not_directly_comparable": new[
                    "median_iptm"
                ],
                "corrected_max_rna_oga_pair_iptm": new["max_rna_oga_pair_iptm"],
                "corrected_median_rna_oga_pair_iptm": new[
                    "median_rna_oga_pair_iptm"
                ],
                "corrected_pocket_stability": new["pocket_stability"],
                "corrected_replicate_pass_count": new["replicate_pass_count"],
                "corrected_hard_filter_pass": new["hard_filter_pass"],
                "delta_pair_iptm_vs_old_invalid_iptm": new[
                    "max_rna_oga_pair_iptm"
                ]
                - float(old["old_invalid_max_iptm"]),
                "comparison_note": (
                    "Primary delta uses corrected RNA-OGA pair ipTM. Corrected "
                    "global ipTM includes the OGA-OGA dimer and is not directly "
                    "comparable with the old monomer-complex ipTM."
                ),
            }
        )
    write_csv(output / "wetlab20_before_after.csv", comparison)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "candidate_count": len(rows),
                "hard_filter_pass_count": sum(row["hard_filter_pass"] for row in rows),
                "top20_count": 20,
                "reserve_count": 20,
                "wetlab_comparison_count": len(comparison),
                "thresholds": {
                    "contact_cutoff_angstrom": CONTACT_CUTOFF,
                    "rna_oga_pair_iptm_min": PAIR_IPTM_MIN,
                    "oga_dimer_pair_iptm_min": DIMER_IPTM_MIN,
                    "minimum_replicate_pass_count": MIN_PASS_COUNT,
                    "sequence_identity_exclusion": 0.90,
                    "maximum_per_parent_family": 2,
                },
                "threshold_calibration": {
                    "rna_oga_pair_iptm_basis": (
                        "0.20 selects the reproducible high tail; the complete "
                        "population median pair-ipTM 99th percentile is ~0.24"
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"ranked={len(rows)} passing={sum(row['hard_filter_pass'] for row in rows)} "
        f"top20=20 wetlab=20"
    )


if __name__ == "__main__":
    main()
