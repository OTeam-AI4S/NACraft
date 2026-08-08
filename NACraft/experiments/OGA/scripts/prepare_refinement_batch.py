#!/usr/bin/env python3
"""Prepare an OGA504x2 de-novo and top-candidate-guided refinement batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def quote(value: str) -> str:
    return json.dumps(value)


def render_config(group: dict, protein: str) -> str:
    lines = [
        f"# Corrected OGA504x2 refinement group: {group['group']}",
        "polymer_type: rna",
        "predictor: boltz",
        "num_states: 1",
        f"length: {group['length']}",
        "motifs: []",
        "states:",
        f"  - [{quote('protein:' + protein)}, {quote('protein:' + protein)}]",
        "losses:",
        "  - type: LigandContactLoss",
        "    state: 0",
    ]
    if group["mode"] == "sim_guided":
        lines.extend(
            [
                "  - type: SequenceSimilarityLoss",
                f"    target_sequence: {group['init_seq']}",
                "    strength: 0.5",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--top-selection", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--denovo-per-length", type=int, default=10)
    parser.add_argument("--guided-per-sequence", type=int, default=6)
    parser.add_argument("--guided-count", type=int, default=5)
    args = parser.parse_args()

    target = json.loads(Path(args.target_manifest).read_text())
    protein = target["sequence"]
    if len(protein) != 504 or target.get("protein_copy_count") != 2:
        raise SystemExit("Refusing non-OGA504x2 target")
    with Path(args.top_selection).open(newline="") as handle:
        top = list(csv.DictReader(handle))[: args.guided_count]
    if len(top) != args.guided_count:
        raise RuntimeError(f"expected {args.guided_count} guided priors, got {len(top)}")

    groups = [
        {
            "group": f"denovo_{length}",
            "mode": "denovo",
            "length": length,
            "init_seq": "",
            "parent_count": args.denovo_per_length,
            "source_candidate_id": "",
            "source_rank": "",
        }
        for length in (20, 40, 60)
    ]
    for rank, row in enumerate(top, 1):
        sequence = row["sequence"].upper()
        if not sequence or set(sequence) - set("ACGU"):
            raise RuntimeError(f"invalid guided sequence at rank {rank}")
        groups.append(
            {
                "group": f"guided_rank{rank:02d}",
                "mode": "sim_guided",
                "length": len(sequence),
                "init_seq": sequence,
                "parent_count": args.guided_per_sequence,
                "source_candidate_id": row["candidate_id"],
                "source_rank": rank,
                "source_pair_iptm": row.get("rna_oga_pair_iptm", ""),
                "source_target_RMSD": row.get("target_RMSD", ""),
            }
        )

    root = Path(args.out_root)
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    task_offset = 0
    for group in groups:
        config_path = config_dir / f"{group['group']}.yaml"
        config_path.write_text(render_config(group, protein))
        group["task_offset"] = task_offset
        group["child_count"] = group["parent_count"] * 10
        group["config_path"] = str(config_path)
        group["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        group["target_system"] = "OGA504x2+RNA"
        task_offset += group["parent_count"]
    manifest = {
        "protocol": "oga504x2_refinement_30denovo_30guided_v1",
        "protein_sequence_sha256": target["sequence_sha256"],
        "parent_count": sum(group["parent_count"] for group in groups),
        "children_per_parent": 10,
        "child_count": sum(group["child_count"] for group in groups),
        "groups": groups,
    }
    if manifest["parent_count"] != 60 or manifest["child_count"] != 600:
        raise RuntimeError(f"unexpected batch counts: {manifest}")
    (root / "group_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("parent_count", "child_count")}))


if __name__ == "__main__":
    main()
