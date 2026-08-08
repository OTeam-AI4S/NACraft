#!/usr/bin/env python3
"""Prepare corrected OGA504x2 NACraft multilength production configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


A3_SEQUENCE = "GGGCGACUGCUGAGUGACAUACCAUCUGGUUCGUCACGAAG"
GROUPS = {
    "denovo_20": {"length": 20, "mode": "denovo", "init_seq": ""},
    "denovo_40": {"length": 40, "mode": "denovo", "init_seq": ""},
    "denovo_60": {"length": 60, "mode": "denovo", "init_seq": ""},
    "denovo_80": {"length": 80, "mode": "denovo", "init_seq": ""},
    "a3_guided_41": {
        "length": 41,
        "mode": "a3_guided",
        "init_seq": A3_SEQUENCE,
    },
}


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def quote(value: str) -> str:
    return json.dumps(value)


def render_config(group: str, spec: dict, protein: str) -> str:
    losses = ["  - type: LigandContactLoss", "    state: 0"]
    if spec["mode"] == "a3_guided":
        losses.extend(
            [
                "  - type: SequenceSimilarityLoss",
                f"    target_sequence: {A3_SEQUENCE}",
                "    strength: 0.5",
            ]
        )
    return "\n".join(
        [
            f"# Corrected OGA504x2 multilength group: {group}",
            "polymer_type: rna",
            "predictor: boltz",
            "num_states: 1",
            f"length: {spec['length']}",
            "motifs: []",
            "states:",
            f"  - [{quote('protein:' + protein)}, {quote('protein:' + protein)}]",
            "losses:",
            *losses,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--presearch-json", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    target = json.loads(Path(args.target_manifest).read_text())
    protein = target["sequence"]
    if len(protein) != 504 or target.get("protein_copy_count") != 2:
        raise SystemExit("Refusing non-OGA504x2 target")
    presearch = json.loads(Path(args.presearch_json).read_text())
    if list(presearch) != [protein]:
        raise SystemExit("Presearch must contain exactly the authoritative 504-aa key")
    entry = presearch[protein]
    paths = [entry["unpairedMsaPath"], entry["pairedMsaPath"]] + [
        record["mmcifPath"] for record in entry["templates"]
    ]
    if not all(Path(path).is_file() for path in paths):
        raise SystemExit("One or more corrected presearch paths are missing")

    root = Path(args.out_root)
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for group, spec in GROUPS.items():
        config_path = config_dir / f"{group}.yaml"
        config_path.write_text(render_config(group, spec, protein))
        manifest.append(
            {
                "group": group,
                **spec,
                "parent_count": 100,
                "children_per_parent": 3,
                "child_count": 300,
                "target_system": "OGA504x2+RNA",
                "protein_sequence_sha256": target["sequence_sha256"],
                "presearch_sequence_sha256": sha256(protein),
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        )
    (root / "group_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "README.txt").write_text(
        "Corrected OGA504x2 multilength production. Raw outputs are external to Git.\n"
    )
    print(f"prepared {len(manifest)} groups under {root}")
    print("parents=500 children=1500 af3_candidates=2000")


if __name__ == "__main__":
    main()
