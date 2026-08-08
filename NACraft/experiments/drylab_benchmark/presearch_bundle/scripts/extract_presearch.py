#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract_one(data_json_path: Path, out_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[dict]]]:
    data = json.loads(data_json_path.read_text())
    stem = data_json_path.name[: -len("_data.json")]
    target_dir = out_dir / stem
    target_dir.mkdir(parents=True, exist_ok=True)

    seq_to_msa = {}
    seq_to_templates = {}
    for entity in data["sequences"]:
        if "protein" not in entity:
            continue
        protein = entity["protein"]
        seq = protein["sequence"]
        seq_id = protein["id"] if isinstance(protein["id"], str) else "_".join(protein["id"])

        unpaired_path = target_dir / f"{seq_id}_unpaired_msa.a3m"
        paired_path = target_dir / f"{seq_id}_paired_msa.a3m"
        unpaired_path.write_text(protein.get("unpairedMsa", ""))
        paired_path.write_text(protein.get("pairedMsa", ""))

        templates = []
        for idx, tpl in enumerate(protein.get("templates", [])):
            cif_path = target_dir / f"{seq_id}_{idx}.cif"
            cif_path.write_text(tpl["mmcif"])
            templates.append(
                {
                    "mmcifPath": str(cif_path),
                    "queryIndices": tpl["queryIndices"],
                    "templateIndices": tpl["templateIndices"],
                }
            )

        seq_to_msa[seq] = {"unpairedMsaPath": str(unpaired_path), "pairedMsaPath": str(paired_path)}
        seq_to_templates[seq] = templates
    return seq_to_msa, seq_to_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AF3 MSA/templates from *_data.json into reusable index.json.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--presearch-outdir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.presearch_outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_jsons = sorted(root.rglob("*_data.json"))
    if not data_jsons:
        print(f"[ERR] no *_data.json under {root}", file=sys.stderr)
        sys.exit(2)

    merged_msa = {}
    merged_templates = {}
    for path in data_jsons:
        seq_to_msa, seq_to_templates = extract_one(path, out_dir)
        for seq, msa_paths in seq_to_msa.items():
            merged_msa[seq] = msa_paths
            merged_templates[seq] = seq_to_templates.get(seq, [])

    index = {}
    for seq, msa_paths in merged_msa.items():
        index[seq] = {
            "unpairedMsaPath": msa_paths["unpairedMsaPath"],
            "pairedMsaPath": msa_paths["pairedMsaPath"],
            "templates": merged_templates.get(seq, []),
        }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    print(f"wrote {out_dir / 'index.json'} with {len(index)} target sequence(s)")
    for idx, seq in enumerate(index, start=1):
        print(f"{idx:2d}. len={len(seq):4d} templates={len(index[seq].get('templates', []))} first40={seq[:40]}...")


if __name__ == "__main__":
    main()
