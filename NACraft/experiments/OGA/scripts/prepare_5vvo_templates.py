#!/usr/bin/env python3
"""Create chain-specific 5VVO templates for native AF3 input JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gemmi


EXPECTED_RESOLVED = {"A": 437, "B": 429}
RELEASE_DATE = "2017-09-27"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_positions(chain: gemmi.Chain) -> list[int]:
    positions = [
        residue.label_seq
        for residue in chain
        if residue.entity_type == gemmi.EntityType.Polymer
    ]
    if len(positions) != len(set(positions)):
        raise ValueError(f"Duplicate label_seq_id values in chain {chain.name}")
    return positions


def extract_chain(source: Path, chain_id: str, output: Path) -> list[int]:
    structure = gemmi.read_structure(str(source)).clone()
    model = structure[0]
    if model.find_chain(chain_id) is None:
        raise ValueError(f"5VVO chain {chain_id} is missing")
    for chain in list(model):
        if chain.name != chain_id:
            model.remove_chain(chain.name)
    structure.remove_ligands_and_waters()
    chain = structure[0][chain_id]
    positions = resolved_positions(chain)
    expected = EXPECTED_RESOLVED[chain_id]
    if len(positions) != expected:
        raise ValueError(
            f"5VVO chain {chain_id}: expected {expected} resolved residues, "
            f"got {len(positions)}"
        )
    if min(positions) < 1 or max(positions) > 504:
        raise ValueError(f"5VVO chain {chain_id}: label_seq_id outside 1..504")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = structure.make_mmcif_document()
    revision_loop = document.sole_block().init_loop(
        "_pdbx_audit_revision_history.",
        [
            "ordinal",
            "data_content_type",
            "major_revision",
            "minor_revision",
            "revision_date",
        ],
    )
    revision_loop.add_row(["1", "'Structure model'", "1", "0", RELEASE_DATE])
    document.write_file(str(output))
    check_document = gemmi.cif.read_file(str(output))
    release_date = check_document.sole_block().find_value(
        "_pdbx_audit_revision_history.revision_date"
    )
    if release_date != RELEASE_DATE:
        raise ValueError(f"5VVO release date missing from {output}: {release_date!r}")
    check = gemmi.read_structure(str(output))
    chains = [chain.name for chain in check[0]]
    if chains != [chain_id]:
        raise ValueError(f"Expected only chain {chain_id} in {output}, got {chains}")
    if resolved_positions(check[0][chain_id]) != positions:
        raise ValueError(f"Resolved-position mapping changed while writing {output}")
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cif", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.source_cif).resolve()
    target = json.loads(Path(args.target_manifest).read_text())
    if target["entry_id"].upper() != "5VVO":
        raise SystemExit("Target manifest is not 5VVO")
    if target["sequence_length"] != 504 or target["protein_copy_count"] != 2:
        raise SystemExit("Target manifest is not OGA504x2")

    output_dir = Path(args.output_dir).resolve()
    records = {}
    for chain_id in ("A", "B"):
        output = output_dir / f"5VVO_chain_{chain_id}.cif"
        positions = extract_chain(source, chain_id, output)
        indices = [position - 1 for position in positions]
        records[chain_id] = {
            "entry_id": "5VVO",
            "source_chain": chain_id,
            "mmcifPath": str(output),
            "queryIndices": indices,
            "templateIndices": indices,
            "resolved_count": len(indices),
            "release_date": RELEASE_DATE,
            "mmcif_sha256": sha256_file(output),
        }

    manifest = {
        "protocol": "oga504x2_af3_5vvo_only_v1",
        "source_cif": str(source),
        "source_cif_sha256": sha256_file(source),
        "target_sequence_sha256": target["sequence_sha256"],
        "rna_template": None,
        "protein_templates": records,
    }
    manifest_path = output_dir / "template_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "resolved_counts": {key: value["resolved_count"] for key, value in records.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
