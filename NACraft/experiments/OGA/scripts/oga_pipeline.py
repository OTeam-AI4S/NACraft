#!/usr/bin/env python3
"""Prepare and validate the corrected OGA 5VVO 504x2 experiment inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


EXPECTED_ENTITY_LENGTH = 504
EXPECTED_PROTEIN_ASYMS = ["A", "B"]
EXPECTED_MUTATION = "D175N"
OLD_MODEL_TAG = "oga437_monomer"
PARENT_GROUPS = {
    ("NACraft_946", "OGA", "oga_denovo"): "denovo",
    ("NACraft_946", "OGA-A3", "oga_a3_guided"): "a3_guided",
    ("NACraft_redesign_0701", "OGA", "redesign_oga"): "prior_redesign",
}
AUTH_HOTSPOTS = [645, 648, 653, 656, 657, 659, 660, 680, 681, 688]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}\s+(.+?)\s*$", text)
    if not match:
        raise ValueError(f"Missing CIF scalar {key}")
    return match.group(1).strip().strip("'\"")


def _semicolon_value(text: str, key: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(key)}\s*\n;(.+?)\n;",
        text,
    )
    if not match:
        raise ValueError(f"Missing CIF semicolon value {key}")
    return "".join(match.group(1).split())


def _resolved_counts(text: str) -> dict[str, int]:
    seen: dict[str, set[int]] = {"A": set(), "B": set()}
    for line in text.splitlines():
        if not line.startswith("ATOM "):
            continue
        fields = line.split()
        if len(fields) < 19:
            continue
        asym = fields[18]
        if asym not in seen:
            continue
        try:
            label_seq_id = int(fields[16])
        except ValueError:
            continue
        seen[asym].add(label_seq_id)
    return {asym: len(values) for asym, values in seen.items()}


def auth_to_label_seq_id(auth_seq_id: int) -> int:
    """Map 5VVO author numbering to the 504-aa construct label numbering."""
    if auth_seq_id == 59:
        return 1
    if 60 <= auth_seq_id <= 400:
        return auth_seq_id - 58
    if 543 <= auth_seq_id <= 552:
        return auth_seq_id - 200
    if 553 <= auth_seq_id <= 704:
        return auth_seq_id - 200
    raise ValueError(f"Author residue {auth_seq_id} is outside the 5VVO construct")


def extract_oga_target(cif_path: Path) -> dict:
    text = Path(cif_path).read_text()
    sequence = _semicolon_value(
        text, "_entity_poly.pdbx_seq_one_letter_code_can"
    )
    if len(sequence) != EXPECTED_ENTITY_LENGTH:
        raise ValueError(
            f"Expected authoritative OGA entity length 504, got {len(sequence)}"
        )
    mutation_match = re.search(
        r"(?m)^1\s+polymer\s+man\s+'Protein O-GlcNAcase'.*?\s(D\d+[A-Z])\s+",
        text,
    )
    mutation = mutation_match.group(1) if mutation_match else EXPECTED_MUTATION
    strand_ids = [
        value.strip()
        for value in _scalar(text, "_entity_poly.pdbx_strand_id").split(",")
    ]
    if strand_ids != EXPECTED_PROTEIN_ASYMS:
        raise ValueError(f"Expected protein asym IDs A,B, got {strand_ids}")
    oligomer_count = int(_scalar(text, "_pdbx_struct_assembly.oligomeric_count"))
    if oligomer_count != 2:
        raise ValueError(f"Expected oligomeric count 2, got {oligomer_count}")
    resolved = _resolved_counts(text)
    if resolved != {"A": 437, "B": 429}:
        raise ValueError(f"Unexpected resolved coordinate counts: {resolved}")
    return {
        "entry_id": _scalar(text, "_entry.id"),
        "sequence": sequence,
        "sequence_length": len(sequence),
        "sequence_sha256": sha256_text(sequence),
        "protein_copy_count": oligomer_count,
        "protein_asym_ids": strand_ids,
        "resolved_residue_counts": resolved,
        "unresolved_residue_counts": {
            asym: len(sequence) - count for asym, count in resolved.items()
        },
        "mutation": mutation,
        "linker": "GGGGSGGGGS",
        "source_cif_sha256": sha256_file(Path(cif_path)),
        "target_system": "OGA504x2+RNA",
        "auth_hotspots": AUTH_HOTSPOTS,
        "label_hotspots": [auth_to_label_seq_id(x) for x in AUTH_HOTSPOTS],
    }


def read_parent_rows(full_table: Path) -> list[dict]:
    rows = []
    with Path(full_table).open(newline="") as handle:
        for source_row in csv.DictReader(handle):
            key = (
                source_row["source"],
                source_row["target"],
                source_row["set"],
            )
            group = PARENT_GROUPS.get(key)
            if group is None:
                continue
            seq = source_row["seq"].strip().upper()
            if not seq or set(seq) - set("ACGU"):
                raise ValueError(f"Invalid RNA sequence for {source_row['name']}")
            design_id = int(source_row["design_id"])
            parent_id = f"oga_{group}_{design_id:03d}"
            rows.append(
                {
                    "parent_id": parent_id,
                    "group": group,
                    "source": source_row["source"],
                    "source_set": source_row["set"],
                    "source_design_id": design_id,
                    "original_name": source_row["name"],
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "sequence_sha256": sha256_text(seq),
                    "old_invalid_max_iptm": source_row["iptm"],
                    "old_invalid_plddt": source_row["plddt"],
                    "invalid_target_model": OLD_MODEL_TAG,
                }
            )
    rows.sort(key=lambda row: (row["group"], row["source_design_id"]))
    counts = Counter(row["group"] for row in rows)
    expected = {"denovo": 200, "a3_guided": 200, "prior_redesign": 200}
    if counts != expected:
        raise ValueError(f"Expected 200 parents per group, got {dict(counts)}")
    if len({row["parent_id"] for row in rows}) != 600:
        raise ValueError("Parent IDs are not unique")
    return rows


def crosscheck_original_jsons(
    parents: Iterable[dict], denovo_json: Path, a3_json: Path
) -> None:
    lookup = {
        "denovo": json.loads(Path(denovo_json).read_text()),
        "a3_guided": json.loads(Path(a3_json).read_text()),
    }
    for group, records in lookup.items():
        if len(records) != 200:
            raise ValueError(f"{group} JSON expected 200 records, got {len(records)}")
        manifest = {
            int(row["source_design_id"]): row["sequence"]
            for row in parents
            if row["group"] == group
        }
        normalized = {int(key): value.upper() for key, value in records.items()}
        if manifest != normalized:
            raise ValueError(f"{group} full-table sequences disagree with JSON")


def read_wetlab_rows(mapping_csv: Path, parents: Iterable[dict]) -> list[dict]:
    by_name = {row["original_name"]: row for row in parents}
    rows = []
    with Path(mapping_csv).open(newline="") as handle:
        for source_row in csv.DictReader(handle):
            parent = by_name.get(source_row["original_name"])
            if parent is None:
                raise ValueError(
                    f"Wet-lab sequence {source_row['original_name']} is not in parents"
                )
            rows.append(
                {
                    "wetlab_name": source_row["new_name"],
                    "parent_id": parent["parent_id"],
                    "original_name": parent["original_name"],
                    "sequence": parent["sequence"],
                    "sequence_sha256": parent["sequence_sha256"],
                    "old_invalid_max_iptm": source_row["iptm"],
                    "old_invalid_plddt": source_row["plddt"],
                    "invalid_target_model": OLD_MODEL_TAG,
                }
            )
    if len(rows) != 20:
        raise ValueError(f"Expected 20 wet-lab rows, got {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, rows: Iterable[dict], id_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(f">{row[id_key]}\n{row['sequence']}\n")


def prepare(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root).resolve()
    cif = Path(args.source_cif).resolve()
    target = extract_oga_target(cif)
    parents = read_parent_rows(Path(args.parent_table))
    crosscheck_original_jsons(
        parents,
        Path(args.denovo_json),
        Path(args.similarity_json),
    )
    wetlab = read_wetlab_rows(
        Path(args.wetlab_mapping),
        parents,
    )

    (out_root / "targets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(cif, out_root / "targets/Human_O-GlcNAcase_5VVO.cif")
    (out_root / "targets/oga_5vvo_504aa.fasta").write_text(
        f">oga_5vvo_504aa\n{target['sequence']}\n"
    )
    (out_root / "targets/target_manifest.json").write_text(
        json.dumps(target, indent=2) + "\n"
    )
    write_csv(out_root / "manifests/parent_manifest.csv", parents)
    write_fasta(out_root / "sequences/parents/oga_parents.fasta", parents, "parent_id")
    write_csv(out_root / "manifests/wetlab_previous_audit.csv", wetlab)
    write_fasta(
        out_root / "sequences/wetlab_previous/oga_wetlab_previous.fasta",
        wetlab,
        "wetlab_name",
    )
    presearch_task = [
        {
            "name": "oga_5vvo_504aa_presearch",
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": target["sequence"],
                    }
                }
            ],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
        }
    ]
    presearch_dir = out_root / "presearch/input"
    presearch_dir.mkdir(parents=True, exist_ok=True)
    (presearch_dir / "presearch_input.json").write_text(
        json.dumps(presearch_task, indent=2) + "\n"
    )
    (presearch_dir / "presearch_manifest.json").write_text(
        json.dumps(
            {
                "name": "oga_5vvo_504aa_presearch",
                "sequence_length": 504,
                "sequence_sha256": target["sequence_sha256"],
                "protein_copies_in_complex": 2,
                "msa_searches": 1,
                "legacy_fallback_allowed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared corrected OGA inputs under {out_root}")
    print("parents=600 wetlab_previous=20 target_length=504 protein_copies=2")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source-cif", required=True)
    prepare_parser.add_argument("--parent-table", required=True)
    prepare_parser.add_argument("--denovo-json", required=True)
    prepare_parser.add_argument("--similarity-json", required=True)
    prepare_parser.add_argument("--wetlab-mapping", required=True)
    prepare_parser.add_argument("--out-root", default="NACraft/experiments/OGA/work")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)


if __name__ == "__main__":
    main()
