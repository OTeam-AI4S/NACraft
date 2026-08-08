#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks).upper()))
            header = line[1:]
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    return records


def design_index(path: Path) -> int | None:
    if not path.name.startswith("design"):
        return None
    suffix = path.name.removeprefix("design")
    if not suffix.isdigit():
        return None
    return int(suffix)


def normalize_sequence(sequence: str) -> str:
    return sequence.split("/", 1)[0].upper()


def collect_config(
    config_dir: Path,
    target_id: str,
    method: str,
    design_limit: int,
    redesign_limit: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    design_dirs = []
    for design_dir in config_dir.glob("design*"):
        idx = design_index(design_dir)
        if idx is None or idx >= design_limit:
            continue
        design_dirs.append((idx, design_dir))
    for idx, design_dir in sorted(design_dirs):
        fasta = design_dir / "nampnn" / "seqs" / "composite.fa"
        if not fasta.exists():
            continue
        records = read_fasta(fasta)
        if not records:
            continue
        rows.append(
            {
                "target_id": target_id,
                "method": method,
                "candidate_id": f"design{idx:03d}_opt",
                "sequence": normalize_sequence(records[0][1]),
                "source_design": f"design{idx}",
                "variant": "opt",
                "variant_index": "0",
                "source_fasta": str(fasta),
            }
        )
        redesign_records = records[1 : 1 + redesign_limit]
        for ridx, (_header, seq) in enumerate(redesign_records, start=1):
            rows.append(
                {
                    "target_id": target_id,
                    "method": method,
                    "candidate_id": f"design{idx:03d}_redesign{ridx}",
                    "sequence": normalize_sequence(seq),
                    "source_design": f"design{idx}",
                    "variant": "redesign",
                    "variant_index": str(ridx),
                    "source_fasta": str(fasta),
                }
            )
    return rows


def collect_candidates(
    root: Path,
    design_limit_default: int,
    redesign_limit: int,
    long_prefixes: tuple[str, ...],
    long_design_limit: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        method_name = method_dir.name
        for target_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            target_id = target_dir.name
            design_limit = (
                long_design_limit
                if long_prefixes and target_id.startswith(long_prefixes)
                else design_limit_default
            )
            rows.extend(
                collect_config(
                    config_dir=target_dir,
                    target_id=target_id,
                    method=f"nacraft_{method_name}",
                    design_limit=design_limit,
                    redesign_limit=redesign_limit,
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "target_id",
        "method",
        "candidate_id",
        "sequence",
        "source_design",
        "variant",
        "variant_index",
        "source_fasta",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NACraft opt/redesign sequences for AF3 validation.")
    parser.add_argument("--root", required=True, help="Candidate root containing method/target/design* directories.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--design-limit", type=int, default=100)
    parser.add_argument("--redesign-limit", type=int, default=2)
    parser.add_argument("--long-prefix", action="append", default=[])
    parser.add_argument("--long-design-limit", type=int, default=10)
    args = parser.parse_args()

    rows = collect_candidates(
        root=Path(args.root),
        design_limit_default=args.design_limit,
        redesign_limit=args.redesign_limit,
        long_prefixes=tuple(args.long_prefix),
        long_design_limit=args.long_design_limit,
    )
    write_csv(Path(args.output), rows)
    print(f"Wrote {len(rows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
