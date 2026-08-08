#!/usr/bin/env python3
"""Generate exactly five RNA-only NA-MPNN children per corrected OGA parent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

NACRAFT_DIR = Path(
    os.environ.get("NACRAFT_DIR", Path(__file__).resolve().parents[3])
)
sys.path.insert(0, str(NACRAFT_DIR))

from utils.na_mpnn_utils import perform_tied_nampnn_redesign  # noqa: E402


def digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--boltz-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()

    with Path(args.manifest).open(newline="") as handle:
        parents = list(csv.DictReader(handle))
    if len(parents) != 600:
        raise SystemExit(f"Expected 600 parents, got {len(parents)}")

    for parent in parents[args.worker_id :: args.num_workers]:
        parent_id = parent["parent_id"]
        design_dir = Path(args.boltz_root) / parent_id
        out_dir = Path(args.output_root) / parent_id
        sentinel = out_dir / "complete.json"
        if sentinel.exists():
            print(f"skip {parent_id}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if not (design_dir / "complete.json").exists():
                raise FileNotFoundError(f"Boltz parent incomplete: {design_dir}")
            with (design_dir / "state0.pkl").open("rb") as handle:
                output = pickle.load(handle)
            sequences, fasta_path, best = perform_tied_nampnn_redesign(
                design_dir=str(design_dir),
                state_results=[(output, [None] * 5, 0)],
                polymer_type="rna",
                num_seqs=5,
                motif_indices=None,
            )
            valid = []
            for sequence in sequences:
                sequence = sequence.upper()
                if set(sequence) <= set("ACGU") and len(sequence) == int(parent["sequence_length"]):
                    valid.append(sequence)
            if len(valid) < 5:
                raise RuntimeError(f"NA-MPNN returned only {len(valid)} valid children")
            children = []
            for child_index, sequence in enumerate(valid[:5], 1):
                children.append(
                    {
                        "child_id": f"{parent_id}_nampnn_{child_index}",
                        "parent_id": parent_id,
                        "child_index": child_index,
                        "sequence": sequence,
                        "sequence_length": len(sequence),
                        "sequence_sha256": digest(sequence),
                        "parent_sequence_sha256": parent["sequence_sha256"],
                        "selected_boltz_sample": int(best[0]),
                        "seed": 0,
                        "temperature": 0.1,
                    }
                )
            sentinel.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "parent_id": parent_id,
                        "source_fasta": fasta_path,
                        "children": children,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"complete {parent_id}: 5 children", flush=True)
        except Exception as error:
            (out_dir / "failed.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "parent_id": parent_id,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"FAILED {parent_id}: {error}", flush=True)


if __name__ == "__main__":
    main()
