#!/usr/bin/env python3
"""Generate validated RNA-only NA-MPNN children for one completed OGA parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

NACRAFT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NACRAFT_DIR))

from utils.na_mpnn_utils import perform_tied_nampnn_redesign  # noqa: E402


GROUPS = ("denovo_20", "denovo_40", "a3_guided_41", "denovo_60")


def _digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def _final_parent_sequence(trace_path: Path) -> str:
    rows = [line.rstrip("\n").split("\t") for line in trace_path.read_text().splitlines()[1:]]
    argmax = [row for row in rows if len(row) > 4 and row[1] == "argmax"]
    if argmax:
        return argmax[-1][3]
    valid = [row for row in rows if len(row) > 4 and row[3]]
    if not valid:
        raise RuntimeError(f"no optimization sequence in {trace_path}")
    # Early-stopped OGA parents may finish in exploration or annealing before
    # an argmax stage exists. Structure generation uses the current sequence,
    # which is the final sequence recorded in the trace.
    return valid[-1][3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--group-index", type=int)
    parser.add_argument("--group-name")
    parser.add_argument("--parent-index", default=0, type=int)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--parents-per-group", default=50, type=int)
    parser.add_argument("--num-seqs", default=3, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.group_name is not None:
        group = args.group_name
        parent_index = args.parent_index
    elif args.task_index is not None:
        group_index, parent_index = divmod(args.task_index, args.parents_per_group)
    else:
        if args.group_index is None:
            parser.error("one of --task-index or --group-index is required")
        group_index, parent_index = args.group_index, args.parent_index
    if args.group_name is None:
        if not 0 <= group_index < len(GROUPS):
            raise ValueError(f"group index out of range: {group_index}")
        group = GROUPS[group_index]
    root = Path(args.root)
    design_dir = root / "designs" / group / f"design{parent_index}"
    parent_id = f"{group}_design{parent_index}"
    output_dir = root / "nampnn_children" / parent_id
    sentinel = output_dir / "complete.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if sentinel.exists() and not args.overwrite:
        print(f"skip {parent_id}: {sentinel} exists", flush=True)
        return

    structure_paths = sorted(design_dir.glob("state0_sample*.pdb"))
    if len(structure_paths) != 5 or not (design_dir / "state0.pkl").exists():
        raise RuntimeError(f"incomplete parent {parent_id}: found {len(structure_paths)} PDB samples")
    parent_sequence = _final_parent_sequence(design_dir / "optimization_trace.tsv")
    with (design_dir / "state0.pkl").open("rb") as handle:
        output = pickle.load(handle)

    valid = []
    provenance = {}
    fasta_paths = []
    best = None
    for temperature, seed in ((0.1, 0), (0.2, 1), (0.3, 2), (0.5, 3)):
        sequences, fasta_path, best = perform_tied_nampnn_redesign(
            design_dir=str(design_dir),
            state_results=[(output, [None] * 5, 0)],
            polymer_type="rna",
            # Each attempt oversamples; later temperatures/seeds add diversity
            # only when the low-temperature attempt has too many duplicates.
            num_seqs=args.num_seqs * 10,
            motif_indices=None,
            temperature=temperature,
            seed=seed,
        )
        fasta_paths.append(fasta_path)
        for sequence in sequences:
            sequence = sequence.upper()
            if (
                sequence != parent_sequence
                and len(sequence) == len(parent_sequence)
                and set(sequence) <= set("ACGU")
                and sequence not in valid
            ):
                valid.append(sequence)
                provenance[sequence] = {"seed": seed, "temperature": temperature}
        if len(valid) >= args.num_seqs:
            break
    if len(valid) < args.num_seqs:
        raise RuntimeError(f"NA-MPNN returned {len(valid)} valid sequences, expected {args.num_seqs}")

    children = [
        {
            "child_id": f"{parent_id}_nampnn_{index}",
            "parent_id": parent_id,
            "child_index": index,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "sequence_sha256": _digest(sequence),
            "parent_sequence": parent_sequence,
            "parent_sequence_sha256": _digest(parent_sequence),
            "selected_boltz_sample": int(best[0]),
            "seed": provenance[sequence]["seed"],
            "temperature": provenance[sequence]["temperature"],
        }
        for index, sequence in enumerate(valid[: args.num_seqs], 1)
    ]
    sentinel.write_text(
        json.dumps(
            {
                "status": "complete",
                "parent_id": parent_id,
                "group": group,
                "source_fastas": fasta_paths,
                "sampling_protocol": "adaptive_temperature_seed_v1",
                "children": children,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"complete {parent_id}: {len(children)} children", flush=True)


if __name__ == "__main__":
    main()
