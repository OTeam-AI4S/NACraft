#!/usr/bin/env python3
"""Refold corrected OGA504x2+RNA parents with Boltz and save real structures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch


_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    if isinstance(kwargs.get("map_location"), str) and "cuda" in kwargs["map_location"]:
        kwargs["map_location"] = "cpu"
    return _torch_load(*args, **kwargs)


torch.load = _safe_torch_load

NACRAFT_DIR = Path(
    os.environ.get("NACRAFT_DIR", Path(__file__).resolve().parents[3])
)
sys.path[:0] = [str(NACRAFT_DIR), str(NACRAFT_DIR / "boltz/src")]
os.chdir(NACRAFT_DIR)

from switchcraft import _init_boltz, build_designer  # noqa: E402
from utils.af3_utils import load_presearch  # noqa: E402
from utils.mydesign_utils import save_structs  # noqa: E402


def values(output: dict, key: str) -> list[float | None]:
    value = output.get(key)
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        value = [value]
    result = []
    for item in value:
        number = float(item)
        result.append(None if number != number else number)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--presearch-json", required=True)
    parser.add_argument("--presearch-outdir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    target = json.loads(Path(args.target_manifest).read_text())
    protein = target["sequence"]
    if len(protein) != 504 or target["protein_copy_count"] != 2:
        raise SystemExit("Refusing non-OGA504x2 target")
    presearch = load_presearch(args.presearch_json, out_dir=args.presearch_outdir)
    if protein not in presearch:
        raise SystemExit("Exact 504-aa target missing from corrected presearch index")

    with Path(args.manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 600:
        raise SystemExit(f"Expected 600 parents, got {len(rows)}")
    mine = rows[args.worker_id :: args.num_workers]

    model = _init_boltz()
    designers = {}
    for length in sorted({int(row["sequence_length"]) for row in mine}):
        config = {
            "polymer_type": "rna",
            "predictor": "boltz",
            "num_states": 1,
            "length": length,
            "motifs": [],
            "states": [[f"protein:{protein}", f"protein:{protein}"]],
            "losses": [{"type": "LigandContactLoss", "state": 0}],
            "af3": {"presearch_lookup": presearch},
        }
        designers[length], _, _ = build_designer(config)

    out_root = Path(args.output_root)
    for offset, row in enumerate(mine, 1):
        parent_id = row["parent_id"]
        out_dir = out_root / parent_id
        sentinel = out_dir / "complete.json"
        if sentinel.exists():
            print(f"[{offset}/{len(mine)}] skip {parent_id}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            seq = row["sequence"]
            structs = designers[len(seq)].get_final_structs(
                model, samples=args.samples, set_seq=seq
            )
            save_structs(structs, str(out_dir))
            states = []
            for output, struct_list, state_idx in structs:
                if len(struct_list) != args.samples:
                    raise RuntimeError(
                        f"Expected {args.samples} structures, got {len(struct_list)}"
                    )
                states.append(
                    {
                        "state": state_idx,
                        "num_structures": len(struct_list),
                        **{
                            key: values(output, key)
                            for key in (
                                "iptm",
                                "ptm",
                                "complex_plddt",
                                "complex_iplddt",
                                "confidence_score",
                            )
                        },
                    }
                )
            record = {
                "status": "complete",
                "parent_id": parent_id,
                "sequence": seq,
                "target_system": "OGA504x2+RNA",
                "protein_sequence_sha256": target["sequence_sha256"],
                "protein_copies": 2,
                "samples": args.samples,
                "states": states,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            sentinel.write_text(json.dumps(record, indent=2) + "\n")
            print(f"[{offset}/{len(mine)}] complete {parent_id}", flush=True)
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
            print(f"[{offset}/{len(mine)}] FAILED {parent_id}: {error}", flush=True)


if __name__ == "__main__":
    main()
