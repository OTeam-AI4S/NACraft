#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


def read_init_seq(config: Path) -> str:
    for line in config.read_text().splitlines():
        if line.startswith("init_seq:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_nacraft_command(
    config_path: Path,
    out_root: str,
    candidates: int,
    nacraft_entry: str,
    python_executable: str,
    num_workers: int = 1,
    worker_id: int = 0,
    init_seq: str = "",
    presearch_json: str = "",
    presearch_outdir: str = "",
    ligandmpnn_seqs: int = 2,
) -> str:
    out_dir = f"{out_root}/{config_path.parent.name}/{config_path.stem}"
    parts = [
        python_executable,
        nacraft_entry,
        "--config",
        str(config_path),
        "--num_designs",
        str(candidates),
        "--num_workers",
        str(num_workers),
        "--worker_id",
        str(worker_id),
        "-o",
        out_dir,
        "--ligandmpnn_seqs",
        str(ligandmpnn_seqs),
        "--skip_existing",
    ]
    if init_seq:
        parts.extend(["--init-seq", init_seq])
    if presearch_json:
        parts.extend(["--presearch-json", presearch_json])
    if presearch_outdir:
        parts.extend(["--presearch-outdir", presearch_outdir])
    return " ".join(shlex.quote(part) for part in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a shell command list for NACraft dry-lab design runs.")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nacraft-entry", default="NACraft/main.py")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--out-root", default="outputs/drylab_benchmark/candidates/nacraft")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--num-workers-per-config", type=int, default=4)
    parser.add_argument("--ligandmpnn-seqs", type=int, default=2)
    parser.add_argument("--presearch-json", default="")
    parser.add_argument("--presearch-outdir", default="")
    parser.add_argument("--limit-configs", type=int, default=0)
    args = parser.parse_args()

    configs = sorted(Path(args.config_dir).rglob("*.yaml"))
    if args.limit_configs:
        configs = configs[: args.limit_configs]
    lines = []
    for config in configs:
        for worker_id in range(args.num_workers_per_config):
            lines.append(
                build_nacraft_command(
                    config_path=config,
                    out_root=args.out_root,
                    candidates=args.candidates,
                    nacraft_entry=args.nacraft_entry,
                    python_executable=args.python_executable,
                    num_workers=args.num_workers_per_config,
                    worker_id=worker_id,
                    init_seq=read_init_seq(config),
                    presearch_json=args.presearch_json,
                    presearch_outdir=args.presearch_outdir,
                    ligandmpnn_seqs=args.ligandmpnn_seqs,
                )
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + ("\n" if lines else ""))


if __name__ == "__main__":
    main()
