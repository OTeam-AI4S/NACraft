#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from make_slurm_array import render_slurm_array
except ModuleNotFoundError:
    from .make_slurm_array import render_slurm_array


def chunk_lines(lines: list[str], chunk_size: int) -> list[list[str]]:
    return [lines[idx : idx + chunk_size] for idx in range(0, len(lines), chunk_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a command file and render one SLURM array per chunk.")
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--script-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--workdir", default="")
    args = parser.parse_args()

    lines = [
        line
        for line in Path(args.commands_file).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    chunk_dir = Path(args.chunk_dir)
    script_dir = Path(args.script_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    for idx, chunk in enumerate(chunk_lines(lines, args.chunk_size), start=1):
        command_file = chunk_dir / f"{Path(args.commands_file).stem}.part{idx:02d}.sh"
        command_file.write_text("\n".join(chunk) + "\n")
        script = render_slurm_array(
            job_name=f"{args.job_name}_{idx:02d}",
            commands_file=str(command_file),
            num_commands=len(chunk),
            partition=args.partition,
            gres=args.gres,
            time_limit=args.time,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            log_dir=args.log_dir,
            concurrency=args.concurrency,
            workdir=args.workdir,
        )
        script_path = script_dir / f"submit_{args.job_name}_{idx:02d}.sh"
        script_path.write_text(script)
        print(script_path)


if __name__ == "__main__":
    main()
