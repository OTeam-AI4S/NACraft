#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def render_slurm_array(
    job_name: str,
    commands_file: str,
    num_commands: int,
    partition: str,
    gres: str,
    time_limit: str,
    cpus_per_task: int,
    mem: str,
    log_dir: str,
    concurrency: int = 0,
    workdir: str = "",
) -> str:
    array = f"1-{num_commands}"
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{gres_line}#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --array={array}
#SBATCH --output={log_dir}/%x_%A_%a.out
#SBATCH --error={log_dir}/%x_%A_%a.err

set -euo pipefail

{f"cd {workdir}" if workdir else ""}
COMMAND=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {commands_file})
echo "[drylab] ${{COMMAND}}"
eval "${{COMMAND}}"
"""


def count_commands(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a SLURM array script from a command list.")
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument("--workdir", default="")
    args = parser.parse_args()

    commands_file = Path(args.commands_file)
    script = render_slurm_array(
        job_name=args.job_name,
        commands_file=str(commands_file),
        num_commands=count_commands(commands_file),
        partition=args.partition,
        gres=args.gres,
        time_limit=args.time,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        log_dir=args.log_dir,
        concurrency=args.concurrency,
        workdir=args.workdir,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script)


if __name__ == "__main__":
    main()
