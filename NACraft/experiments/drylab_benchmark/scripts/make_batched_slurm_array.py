#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path


def count_commands(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))


def render_batched_slurm_array(
    job_name: str,
    commands_file: str,
    num_commands: int,
    commands_per_task: int,
    partition: str,
    gres: str,
    time_limit: str,
    cpus_per_task: int,
    mem: str,
    log_dir: str,
    concurrency: int = 0,
    workdir: str = "",
    exclude: str = "",
) -> str:
    num_tasks = math.ceil(num_commands / commands_per_task)
    array = f"1-{num_tasks}"
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""
    exclude_line = f"#SBATCH --exclude={exclude}\n" if exclude else ""
    cd_line = f"cd {workdir}" if workdir else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{gres_line}#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --array={array}
{exclude_line}#SBATCH --output={log_dir}/%x_%A_%a.out
#SBATCH --error={log_dir}/%x_%A_%a.err

set -uo pipefail

{cd_line}
COMMANDS_FILE={commands_file}
COMMANDS_PER_TASK={commands_per_task}
START=$(( (SLURM_ARRAY_TASK_ID - 1) * COMMANDS_PER_TASK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * COMMANDS_PER_TASK ))

echo "[drylab-batch] task=${{SLURM_ARRAY_TASK_ID}} range=${{START}}-${{END}} file=${{COMMANDS_FILE}}"
failures=0
line_no=${{START}}
while IFS= read -r COMMAND; do
    if [[ -z "${{COMMAND}}" || "${{COMMAND}}" =~ ^[[:space:]]*# ]]; then
        line_no=$(( line_no + 1 ))
        continue
    fi
    echo "[drylab-batch] line=${{line_no}} ${{COMMAND}}"
    bash -lc "${{COMMAND}}"
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "[drylab-batch] FAILED line=${{line_no}} status=${{status}}" >&2
        failures=$(( failures + 1 ))
    fi
    line_no=$(( line_no + 1 ))
done < <(sed -n "${{START}},${{END}}p" "${{COMMANDS_FILE}}")

if [[ $failures -ne 0 ]]; then
    echo "[drylab-batch] completed with ${{failures}} failed commands" >&2
    exit 1
fi
echo "[drylab-batch] completed successfully"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a SLURM array where each task executes multiple command-list rows.")
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--commands-per-task", type=int, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--exclude", default="")
    args = parser.parse_args()

    commands_file = Path(args.commands_file)
    script = render_batched_slurm_array(
        job_name=args.job_name,
        commands_file=str(commands_file),
        num_commands=count_commands(commands_file),
        commands_per_task=args.commands_per_task,
        partition=args.partition,
        gres=args.gres,
        time_limit=args.time,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        log_dir=args.log_dir,
        concurrency=args.concurrency,
        workdir=args.workdir,
        exclude=args.exclude,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script)


if __name__ == "__main__":
    main()
