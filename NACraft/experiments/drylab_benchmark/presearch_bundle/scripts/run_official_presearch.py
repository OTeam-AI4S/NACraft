#!/usr/bin/env python3
"""Run list-form NACraft presearch inputs through official AlphaFold 3."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_task(args: argparse.Namespace, task_path: Path) -> None:
    task_path = task_path.resolve()
    output_dir = args.output_dir.resolve()
    code_dir = args.code_dir.resolve()
    model_dir = args.model_dir.resolve()
    db_dir = args.db_dir.resolve()
    command = [
        args.apptainer,
        "exec",
        "--nv",
        "--writable-tmpfs",
        "-B",
        f"{model_dir}:/root/models",
        "-B",
        f"{db_dir}:/root/public_databases",
        "-B",
        f"{task_path}:{task_path}:ro",
        "-B",
        f"{output_dir}:{output_dir}",
        "-B",
        f"{code_dir}:/app/alphafold:ro",
        str(args.image),
        "python",
        "/app/alphafold/run_alphafold.py",
        f"--json_path={task_path}",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        f"--output_dir={output_dir}",
        "--run_data_pipeline=true",
        "--run_inference=false",
    ]
    print("[presearch]", task_path.name, flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--apptainer", default="apptainer")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text())
    tasks = payload if isinstance(payload, list) else [payload]
    if not tasks:
        raise SystemExit("Presearch input contains no AF3 tasks")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.task_dir.mkdir(parents=True, exist_ok=True)
    task_paths = []
    for index, task in enumerate(tasks, 1):
        name = str(task.get("name") or f"target_{index}")
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
        task_path = args.task_dir / f"{index:04d}_{safe_name}.json"
        task_path.write_text(json.dumps(task, indent=2) + "\n")
        task_paths.append(task_path)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda path: run_task(args, path), task_paths))


if __name__ == "__main__":
    main()
