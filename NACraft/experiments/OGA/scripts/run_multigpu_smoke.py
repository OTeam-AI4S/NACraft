#!/usr/bin/env python3
"""Run three differentiable FP32 Boltz steps and audit two-GPU gradients."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import yaml

from main import _init_boltz, build_designer
from utils.af3_utils import load_presearch
from utils.mydesign_utils import Annealer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--presearch-json", required=True)
    parser.add_argument("--presearch-outdir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    device_names = tuple(name.strip() for name in args.devices.split(",") if name.strip())
    os.environ["NACRAFT_BOLTZ_DEVICES"] = ",".join(device_names)
    torch.manual_seed(41)
    torch.cuda.manual_seed_all(41)
    visible_devices = torch.cuda.device_count()
    print(
        f"[cuda-audit] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
        f"torch.cuda.device_count={visible_devices}",
        flush=True,
    )
    if visible_devices != len(device_names):
        raise RuntimeError(
            f"expected {len(device_names)} visible CUDA devices, got {visible_devices}"
        )
    for device_index in range(len(device_names)):
        # Establish an allocator context before resetting its statistics.  This
        # PyTorch build rejects reset_peak_memory_stats on a pristine device.
        torch.empty(1, device=f"cuda:{device_index}")
        torch.cuda.reset_peak_memory_stats(device_index)

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    config.setdefault("af3", {})["presearch_lookup"] = load_presearch(
        args.presearch_json, out_dir=args.presearch_outdir
    )

    model = _init_boltz(recycles=0)
    designer, _, _ = build_designer(config)
    initial_sequence = designer.get_seq()
    rows = []
    for step, opt in enumerate(Annealer(hard=0, e_hard=0, iters=args.steps, lr=0.2)):
        captured = []
        designer.logits.requires_grad_(True)
        hook = designer.logits.register_hook(lambda grad: captured.append(grad.detach().clone()))
        designer.do_iter(model, opt, pre_run=True, stage="smoke", stage_step=step)
        hook.remove()
        if len(captured) != 1:
            raise RuntimeError(f"step {step}: expected one logits gradient, got {len(captured)}")
        gradient = captured[0]
        grad_norm = float(gradient.norm().item())
        loss = float(designer.last_total_loss)
        if not math.isfinite(loss) or not torch.isfinite(gradient).all() or grad_norm <= 0:
            raise RuntimeError(f"step {step}: invalid loss/gradient loss={loss}, norm={grad_norm}")
        if not torch.isfinite(designer.logits).all():
            raise RuntimeError(f"step {step}: non-finite logits")
        sequence = designer.get_seq()
        if set(sequence) - set("ACGU"):
            raise RuntimeError(f"step {step}: invalid RNA sequence {sequence!r}")
        rows.append({"step": step, "loss": loss, "gradient_norm": grad_norm, "sequence": sequence})

    report = {
        "protocol": f"oga504x2_fp32_{len(device_names)}gpu_{args.steps}step_smoke_v1",
        "config": str(Path(args.config).resolve()),
        "initial_sequence": initial_sequence,
        "steps": rows,
        "peak_memory_gib": {
            f"cuda:{index}": torch.cuda.max_memory_allocated(index) / 1024**3
            for index in range(len(device_names))
        },
        "dtype": "float32",
        "model_parallel_map": model._nacraft_model_parallel_map,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
