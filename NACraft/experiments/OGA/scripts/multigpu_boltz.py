"""Experiment-side FP32 model parallelism for differentiable Boltz design.

This module deliberately does not modify the vendored Boltz implementation.
It places contiguous trunk layer ranges on two CUDA devices and installs
forward pre-hooks that move each layer's inputs to the layer's device.  The
final MSA and Pairformer layers stay on cuda:0 so the unchanged Boltz residual
and distogram code receive tensors on the primary device.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _move_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value if value.device == device else value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, Mapping):
        return type(value)((key, _move_tree(item, device)) for key, item in value.items())
    return value


def _input_mover(device: torch.device):
    def move_inputs(_module, args, kwargs):
        return _move_tree(args, device), _move_tree(kwargs, device)

    return move_inputs


def _place_layers(layers, device_names: list[str], label: str):
    if len(layers) != len(device_names):
        raise ValueError(f"{label}: {len(layers)} layers but {len(device_names)} assignments")
    handles = []
    for index, (layer, device_name) in enumerate(zip(layers, device_names, strict=True)):
        device = torch.device(device_name)
        layer.to(device)
        handles.append(
            layer.register_forward_pre_hook(_input_mover(device), with_kwargs=True)
        )
        print(f"[boltz-model-parallel] {label}[{index}] -> {device}", flush=True)
    return handles


def dispatch_oga_boltz(model, devices: tuple[str, ...] = ("cuda:0", "cuda:1")):
    """Dispatch the unchanged FP32 Boltz trunk across two to five GPUs."""
    if len(devices) not in (2, 3, 4, 5) or len(set(devices)) != len(devices):
        raise ValueError(f"expected two to five distinct CUDA devices, got {devices!r}")
    if not all(torch.device(name).type == "cuda" for name in devices):
        raise ValueError(f"only CUDA devices are supported, got {devices!r}")
    required = max(torch.device(name).index or 0 for name in devices) + 1
    if torch.cuda.device_count() < required:
        raise RuntimeError(
            f"requested {devices!r}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )

    msa_layers = model.msa_module.layers
    pair_layers = model.pairformer_module.layers
    if len(msa_layers) != 4 or len(pair_layers) != 48:
        raise RuntimeError(
            f"unexpected Boltz trunk layout: MSA={len(msa_layers)}, Pairformer={len(pair_layers)}"
        )

    primary, secondary = devices[:2]
    # Keep the final layer of each stack on the primary device.  The unchanged
    # Boltz caller immediately combines/consumes those outputs on cuda:0.
    if len(devices) == 2:
        msa_map = [primary, secondary, secondary, primary]
        pair_map = [primary] * 20 + [secondary] * 27 + [primary]
    elif len(devices) == 3:
        tertiary = devices[2]
        # GPU0 owns the large input embeddings, residuals, and distogram head.
        # Keep only the mandatory final stack layers there; distribute the
        # movable trunk almost evenly across the two secondary devices.
        msa_map = [secondary, secondary, tertiary, primary]
        pair_map = [secondary] * 24 + [tertiary] * 23 + [primary]
    elif len(devices) == 4:
        tertiary, quaternary = devices[2:]
        msa_map = [secondary, tertiary, quaternary, primary]
        pair_map = (
            [secondary] * 16
            + [tertiary] * 15
            + [quaternary] * 15
            + [primary] * 2
        )
    else:
        tertiary, quaternary, quinary = devices[2:]
        msa_map = [secondary, tertiary, quaternary, primary]
        pair_map = (
            [secondary] * 12
            + [tertiary] * 12
            + [quaternary] * 12
            + [quinary] * 11
            + [primary]
        )

    handles = []
    handles.extend(_place_layers(msa_layers, msa_map, "msa"))
    handles.extend(_place_layers(pair_layers, pair_map, "pairformer"))
    model._nacraft_model_parallel_handles = handles
    model._nacraft_model_parallel_map = {"msa": msa_map, "pairformer": pair_map}
    print(
        "[boltz-model-parallel] FP32 dispatch complete: "
        f"MSA {[msa_map.count(name) for name in devices]}, "
        f"Pairformer {[pair_map.count(name) for name in devices]}",
        flush=True,
    )
    return model
