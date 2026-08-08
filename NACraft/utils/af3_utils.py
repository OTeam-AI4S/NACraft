"""AlphaFold 3 container client for NACraft.

Wraps an AF3 Apptainer container:
  build_input_json  - NACraft state → AF3 task JSON
  run_af3           - invoke official run_alphafold.py or a legacy launch.py
  parse_outputs     - CIF + summary_confidence → NACraft schema

Used by ``MultistateDesigner.get_final_structs_af3`` for final structure
prediction. Sequence optimization always uses Boltz-1 distogram gradients.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

# ---------- defaults ----------

DEFAULTS = {
    "sif_path":       os.environ.get("AF3_SIF_PATH", ""),
    "sandbox_dir":    os.environ.get("AF3_SANDBOX_DIR", ""),
    "af3_code_dir":   os.environ.get("AF3_CODE_DIR", ""),
    "entrypoint":     os.environ.get("AF3_ENTRYPOINT", ""),
    "host_model_dir": os.environ.get("AF3_MODEL_DIR", ""),
    "host_db_dir":    os.environ.get("AF3_DB_DIR", ""),
    "apptainer_bin":  os.environ.get("APPTAINER_BIN", "apptainer"),
    "container_model_dir": "/root/models",
    "container_db_dir":    "/root/public_databases",
    "container_code_dir":  "/app/alphafold",
    "num_samples":    5,
    "run_data_pipeline": False,
    "timeout_sec":    1800,
    "cleanup":        False,
}


def _resolve_cfg(user_cfg: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    if user_cfg:
        cfg.update({k: v for k, v in user_cfg.items() if v is not None})

    # Prefer an executable on PATH while still accepting an explicit path.
    apptainer_bin_dir = os.path.dirname(cfg["apptainer_bin"])
    if apptainer_bin_dir:
        os.environ["PATH"] = apptainer_bin_dir + os.pathsep + os.environ.get("PATH", "")

    from shutil import which
    resolved = which(os.path.basename(cfg["apptainer_bin"]))
    if resolved:
        cfg["apptainer_bin"] = resolved
        return cfg

    if not os.path.exists(cfg["apptainer_bin"]):
        for executable in ("apptainer", "singularity"):
            resolved = which(executable)
            if resolved:
                cfg["apptainer_bin"] = resolved
                break
    return cfg


# ---------- input JSON builder ----------

_POLY_KEY = {"protein": "protein", "rna": "rna", "dna": "dna"}


def _single_seq_a3m(seq: str, description: str = "query") -> str:
    """A minimal A3M with only the query sequence — satisfies AF3's MSA
    validation without invoking the data pipeline.

    AF3's `validate_fold_input` only checks that `unpaired_msa is None`; a
    single-record MSA parses cleanly and yields no homologous diversity, which
    is the right semantic for de novo design where the protein/RNA target is
    given and we do not want external MSA biasing the structure.
    """
    return f">{description}\n{seq}\n"


def load_presearch(index_json_path: str, out_dir: str | None = None) -> dict:
    """Load af3_presearch/out/index.json and rewrite all paths to `out_dir`.

    The presearch cache stores absolute paths in index.json keyed on whatever
    machine produced it. On a different cluster (different NFS mount), those
    paths are stale. We rewrite all paths relative to the cache's out/ root,
    then re-anchor them under `out_dir` (defaults to the index.json's own
    parent dir).

    Returns: dict mapping full protein sequence → {
        unpairedMsaPath, pairedMsaPath, templates: [{mmcifPath, queryIndices,
        templateIndices}, ...]
    } with paths rewritten.
    """
    with open(index_json_path) as f:
        raw = json.load(f)
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(index_json_path))
    out_dir = os.path.abspath(out_dir)

    # Derive the original cache root from the first entry so relpath works.
    sample = next(iter(raw.values()))
    orig_root = os.path.dirname(os.path.dirname(sample["unpairedMsaPath"]))

    def rewrite(p: str) -> str:
        return os.path.join(out_dir, os.path.relpath(p, orig_root))

    lookup: dict[str, dict] = {}
    for seq, entry in raw.items():
        lookup[seq] = {
            "unpairedMsaPath": rewrite(entry["unpairedMsaPath"]),
            "pairedMsaPath":   rewrite(entry["pairedMsaPath"]),
            "templates": [
                {
                    "mmcifPath":       rewrite(t["mmcifPath"]),
                    "queryIndices":    list(t["queryIndices"]),
                    "templateIndices": list(t["templateIndices"]),
                }
                for t in entry.get("templates", [])
            ],
        }
    return lookup


def _protein_chain(seq: str, chain_id: str, presearch: dict | None = None) -> dict:
    """Protein chain with MSA + templates.

    If `presearch` is None: single-sequence MSA, no templates (de novo default).
    If `presearch` is a dict (from load_presearch): real unpaired/paired MSA +
      up to 4 templates, all referenced by path so we can keep
      `run_data_pipeline=False`.

    AF3's folding_input.py accepts BOTH `unpairedMsa`/`pairedMsa` (inline
    strings) and `unpairedMsaPath`/`pairedMsaPath` (filesystem paths), and
    `mmcif`/`mmcifPath` for templates. We use paths because the presearch
    cache already exists as files and inline strings would bloat the JSON.
    """
    if presearch is None:
        return {
            "id": [chain_id],
            "sequence": seq,
            "unpairedMsa": _single_seq_a3m(seq),
            "pairedMsa": _single_seq_a3m(seq),
            "templates": [],
        }
    return {
        "id": [chain_id],
        "sequence": seq,
        "unpairedMsaPath": presearch["unpairedMsaPath"],
        "pairedMsaPath":   presearch["pairedMsaPath"],
        "templates": [
            {
                "mmcifPath":       t["mmcifPath"],
                "queryIndices":    t["queryIndices"],
                "templateIndices": t["templateIndices"],
            }
            for t in presearch.get("templates", [])
        ],
    }


def _rna_chain(seq: str, chain_id: str) -> dict:
    """RNA chain with single-sequence MSA — passes MSA validation."""
    return {
        "id": [chain_id],
        "sequence": seq,
        "unpairedMsa": _single_seq_a3m(seq),
    }


def _dna_chain(seq: str, chain_id: str) -> dict:
    """DNA chain — AF3 validation does not require MSA for DNA."""
    return {"id": [chain_id], "sequence": seq}


def build_input_json(
    name: str,
    seq: str,
    ligands: list[tuple[str, str]] | None,
    polymer_type: str,
    seeds: list[int] | None = None,
    presearch_lookup: dict | None = None,
) -> dict:
    """Build a single AF3 task dict from a NACraft state.

    ligands: list of (value, mol_type) tuples - same shape as designer.ligands[state].
    mol_type ∈ {'protein','rna','dna','ligand','ccd'}.
    presearch_lookup: optional dict[full_protein_seq → presearch entry] from
      load_presearch(). When a protein chain's sequence hits, the chain is
      populated with real MSA + templates from the cache; otherwise the chain
      falls back to single-sequence MSA. RNA/DNA chains are never affected.

    All polymer chains (designed + ligand) pass AF3 featurisation validation
    whether or not presearch is supplied, so `run_data_pipeline=False` remains
    valid in both modes.
    """
    if polymer_type not in _POLY_KEY:
        raise ValueError(f"Unsupported polymer_type for AF3: {polymer_type!r}")

    def _hit(s: str) -> dict | None:
        return presearch_lookup.get(s) if presearch_lookup else None

    sequences: list[dict] = []
    # Designed polymer → chain A
    if polymer_type == "protein":
        sequences.append({"protein": _protein_chain(seq, "A", _hit(seq))})
    elif polymer_type == "rna":
        sequences.append({"rna": _rna_chain(seq, "A")})
    else:  # dna
        sequences.append({"dna": _dna_chain(seq, "A")})

    # Ligands → chains B, C, D...
    chain_letters = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_idx = 0
    for value, mol_type in (ligands or []):
        if chain_idx >= len(chain_letters):
            raise ValueError("Too many ligands - AF3 supports up to 25 chains")
        ch = chain_letters[chain_idx]
        if mol_type == "protein":
            sequences.append({"protein": _protein_chain(value, ch, _hit(value))})
            chain_idx += 1
        elif mol_type == "rna":
            sequences.append({"rna": _rna_chain(value, ch)})
            chain_idx += 1
        elif mol_type == "dna":
            sequences.append({"dna": _dna_chain(value, ch)})
            chain_idx += 1
        elif mol_type == "ligand":
            sequences.append({"ligand": {"id": [ch], "smiles": value}})
            chain_idx += 1
        elif mol_type == "ccd":
            sequences.append({"ligand": {"id": [ch], "ccd": value}})
            chain_idx += 1
        else:
            raise ValueError(f"Unsupported mol_type for AF3: {mol_type!r}")

    task = {
        "name": name,
        "modelSeeds": seeds or [1, 2, 3, 4, 5],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 1,
    }
    return task


# ---------- subprocess runner ----------

def run_af3(
    input_json_path: str,
    output_dir: str,
    cfg: dict | None = None,
    exp_name: str = "nacraft",
    gpus: list[int] | None = None,
    num_samples: int | None = None,
) -> subprocess.CompletedProcess:
    """Invoke AF3 via apptainer. Returns the CompletedProcess (raises on non-zero exit)."""
    cfg = _resolve_cfg(cfg)
    required = {
        "AF3_CODE_DIR": cfg.get("af3_code_dir"),
        "AF3_MODEL_DIR": cfg.get("host_model_dir"),
    }
    if cfg.get("run_data_pipeline"):
        required["AF3_DB_DIR"] = cfg.get("host_db_dir")
    missing = [name for name, path in required.items() if not path or not os.path.exists(str(path))]
    if missing:
        raise FileNotFoundError(
            "Missing AF3 resources: " + ", ".join(missing) +
            ". Set the corresponding environment variables or af3 YAML fields."
        )
    if not any(
        path and os.path.exists(str(path))
        for path in (cfg.get("sandbox_dir"), cfg.get("sif_path"))
    ):
        raise FileNotFoundError(
            "Set AF3_SANDBOX_DIR or AF3_SIF_PATH to an existing AF3 container."
        )
    os.makedirs(output_dir, exist_ok=True)
    input_json_path = os.path.abspath(input_json_path)
    output_dir = os.path.abspath(output_dir)

    configured_entrypoint = str(cfg.get("entrypoint") or "").strip()
    if configured_entrypoint:
        host_entrypoint = Path(configured_entrypoint)
        if not host_entrypoint.is_absolute():
            host_entrypoint = Path(str(cfg["af3_code_dir"])) / host_entrypoint
    else:
        official = Path(str(cfg["af3_code_dir"])) / "run_alphafold.py"
        legacy = Path(str(cfg["af3_code_dir"])) / "launch.py"
        host_entrypoint = official if official.is_file() else legacy
    if not host_entrypoint.is_file():
        raise FileNotFoundError(
            "AF3 entry point not found. Set AF3_CODE_DIR to an official AF3 "
            "checkout containing run_alphafold.py, or set AF3_ENTRYPOINT. "
            f"Checked: {host_entrypoint}"
        )
    entrypoint_name = host_entrypoint.name
    official_runner = entrypoint_name == "run_alphafold.py"

    runner_input_path = input_json_path
    if official_runner:
        payload = json.loads(Path(input_json_path).read_text())
        if isinstance(payload, list):
            if len(payload) != 1:
                raise ValueError(
                    "Official AF3 accepts one task per JSON file; NACraft's "
                    f"single-run adapter received {len(payload)} tasks."
                )
            payload = payload[0]
        official_input = Path(output_dir) / f"{Path(input_json_path).stem}_official.json"
        official_input.write_text(json.dumps(payload, indent=2) + "\n")
        runner_input_path = str(official_input.resolve())

    ns = num_samples if num_samples is not None else cfg["num_samples"]
    # AF3 launch.py declares --gpus with nargs='+' so each id must be its own
    # argv element. Passing "0 1 2 3" as a single string fails with
    # `invalid int value: '0 1 2 3'`.
    gpu_list = gpus if gpus is not None else _visible_gpu_ids()
    if not gpu_list:
        gpu_list = [0]

    # launch.py writes tmp split-JSONs and logs under <container_code_dir>/{tmp,log}
    # which is bind-mounted from a read-only source. Overlay writable host dirs.
    writable_tmp = os.path.join(output_dir, "_af3_tmp")
    writable_log = os.path.join(output_dir, "_af3_log")
    os.makedirs(writable_tmp, exist_ok=True)
    os.makedirs(writable_log, exist_ok=True)

    # Presearch cache (MSA + template files referenced by path in input JSON).
    # Identity-bind the configured dir so AF3 can read A3M/CIF files. Skip if
    # not set or already covered by another bind.
    binds = [
        f'{cfg["host_model_dir"]}:{cfg["container_model_dir"]}',
        f'{cfg["af3_code_dir"]}:{cfg["container_code_dir"]}',
        f'{writable_tmp}:{cfg["container_code_dir"]}/tmp',
        f'{writable_log}:{cfg["container_code_dir"]}/log',
        f'{runner_input_path}:{runner_input_path}',
        f'{output_dir}:{output_dir}',
    ]
    if cfg.get("run_data_pipeline") or os.path.exists(str(cfg.get("host_db_dir", ""))):
        binds.insert(1, f'{cfg["host_db_dir"]}:{cfg["container_db_dir"]}')
    presearch_dir = cfg.get("host_presearch_dir")
    if presearch_dir:
        presearch_dir = os.path.abspath(presearch_dir)
        binds.append(f'{presearch_dir}:{presearch_dir}:ro')

    cmd = [
        cfg["apptainer_bin"],
        "exec", "--nv",
        "--env", "NVIDIA_VISIBLE_DEVICES=all",
    ]
    # Prefer pre-extracted sandbox dir over SIF+writable-tmpfs. The tmpfs path
    # extracts the SIF on every invocation and fails with proc-mount denied
    # under SLURM's container nesting. Sandbox dirs are already extracted.
    container_root = cfg.get("sandbox_dir") or cfg["sif_path"]
    if not os.path.isdir(container_root):
        # Sandbox dir missing → fall back to SIF + writable-tmpfs.
        cmd.append("--writable-tmpfs")
        container_root = cfg["sif_path"]
    for b in binds:
        cmd += ["-B", b]
    container_entrypoint = f'{cfg["container_code_dir"]}/{entrypoint_name}'
    if official_runner:
        cmd += [
            container_root,
            "python", container_entrypoint,
            f"--json_path={runner_input_path}",
            f'--model_dir={cfg["container_model_dir"]}',
            f"--output_dir={output_dir}",
            f'--run_data_pipeline={str(cfg["run_data_pipeline"]).lower()}',
            "--run_inference=true",
            f"--num_diffusion_samples={ns}",
        ]
        if cfg.get("run_data_pipeline"):
            cmd.append(f'--db_dir={cfg["container_db_dir"]}')
    else:
        cmd += [
            container_root,
            "python", container_entrypoint,
            "--input_json", input_json_path,
            "--output_dir", output_dir,
            "--run_data_pipeline", str(cfg["run_data_pipeline"]).lower(),
            "--gpus", *[str(g) for g in gpu_list],
            "--exp_name", exp_name,
            "--num_diffusion_samples", str(ns),
        ]

    try:
        # AF3's bundled jax rejects XLA_PYTHON_CLIENT_MEM_FRACTION (the older
        # alias) if XLA_CLIENT_MEM_FRACTION is also present. Pass a clean
        # environment so the container's JAX runtime is deterministic.
        sub_env = {k: v for k, v in os.environ.items()
                   if not k.startswith("XLA_PYTHON_CLIENT_")}
        sub_env.pop("XLA_CLIENT_MEM_FRACTION", None)
        if official_runner and gpu_list:
            sub_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_list)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg["timeout_sec"],
            check=False,
            env=sub_env,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"AF3 subprocess timed out after {cfg['timeout_sec']}s "
            f"(input={input_json_path}). Consider raising af3.timeout_sec."
        ) from e

    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-50:])
        raise RuntimeError(
            f"AF3 subprocess exited {result.returncode}.\n"
            f"cmd: {' '.join(cmd)}\n--- last 50 lines ---\n{tail}"
        )
    return result


def _visible_gpu_ids() -> list[int]:
    """Respect CUDA_VISIBLE_DEVICES if set, else [0]."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not env:
        return [0]
    try:
        return [int(x) for x in env.split(",") if x.strip() != ""]
    except ValueError:
        return [0]


# ---------- output parser ----------

def find_run_dir(output_dir: str, name: str) -> str | None:
    """Locate the AF3 per-job output subdir.

    AF3 typically writes to either:
      <output_dir>/<name>_<timestamp>/
      <output_dir>/<name>/
    Returns the most recently modified matching dir, or None.
    """
    candidates = []
    for entry in Path(output_dir).iterdir():
        if not entry.is_dir():
            continue
        if entry.name == name or entry.name.startswith(f"{name}_"):
            candidates.append(entry)
    if not candidates:
        return None
    # Pick newest by mtime
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def parse_outputs(
    output_dir: str,
    name: str,
    num_samples: int,
) -> tuple[list[Any], dict[str, np.ndarray]]:
    """Parse AF3 output dir.

    Returns (struct_list, metrics_dict).
      struct_list: list of gemmi.Structure, len=num_samples. AF3 only writes the
                   top-ranked model per invocation; we tile that structure across
                   `num_samples` slots so downstream code sees the expected count.
                   (Caller treats the duplicated entries as repeated measurements
                   of the same prediction — semantically honest, since AF3's
                   stochastic sampling is internal.)
      metrics_dict: keys = confidence_score, iptm, ptm, complex_iplddt, complex_plddt,
                    complex_ipde; each np.float32 array of len=num_samples.
                    complex_ipde is NaN-filled (AF3 doesn't expose per-sample PDE
                    in summary form).
    """
    import gemmi

    run_dir = find_run_dir(output_dir, name)
    if run_dir is None:
        raise FileNotFoundError(
            f"No AF3 output subdir matching name={name!r} under {output_dir}"
        )

    # AF3 writes only the top-ranked model. Find the single *_model.cif and load
    # it; we then tile to num_samples at the end.
    cif_path = None
    for cand in Path(run_dir).glob("*_model.cif"):
        cif_path = cand
        break
    if cif_path is None:
        raise FileNotFoundError(f"No *_model.cif under {run_dir}")

    full = gemmi.read_structure(str(cif_path))
    # Take the first model in the CIF (AF3 outputs one model per file).
    if len(full) == 0:
        raise ValueError(f"CIF {cif_path} has no models")
    src_model = full[0]
    base_struct = gemmi.Structure()
    base_struct.name = f"{name}_s0"
    new_model = gemmi.Model("1")
    for chain in src_model:
        new_model.add_chain(chain)
    base_struct.add_model(new_model)
    structs = [base_struct] * num_samples

    # summary_confidences.json — single top-ranked sample.
    # Layout observed on the simplified-AF3 install: scalars for most fields
    # (`ranking_score`, `ptm`, `iptm`, `has_clash`, `fraction_disordered`),
    # plus per-chain lists (`chain_iptm`, `chain_ptm`, ...). Older / standard
    # AF3 writes lists of length N for N ranked samples. We accept both shapes.
    summary_path = None
    for cand in Path(run_dir).glob("*summary_confidence*.json"):
        summary_path = cand
        break
    if summary_path is None:
        raise FileNotFoundError(f"No *summary_confidence*.json under {run_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    per_sample_metrics = _normalize_summary(summary, num_samples)

    # Per-atom pLDDT for interface computation. Single top-ranked file.
    atom_plddts_paths = sorted(Path(run_dir).glob("*confidences*.json"))
    plddt_arr: np.ndarray | None = None
    if atom_plddts_paths:
        with open(atom_plddts_paths[0]) as f:
            conf = json.load(f)
        if "atom_plddts" in conf:
            raw = conf["atom_plddts"]
            # Either flat list (single sample) or nested (multiple samples).
            if raw and isinstance(raw[0], list):
                plddt_arr = np.asarray(raw[0], dtype=np.float32)
            else:
                plddt_arr = np.asarray(raw, dtype=np.float32)

    # Compute complex_plddt + complex_iplddt (same value tiled across samples).
    complex_plddt = np.full(num_samples, np.nan, dtype=np.float32)
    complex_iplddt = np.full(num_samples, np.nan, dtype=np.float32)
    if plddt_arr is not None and len(base_struct) > 0:
        complex_plddt_val = float(np.mean(plddt_arr)) / 100.0  # AF3 0-100 → 0-1
        iplddt_val = _compute_interface_plddt(base_struct, plddt_arr)
        complex_plddt[:] = complex_plddt_val
        if iplddt_val is not None:
            complex_iplddt[:] = iplddt_val

    metrics = {
        "confidence_score": per_sample_metrics.get("ranking_confidence",
                              np.full(num_samples, np.nan, dtype=np.float32)).astype(np.float32),
        "iptm":             per_sample_metrics.get("iptm",
                              np.full(num_samples, np.nan, dtype=np.float32)).astype(np.float32),
        "ptm":              per_sample_metrics.get("ptm",
                              np.full(num_samples, np.nan, dtype=np.float32)).astype(np.float32),
        "complex_iplddt":   complex_iplddt,
        "complex_plddt":    complex_plddt,
        "complex_ipde":     np.full(num_samples, np.nan, dtype=np.float32),
    }
    return structs, metrics


def _normalize_summary(summary: Any, num_samples: int) -> dict[str, np.ndarray]:
    """Coerce various AF3 summary_confidence.json layouts into {key: np.array(len=N)}.

  Handles:
    Form A (standard multi-sample):  {"ranking_confidence": [v0,v1,...], ...}
    Form B (per-sample dicts):       {"sample_0": {"ptm": 0.5, ...}, ...}
    Form C (list of dicts):          [{"ptm": 0.5, ...}, ...]
    Form D (simplified-AF3 scalar):  {"ranking_score": 0.58, "ptm": 0.17, ...}
                                     (top-ranked only — tile to num_samples)

  Aliases `ranking_score` (simplified AF3) → `ranking_confidence` (standard).
  """
    keys = ("ranking_confidence", "ptm", "iptm", "has_clash", "fraction_disordered")
    out: dict[str, np.ndarray] = {}

    def _tile(v: float) -> np.ndarray:
        return np.full(num_samples, float(v), dtype=np.float32)

    if isinstance(summary, dict):
        # ranking_score → ranking_confidence alias
        if "ranking_score" in summary and "ranking_confidence" not in summary:
            summary = dict(summary)
            summary["ranking_confidence"] = summary["ranking_score"]

        # Form A: lists of length N
        for k in keys:
            if k in summary and isinstance(summary[k], list):
                out[k] = np.asarray(summary[k], dtype=np.float32)

        if not out:
            # Form B: per-sample dicts nested under sample-name keys
            sample_vals = list(summary.values())[:num_samples]
            if all(isinstance(v, dict) for v in sample_vals):
                for k in keys:
                    if all(k in v for v in sample_vals):
                        out[k] = np.asarray([v[k] for v in sample_vals], dtype=np.float32)

        if not out:
            # Form D: scalars (top-ranked only). Tile to num_samples.
            for k in keys:
                if k in summary and isinstance(summary[k], (int, float)):
                    out[k] = _tile(summary[k])
    elif isinstance(summary, list):
        # Form C: list of per-sample dicts
        for k in keys:
            if all(isinstance(v, dict) and k in v for v in summary[:num_samples]):
                out[k] = np.asarray([v[k] for v in summary[:num_samples]], dtype=np.float32)
    return out


def _compute_interface_plddt(struct: Any, atom_plddts: np.ndarray, cutoff: float = 5.0) -> float | None:
    """Mean pLDDT of atoms within `cutoff` Å of any atom in another chain.

    struct: gemmi.Structure (single model)
    atom_plddts: 1D array, one value per ATOM/HETATM record in CIF order.
    """
    try:
        model = struct[0]
    except IndexError:
        return None

    # Collect per-chain atom positions and global indices
    chains = []
    global_idx = 0
    for chain in model:
        atoms_pos = []
        idx_range = []
        for res in chain:
            for atom in res:
                atoms_pos.append([atom.pos.x, atom.pos.y, atom.pos.z])
                idx_range.append(global_idx)
                global_idx += 1
        if atoms_pos:
            chains.append((chain.name, np.asarray(atoms_pos), idx_range))

    if len(chains) < 2 or global_idx == 0:
        return None

    # Trim atom_plddts to actual atom count
    plddt = atom_plddts[:global_idx]
    if len(plddt) < global_idx:
        return None

    interface_mask = np.zeros(global_idx, dtype=bool)
    for i, (name_i, pos_i, idx_i) in enumerate(chains):
        for j, (name_j, pos_j, idx_j) in enumerate(chains):
            if j <= i:
                continue
            # pairwise distances between chain i and chain j atoms
            d = np.linalg.norm(pos_i[:, None, :] - pos_j[None, :, :], axis=-1)
            pairs_i, pairs_j = np.where(d < cutoff)
            for pi in pairs_i:
                interface_mask[idx_i[pi]] = True
            for pj in pairs_j:
                interface_mask[idx_j[pj]] = True

    if not interface_mask.any():
        return None
    return float(np.mean(plddt[interface_mask])) / 100.0  # 0-100 → 0-1


# ---------- combined entrypoint ----------

def predict_complex(
    name: str,
    seq: str,
    ligands: list[tuple[str, str]],
    polymer_type: str,
    output_dir: str,
    cfg: dict | None = None,
    exp_name: str | None = None,
    num_samples: int | None = None,
    seeds: list[int] | None = None,
    presearch_lookup: dict | None = None,
) -> tuple[list[Any], dict[str, np.ndarray]]:
    """One-shot: build JSON, run AF3, parse outputs.

    Returns (struct_list, metrics_dict) - see parse_outputs() for shape.
    """
    cfg = _resolve_cfg(cfg)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    task = build_input_json(name, seq, ligands, polymer_type, seeds=seeds,
                            presearch_lookup=presearch_lookup)
    input_json = os.path.join(output_dir, f"{name}_input.json")
    with open(input_json, "w") as f:
        json.dump([task], f, indent=2)  # launch.py expects a list

    ns = num_samples if num_samples is not None else cfg["num_samples"]
    run_af3(
        input_json_path=input_json,
        output_dir=output_dir,
        cfg=cfg,
        exp_name=exp_name or name,
        num_samples=ns,
    )

    structs, metrics = parse_outputs(output_dir, name, ns)

    if cfg.get("cleanup"):
        # Keep only the parsed metrics + CIFs; remove raw subdirs
        for sub in Path(output_dir).iterdir():
            if sub.is_dir() and sub.name.startswith(name):
                shutil.rmtree(sub, ignore_errors=True)
        Path(input_json).unlink(missing_ok=True)

    return structs, metrics
