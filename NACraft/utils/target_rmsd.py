"""Target-structure RMSD utilities for NACraft validation outputs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa

try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:  # Biopython >= 1.82 removed this public helper.
    from Bio.Data.PDBData import protein_letters_3to1_extended

    def three_to_one(residue_name: str) -> str:
        return protein_letters_3to1_extended[residue_name.upper()]


@dataclass(frozen=True)
class TargetRMSDSpec:
    structure_path: str
    chains: tuple[str, ...]
    residue_ranges: tuple[tuple[int | None, int | None], ...] = ()


@dataclass(frozen=True)
class TargetRMSDResult:
    rmsd: float
    matched_ca_atoms: int
    mean_sequence_identity: float
    matched_chain_pairs: str


def _parse_chain_list(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    text = str(value or "").strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    return tuple(str(item) for item in parsed)


def target_spec_from_metadata(metadata: Mapping[str, object]) -> TargetRMSDSpec | None:
    """Build a target specification from a NACraft benchmark manifest row."""
    structure_path = str(metadata.get("structure_path") or "")
    if not structure_path:
        return None

    state_target_id = str(metadata.get("state_target_id") or "")
    match = re.match(r"^[^_]+_([^_]+)_(\d+)_(\d+)$", state_target_id)
    if match:
        return TargetRMSDSpec(
            structure_path=structure_path,
            chains=(match.group(1),),
            residue_ranges=((int(match.group(2)), int(match.group(3))),),
        )

    chains = _parse_chain_list(metadata.get("protein_chains"))
    if not chains:
        return None
    return TargetRMSDSpec(
        structure_path=structure_path,
        chains=chains,
        residue_ranges=tuple((None, None) for _ in chains),
    )


def _load_structure(path: str | Path):
    path = Path(path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def _protein_chain_records(chain, start: int | None = None, end: int | None = None):
    records = []
    for residue in chain:
        if not is_aa(residue, standard=False) or "CA" not in residue:
            continue
        residue_number = int(residue.id[1])
        if start is not None and residue_number < start:
            continue
        if end is not None and residue_number > end:
            continue
        try:
            letter = three_to_one(residue.get_resname().strip().upper())
        except KeyError:
            letter = "X"
        records.append((letter, np.asarray(residue["CA"].coord, dtype=float)))
    return records


@lru_cache(maxsize=None)
def _native_chain_records(spec: TargetRMSDSpec):
    native_model = _load_structure(spec.structure_path)[0]
    ranges = spec.residue_ranges or tuple((None, None) for _ in spec.chains)
    native_chains = []
    for chain_id, bounds in zip(spec.chains, ranges):
        if chain_id not in native_model:
            raise ValueError(f"input target chain {chain_id!r} is absent from {spec.structure_path}")
        native_chains.append(
            (chain_id, _protein_chain_records(native_model[chain_id], bounds[0], bounds[1]))
        )
    return native_chains


@lru_cache(maxsize=None)
def _aligned_indices(native_sequence: str, predicted_sequence: str):
    if len(native_sequence) < 3 or len(predicted_sequence) < 3:
        raise ValueError("target chain has fewer than three resolved C-alpha atoms")

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(native_sequence, predicted_sequence)[0]
    index_pairs = []
    identities = 0
    for native_block, predicted_block in zip(alignment.aligned[0], alignment.aligned[1]):
        native_start, native_end = map(int, native_block)
        predicted_start, predicted_end = map(int, predicted_block)
        block_length = min(native_end - native_start, predicted_end - predicted_start)
        for offset in range(block_length):
            native_index = native_start + offset
            predicted_index = predicted_start + offset
            index_pairs.append((native_index, predicted_index))
            identities += int(native_sequence[native_index] == predicted_sequence[predicted_index])
    if len(index_pairs) < 3:
        raise ValueError("sequence alignment yielded fewer than three matched C-alpha atoms")
    return tuple(index_pairs), identities / len(index_pairs)


def _aligned_ca_pairs(native_records, predicted_records):
    native_sequence = "".join(item[0] for item in native_records)
    predicted_sequence = "".join(item[0] for item in predicted_records)
    index_pairs, identity = _aligned_indices(native_sequence, predicted_sequence)
    native_coords = [native_records[i][1] for i, _ in index_pairs]
    predicted_coords = [predicted_records[j][1] for _, j in index_pairs]
    return native_coords, predicted_coords, identity


def _kabsch_rmsd(native_coords, predicted_coords) -> float:
    native = np.asarray(native_coords, dtype=float)
    predicted = np.asarray(predicted_coords, dtype=float)
    native_centroid = native.mean(axis=0)
    predicted_centroid = predicted.mean(axis=0)
    native_zero = native - native_centroid
    predicted_zero = predicted - predicted_centroid
    covariance = predicted_zero.T @ native_zero
    v, _singular_values, wt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(v @ wt)
    rotation = v @ correction @ wt
    aligned = predicted_zero @ rotation + native_centroid
    differences = aligned - native
    return float(np.sqrt(np.mean(np.sum(differences * differences, axis=1))))


def calculate_target_rmsd(refold_path: str | Path, spec: TargetRMSDSpec) -> TargetRMSDResult:
    """Compare the refolded protein target with its input target coordinates."""
    refold_model = _load_structure(refold_path)[0]

    predicted_chains = []
    for chain in refold_model:
        records = _protein_chain_records(chain)
        if len(records) >= 3:
            predicted_chains.append((chain.id, records))
    if not predicted_chains:
        raise ValueError("refold structure contains no protein chain with at least three C-alpha atoms")

    all_native = []
    all_predicted = []
    identities = []
    chain_pairs = []
    used_predicted = set()
    for native_chain_id, native_records in _native_chain_records(spec):
        matches = []
        for predicted_chain_id, predicted_records in predicted_chains:
            if predicted_chain_id in used_predicted:
                continue
            native_coords, predicted_coords, identity = _aligned_ca_pairs(
                native_records, predicted_records
            )
            matches.append(
                (len(native_coords), identity, predicted_chain_id, native_coords, predicted_coords)
            )
        if not matches:
            raise ValueError(f"no refold chain matches input target chain {native_chain_id}")
        matched_count, identity, predicted_chain_id, native_coords, predicted_coords = max(
            matches, key=lambda item: (item[0], item[1])
        )
        used_predicted.add(predicted_chain_id)
        all_native.extend(native_coords)
        all_predicted.extend(predicted_coords)
        identities.append(identity)
        chain_pairs.append(f"{native_chain_id}:{predicted_chain_id}:{matched_count}")

    return TargetRMSDResult(
        rmsd=_kabsch_rmsd(all_native, all_predicted),
        matched_ca_atoms=len(all_native),
        mean_sequence_identity=float(np.mean(identities)),
        matched_chain_pairs=";".join(chain_pairs),
    )


def target_rmsd_fields(
    refold_path: str | Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return CSV-ready target RMSD fields without aborting batch analysis."""
    spec = target_spec_from_metadata(metadata)
    base = {
        "target_RMSD": None,
        "target_RMSD_matched_CA": None,
        "target_RMSD_sequence_identity": None,
        "target_RMSD_chain_pairs": "",
        "target_RMSD_error": "",
    }
    if spec is None:
        base["target_RMSD_error"] = "input target structure or chain metadata is unavailable"
        return base
    if not refold_path or not Path(refold_path).exists():
        base["target_RMSD_error"] = f"refold structure is unavailable: {refold_path}"
        return base
    try:
        result = calculate_target_rmsd(refold_path, spec)
    except Exception as exc:
        base["target_RMSD_error"] = str(exc)
        return base
    base.update(
        {
            "target_RMSD": result.rmsd,
            "target_RMSD_matched_CA": result.matched_ca_atoms,
            "target_RMSD_sequence_identity": result.mean_sequence_identity,
            "target_RMSD_chain_pairs": result.matched_chain_pairs,
        }
    )
    return base
