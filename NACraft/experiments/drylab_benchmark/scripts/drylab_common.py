from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ODESIGN_VISIBLE_RNA = {"7WKP", "7YEW", "7YGL", "9C7A", "8TG4"}
ODESIGN_VISIBLE_DNA = {"7XVN", "7YSF", "7YUK", "8PMF"}

RNA_ALPHABET = "ACGU"
DNA_ALPHABET = "ACGT"
PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
RNA_RESIDUES = {"A", "C", "G", "U", "RA", "RC", "RG", "RU"}
DNA_RESIDUES = {"DA", "DC", "DG", "DT", "A", "C", "G", "T"}


@dataclass(frozen=True)
class AtomRecord:
    target_id: str
    molecule_type: str
    chain_id: str
    residue_number: int
    residue_name: str
    atom_name: str
    x: float
    y: float
    z: float

    @property
    def residue_key(self) -> str:
        return f"{self.chain_id}:{self.residue_number}:{self.residue_name}"


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_json_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(";") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def alphabet_for_polymer(polymer_type: str) -> str:
    polymer_type = polymer_type.lower()
    if polymer_type == "rna":
        return RNA_ALPHABET
    if polymer_type == "dna":
        return DNA_ALPHABET
    raise ValueError(f"Unsupported polymer_type: {polymer_type}")


def select_length_stratified_targets(
    rows: Sequence[Mapping[str, str]],
    polymer_type: str,
    bins: Sequence[tuple[int, int, int]],
    exclude_target_ids: set[str] | None = None,
    max_protein_length: int | None = None,
    unique_protein_sequence: bool = False,
    random_seed: int | None = None,
) -> list[dict[str, str]]:
    exclude_target_ids = {item.upper() for item in (exclude_target_ids or set())}
    candidates = [
        dict(row)
        for row in rows
        if str(row.get("polymer_type", "")).lower() == polymer_type.lower()
        and str(row.get("target_id", "")).upper() not in exclude_target_ids
        and (
            max_protein_length is None
            or len(str(row.get("protein_sequence", "")).replace(":", "")) <= max_protein_length
        )
    ]
    rng = random.Random(random_seed) if random_seed is not None else None
    candidates.sort(key=lambda row: (str(row.get("release_date", "")), str(row.get("target_id", ""))))
    selected: list[dict[str, str]] = []
    used: set[str] = set()
    used_protein_sequences: set[str] = set()
    for low, high, count in bins:
        in_bin = [
            row
            for row in candidates
            if row.get("target_id") not in used and low <= int(float(row.get("na_length", 0))) <= high
            and (not unique_protein_sequence or row.get("protein_sequence") not in used_protein_sequences)
        ]
        if rng is not None:
            rng.shuffle(in_bin)
        selected_in_bin = 0
        for row in in_bin:
            if selected_in_bin >= count:
                break
            if unique_protein_sequence and row.get("protein_sequence") in used_protein_sequences:
                continue
            out = dict(row)
            out["selection_reason"] = f"length_bin_{low}_{high}"
            selected.append(out)
            used.add(str(row.get("target_id")))
            used_protein_sequences.add(str(row.get("protein_sequence", "")))
            selected_in_bin += 1
        missing = count - selected_in_bin
        if missing > 0:
            fallback = [
                row
                for row in candidates
                if row.get("target_id") not in used
                and (not unique_protein_sequence or row.get("protein_sequence") not in used_protein_sequences)
            ]
            fallback.sort(key=lambda row: (abs(int(float(row.get("na_length", 0))) - ((low + high) // 2)), row.get("target_id", "")))
            if rng is not None:
                grouped: dict[int, list[dict[str, str]]] = {}
                center = (low + high) // 2
                for row in fallback:
                    grouped.setdefault(abs(int(float(row.get("na_length", 0))) - center), []).append(row)
                fallback = []
                for distance_key in sorted(grouped):
                    group = grouped[distance_key]
                    rng.shuffle(group)
                    fallback.extend(group)
            for row in fallback[:missing]:
                out = dict(row)
                out["selection_reason"] = f"adjacent_fill_for_{low}_{high}"
                selected.append(out)
                used.add(str(row.get("target_id")))
                used_protein_sequences.add(str(row.get("protein_sequence", "")))
    return selected


def filter_manifest_candidates(
    rows: Sequence[Mapping[str, str]],
    min_na_length: int = 1,
    max_na_length: int = 10_000,
    max_protein_length: int = 2000,
    max_protein_chains: int = 4,
    max_na_chains: int = 1,
    require_unambiguous_sequence: bool = True,
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        na_length = int(float(row.get("na_length", 0)))
        protein_sequence = str(row.get("protein_sequence", ""))
        protein_length = len(protein_sequence.replace(":", ""))
        protein_chains = parse_json_list(row.get("protein_chains"))
        na_chains = parse_json_list(row.get("na_chains"))
        polymer_type = str(row.get("polymer_type", "")).lower()
        native_na_sequence = str(row.get("native_na_sequence", "")).upper()
        reasons = []
        if not (min_na_length <= na_length <= max_na_length):
            reasons.append("na_length")
        if protein_length > max_protein_length:
            reasons.append("protein_length")
        if len(protein_chains) > max_protein_chains:
            reasons.append("protein_chains")
        if len(na_chains) > max_na_chains:
            reasons.append("na_chains")
        if require_unambiguous_sequence and native_na_sequence:
            allowed = set(alphabet_for_polymer(polymer_type))
            if set(native_na_sequence) - allowed:
                reasons.append("ambiguous_na_sequence")
        if reasons:
            continue
        out = dict(row)
        out["filter_reason"] = "pass"
        filtered.append(out)
    return filtered


def distance(a: AtomRecord, b: AtomRecord) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def compute_hotspots(
    protein_atoms: Sequence[AtomRecord],
    na_atoms: Sequence[AtomRecord],
    cutoff: float = 5.0,
    top_n: int = 4,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    def cell(atom: AtomRecord) -> tuple[int, int, int]:
        return (
            math.floor(atom.x / cutoff),
            math.floor(atom.y / cutoff),
            math.floor(atom.z / cutoff),
        )

    na_grid: dict[tuple[int, int, int], list[AtomRecord]] = {}
    for natom in na_atoms:
        na_grid.setdefault(cell(natom), []).append(natom)

    contacts: dict[str, dict[str, object]] = {}
    for patom in protein_atoms:
        cx, cy, cz = cell(patom)
        nearby_atoms = (
            natom
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            for natom in na_grid.get((cx + dx, cy + dy, cz + dz), [])
        )
        for natom in nearby_atoms:
            d = distance(patom, natom)
            if d <= cutoff:
                entry = contacts.setdefault(
                    patom.residue_key,
                    {
                        "residue_key": patom.residue_key,
                        "chain_id": patom.chain_id,
                        "residue_number": patom.residue_number,
                        "residue_name": patom.residue_name,
                        "contact_count": 0,
                        "min_distance": d,
                    },
                )
                entry["contact_count"] = int(entry["contact_count"]) + 1
                entry["min_distance"] = min(float(entry["min_distance"]), d)
    patch = sorted(contacts.values(), key=lambda row: (-int(row["contact_count"]), float(row["min_distance"]), str(row["residue_key"])))
    return patch, patch[:top_n]


def build_nacraft_config(
    target: Mapping[str, str],
    mode: str,
    sequence_guidance_weight: float = 0.1,
) -> dict[str, object]:
    mode = mode.lower()
    mode = {
        "sequence_guided": "similarity_guided",
        "conformation_selective": "target_selective",
    }.get(mode, mode)
    polymer_type = str(target["polymer_type"]).lower()
    length = int(float(target["na_length"]))
    protein_sequence = target.get("protein_sequence", "")
    losses: list[dict[str, object]] = []
    init_seq = ""
    ablation = ""
    if mode in {"denovo", "similarity_guided", "no_polymer_specific"}:
        losses.append({"type": "LigandContactLoss", "state": 0})
    if mode == "similarity_guided":
        init_seq = str(target.get("native_na_sequence") or target.get("seed_sequence") or "")
        losses.append(
            {
                "type": "SequenceSimilarityLoss",
                "target_sequence": init_seq,
                "strength": sequence_guidance_weight,
            }
        )
    elif mode == "target_selective":
        positive_sequence = str(
            target.get("positive_protein_sequence")
            or target.get("holo_protein_sequence")
            or target.get("target_protein_sequence")
            or target.get("protein_sequence")
            or ""
        )
        negative_sequence = str(
            target.get("negative_protein_sequence")
            or target.get("apo_protein_sequence")
            or target.get("anti_target_protein_sequence")
            or ""
        )
        if not positive_sequence or not negative_sequence:
            raise ValueError(
                "target_selective mode requires positive and negative "
                "protein sequences in the manifest"
            )
        losses.extend(
            [
                {"type": "AntiLigandContactLoss", "state": 0, "strength": 1.0},
                {"type": "LigandContactLoss", "state": 1, "strength": 1.0},
            ]
        )
        hotspots = parse_json_list(target.get("hotspots") or target.get("selected_hotspots"))
        target_id = target.get("target_id") or target.get("benchmark_id")
        if not target_id:
            raise ValueError("target_selective mode requires target_id or benchmark_id")
        return {
            "target_id": target_id,
            "method": "NACraft-target-selective",
            "polymer_type": polymer_type,
            "predictor": "boltz",
            "num_states": 2,
            "length": length,
            "motifs": [],
            "states": [[f"protein:{negative_sequence}"], [f"protein:{positive_sequence}"]],
            "hotspots": hotspots,
            "losses": losses,
            "loss_types": [str(loss["type"]) for loss in losses],
            "init_seq": init_seq,
            "ablation": "",
            "contact_loss": False,
        }
    elif mode == "antibind_loss":
        ablation = "NACraft-antibind-loss"
        losses.append({"type": "AntiLigandContactLoss", "state": 0, "strength": 1.0})
    elif mode == "no_polymer_specific":
        ablation = "NACraft-no-polymer-specific"
    elif mode != "denovo":
        raise ValueError(f"Unsupported NACraft config mode: {mode}")
    hotspots = parse_json_list(target.get("hotspots") or target.get("selected_hotspots"))
    return {
        "target_id": target["target_id"],
        "method": f"NACraft-{mode}",
        "polymer_type": polymer_type,
        "predictor": "boltz",
        "num_states": 1,
        "length": length,
        "motifs": [],
        "states": [[f"protein:{protein_sequence}"]],
        "hotspots": hotspots,
        "losses": losses,
        "loss_types": [str(loss["type"]) for loss in losses],
        "init_seq": init_seq,
        "ablation": ablation,
        "contact_loss": False if mode == "antibind_loss" else True,
    }


def to_float(value: object, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ranking_score(row: Mapping[str, object]) -> float:
    iptm = to_float(row.get("iptm"), 0.0) or 0.0
    plddt = to_float(row.get("plddt_aptamer") or row.get("plddt"), 0.0) or 0.0
    ipae = to_float(row.get("ipae") or row.get("interface_pae"), None)
    score = iptm + 0.01 * plddt
    if ipae is not None:
        score -= 0.05 * ipae
    return score


def best_by_method_target(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    best: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["target_id"]), str(row["method"]))
        score = ranking_score(row)
        candidate = dict(row)
        candidate["score"] = score
        if key not in best or score > float(best[key]["score"]):
            best[key] = candidate
    return best


def best_of_k(rows: Sequence[Mapping[str, object]], ks: Sequence[int] = (1, 5, 10, 20, 50, 100, 200, 400)) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        candidate = dict(row)
        candidate["score"] = ranking_score(candidate)
        groups.setdefault((str(candidate["target_id"]), str(candidate["method"])), []).append(candidate)
    out: list[dict[str, object]] = []
    for (target_id, method), candidates in groups.items():
        for k in ks:
            subset = candidates[:k]
            if not subset:
                continue
            best = max(subset, key=lambda row: float(row["score"]))
            out.append({"target_id": target_id, "method": method, "k": k, "best_score": best["score"], "candidate_id": best.get("candidate_id", "")})
    return out


def manifest_qc_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        hotspots = parse_json_list(row.get("selected_hotspots") or row.get("hotspots"))
        patch = parse_json_list(row.get("interface_patch"))
        structure_path = str(row.get("structure_path", ""))
        out.append(
            {
                "target_id": row.get("target_id", ""),
                "polymer_type": row.get("polymer_type", ""),
                "na_length": row.get("na_length", ""),
                "selection_reason": row.get("selection_reason", ""),
                "hotspot_count": len(hotspots),
                "interface_patch_size": len(patch),
                "structure_path": structure_path,
                "structure_exists": Path(structure_path).exists() if structure_path else False,
            }
        )
    return out


def write_simple_yaml(path: str | Path, data: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_yaml(data), encoding="utf-8")


def _format_yaml(value: object, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if item == []:
                lines.append(f"{pad}{key}: []")
            elif isinstance(item, (Mapping, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_format_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{pad}{key}: {json.dumps(item) if isinstance(item, str) and ':' in item else item}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (Mapping, list)):
                lines.append(f"{pad}-")
                lines.append(_format_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{pad}- {item}")
        return "\n".join(lines) + "\n"
    return f"{pad}{value}\n"


def add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output path")


def parse_structure_atoms(path: str | Path, target_id: str | None = None) -> list[AtomRecord]:
    path = Path(path)
    target_id = target_id or path.stem
    if path.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError(f"Only PDB files are supported by the lightweight parser: {path}")
    raw: list[dict[str, object]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21].strip() or "_"
        try:
            residue_number = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        raw.append(
            {
                "chain_id": chain_id,
                "residue_number": residue_number,
                "residue_name": residue_name,
                "atom_name": atom_name,
                "x": x,
                "y": y,
                "z": z,
            }
        )
    chain_residues: dict[str, set[str]] = {}
    for atom in raw:
        chain_residues.setdefault(str(atom["chain_id"]), set()).add(str(atom["residue_name"]))
    chain_types: dict[str, str] = {}
    for chain_id, residues in chain_residues.items():
        if residues & PROTEIN_RESIDUES:
            chain_types[chain_id] = "protein"
        elif residues & {"U", "RU"}:
            chain_types[chain_id] = "rna"
        elif residues & {"T", "DT", "DA", "DC", "DG"}:
            chain_types[chain_id] = "dna"
        elif residues <= {"A", "C", "G"}:
            chain_types[chain_id] = "rna"
        else:
            chain_types[chain_id] = "unknown"
    atoms = []
    for atom in raw:
        chain_id = str(atom["chain_id"])
        atoms.append(
            AtomRecord(
                target_id=target_id,
                molecule_type=chain_types.get(chain_id, "unknown"),
                chain_id=chain_id,
                residue_number=int(atom["residue_number"]),
                residue_name=str(atom["residue_name"]),
                atom_name=str(atom["atom_name"]),
                x=float(atom["x"]),
                y=float(atom["y"]),
                z=float(atom["z"]),
            )
        )
    return atoms


def build_target_manifest_from_structures(paths: Sequence[str | Path], release_date: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        path = Path(path)
        target_id = path.stem
        atoms = parse_structure_atoms(path, target_id=target_id)
        protein_chains = sorted({atom.chain_id for atom in atoms if atom.molecule_type == "protein"})
        na_atoms = [atom for atom in atoms if atom.molecule_type in {"rna", "dna"}]
        if not protein_chains or not na_atoms:
            continue
        polymer_type = "rna" if any(atom.molecule_type == "rna" for atom in na_atoms) else "dna"
        na_chains = sorted({atom.chain_id for atom in na_atoms if atom.molecule_type == polymer_type})
        na_residues = {(atom.chain_id, atom.residue_number) for atom in na_atoms if atom.molecule_type == polymer_type}
        rows.append(
            {
                "target_id": target_id,
                "release_date": release_date,
                "polymer_type": polymer_type,
                "na_length": len(na_residues),
                "protein_chains": json.dumps(protein_chains),
                "na_chains": json.dumps(na_chains),
                "structure_path": str(path),
            }
        )
    return rows


def rcsb_entries_to_manifest_candidates(
    payload: Mapping[str, object],
    polymer_type: str,
    structure_root: str | Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_polymer = "RNA" if polymer_type.lower() == "rna" else "DNA"
    entries = payload.get("data", {}).get("entries", []) if isinstance(payload.get("data"), Mapping) else []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        target_id = str(entry.get("rcsb_id", "")).upper()
        release = ""
        accession = entry.get("rcsb_accession_info")
        if isinstance(accession, Mapping):
            release = str(accession.get("initial_release_date", ""))[:10]
        proteins = []
        nas = []
        for entity in entry.get("polymer_entities", []) or []:
            if not isinstance(entity, Mapping):
                continue
            entity_poly = entity.get("entity_poly") or {}
            identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
            if not isinstance(entity_poly, Mapping) or not isinstance(identifiers, Mapping):
                continue
            entity_type = str(entity_poly.get("rcsb_entity_polymer_type", ""))
            sequence = str(entity_poly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "").replace(" ", "")
            chains = identifiers.get("auth_asym_ids") or identifiers.get("asym_ids") or []
            item = {"sequence": sequence, "chains": [str(chain) for chain in chains]}
            if entity_type == "Protein":
                proteins.append(item)
            elif entity_type == target_polymer:
                nas.append(item)
        if not proteins or not nas:
            continue
        na = max(nas, key=lambda item: len(item["sequence"]))
        protein_sequence = ":".join(item["sequence"] for item in proteins if item["sequence"])
        rows.append(
            {
                "target_id": target_id,
                "release_date": release,
                "polymer_type": polymer_type.lower(),
                "na_length": len(na["sequence"]),
                "native_na_sequence": na["sequence"],
                "protein_sequence": protein_sequence,
                "protein_chains": json.dumps(sorted({chain for item in proteins for chain in item["chains"]})),
                "na_chains": json.dumps(na["chains"]),
                "structure_path": str(Path(structure_root) / f"{target_id}.pdb"),
            }
        )
    return rows


def collect_af3_metrics(root: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(root).rglob("*summary*.json")):
        data = json.loads(path.read_text())
        stem = path.name.replace("_summary", "").replace(".json", "")
        parts = stem.split("__")
        target_id = parts[0] if len(parts) >= 1 else path.parent.name
        method = parts[1] if len(parts) >= 2 else path.parent.parent.name
        candidate_id = parts[2] if len(parts) >= 3 else path.stem.replace("_summary", "")
        chain_plddt = data.get("chain_plddt") or data.get("chain_ptm_plddt") or []
        plddt_aptamer = chain_plddt[-1] if isinstance(chain_plddt, list) and chain_plddt else data.get("plddt")
        ipae = data.get("ipae") or data.get("interface_pae")
        if ipae is None:
            pair = data.get("chain_pair_pae_min") or data.get("chain_pair_pae")
            if isinstance(pair, list) and len(pair) >= 2 and isinstance(pair[0], list):
                values = []
                for i, row in enumerate(pair):
                    for j, value in enumerate(row):
                        if i != j:
                            values.append(float(value))
                ipae = sum(values) / len(values) if values else ""
        rows.append(
            {
                "target_id": target_id,
                "method": method,
                "candidate_id": candidate_id,
                "iptm": data.get("iptm") or data.get("ranking_confidence") or "",
                "plddt": data.get("plddt") or "",
                "plddt_aptamer": plddt_aptamer or "",
                "ipae": ipae or "",
                "summary_path": str(path),
            }
        )
    return rows


def protein_aligned_rmsd(
    native_protein: Sequence[Sequence[float]],
    pred_protein: Sequence[Sequence[float]],
    native_na: Sequence[Sequence[float]],
    pred_na: Sequence[Sequence[float]],
) -> float:
    try:
        import numpy as np
    except ImportError:
        if len(native_protein) != len(pred_protein) or len(native_na) != len(pred_na):
            raise ValueError("Native and predicted coordinate lists must have matching lengths")
        q_centroid = tuple(sum(coord[i] for coord in native_protein) / len(native_protein) for i in range(3))
        p_centroid = tuple(sum(coord[i] for coord in pred_protein) / len(pred_protein) for i in range(3))
        sq = 0.0
        for native, pred in zip(native_na, pred_na):
            aligned = tuple(float(pred[i]) - p_centroid[i] + q_centroid[i] for i in range(3))
            sq += sum((aligned[i] - float(native[i])) ** 2 for i in range(3))
        return math.sqrt(sq / len(native_na))
    if len(native_protein) != len(pred_protein) or len(native_na) != len(pred_na):
        raise ValueError("Native and predicted coordinate lists must have matching lengths")
    if len(native_protein) < 3:
        raise ValueError("At least three protein anchor atoms are required for Kabsch alignment")
    p = np.asarray(pred_protein, dtype=float)
    q = np.asarray(native_protein, dtype=float)
    p_centroid = p.mean(axis=0)
    q_centroid = q.mean(axis=0)
    p0 = p - p_centroid
    q0 = q - q_centroid
    covariance = p0.T @ q0
    v, _s, wt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(v @ wt)
    rotation = v @ correction @ wt
    pred_na_arr = np.asarray(pred_na, dtype=float)
    native_na_arr = np.asarray(native_na, dtype=float)
    aligned = (pred_na_arr - p_centroid) @ rotation + q_centroid
    diff = aligned - native_na_arr
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))
