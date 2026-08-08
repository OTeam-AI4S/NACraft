"""NA-MPNN integration for nucleic acid sequence redesign.

Parallel to tied_lmpnn.py but uses NA-MPNN for NA polymer redesign.
Handles the non-standard one-letter encoding used by NA-MPNN.
"""

import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

from Bio.PDB import PDBParser, PDBIO
import numpy as np

from . import na_constants


def _select_best_sample_idx(output: dict) -> int:
    try:
        ptm = output.get("ptm", None)
        idx = np.argmax(ptm.cpu().numpy())
        print(f"NA-MPNN: Best sample index: {idx}")
    except Exception:
        idx = 0
    return idx


def _existing_best_pdb_path(design_dir: str, state_idx: int, sample_idx: int) -> str:
    return os.path.join(design_dir, f"state{state_idx}_sample{sample_idx}.pdb")


def _parse_structure(pdb_path: str):
    parser = PDBParser(QUIET=True)
    strucid = os.path.basename(pdb_path)[:4] or "1xxx"
    return parser.get_structure(strucid, pdb_path)


def _write_structure(structure, out_path: str):
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)


def _is_na_residue(residue) -> bool:
    """Check if a residue is a nucleic acid residue."""
    return residue.resname.strip() in na_constants.NA_RESIDUE_NAMES


def _count_na_residues(chain) -> int:
    return sum(1 for r in chain if _is_na_residue(r))


def _build_composite_pdb(pdb_paths_by_state: Dict[int, str], out_path: str) -> Tuple[str, Dict[int, str], int]:
    """Build composite PDB with shifted coordinates for multi-state tied redesign."""
    chain_sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    binder_chain_map: Dict[int, str] = {}

    first_state = next(iter(pdb_paths_by_state))
    base_struct = _parse_structure(pdb_paths_by_state[first_state])
    model = base_struct[0]

    for chain in list(model):
        model.detach_child(chain.id)

    binder_length = None
    shift_step = 100.0
    state_index = 0
    used_chain_letters = set()

    for state_idx, pdb_path in pdb_paths_by_state.items():
        src = _parse_structure(pdb_path)
        src_model = src[0]

        binder_letter = chain_sequence[(state_index * 3) % len(chain_sequence)]
        while binder_letter in used_chain_letters:
            state_index += 1
            binder_letter = chain_sequence[(state_index * 3) % len(chain_sequence)]
        used_chain_letters.add(binder_letter)
        binder_chain_map[state_idx] = binder_letter

        shift = np.array([state_index * shift_step, 0.0, 0.0])
        next_letter_idx = (state_index * 3 + 1) % len(chain_sequence)

        for chain in src_model:
            new_chain = chain.copy()
            for residue in new_chain:
                for atom in residue:
                    atom.set_coord(atom.get_coord() + shift)

            if chain.id == "A":
                new_id = binder_letter
                if binder_length is None:
                    binder_length = _count_na_residues(new_chain)
            else:
                new_id = chain_sequence[next_letter_idx]
                while new_id in used_chain_letters:
                    next_letter_idx = (next_letter_idx + 1) % len(chain_sequence)
                    new_id = chain_sequence[next_letter_idx]
                used_chain_letters.add(new_id)
                next_letter_idx = (next_letter_idx + 1) % len(chain_sequence)

            new_chain.id = new_id
            model.add(new_chain)

        state_index += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_structure(base_struct, out_path)
    return out_path, binder_chain_map, int(binder_length or 0)


def _get_na_interface_residues(composite_pdb_path: str, binder_chain: str, cutoff: float = 6.0) -> List[int]:
    """Find NA residues at the interface with ligands/other chains.

    Returns sorted list of 0-indexed residue positions within the binder chain.
    """
    try:
        from prody import parsePDB
        pdb = parsePDB(composite_pdb_path)
        binder = pdb.select(f"nucleic and chain {binder_chain}")
        if binder is None:
            return []
        other = pdb.select(f"not nucleic and not water and not chain {binder_chain}")
        if other is None:
            return []
        # Get residue indices of binder residues within cutoff of other chains
        binder_ca = binder.select("name C1'")
        other_ca = other.select("name CA or name C1' or name P")
        if binder_ca is None or other_ca is None:
            return []
        # Pairwise distance via numpy broadcasting. ProDy's calcDistance requires
        # same-shape arrays (element-wise); for n_binder × n_other min-distance
        # we compute it manually.
        import numpy as np
        bcoords = binder_ca.getCoords()
        ocoords = other_ca.getCoords()
        diff = bcoords[:, None, :] - ocoords[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        close_mask = dists.min(axis=1) < cutoff
        close_resindices = binder_ca.getResindices()[close_mask]
        # Map to sequential indices
        struct = _parse_structure(composite_pdb_path)
        for chain in struct[0]:
            if chain.id == binder_chain:
                na_residues = [r for r in chain if _is_na_residue(r)]
                resindex_to_seq = {}
                for i, r in enumerate(na_residues):
                    resindex_to_seq[r.id[1]] = i
                return sorted(set(
                    resindex_to_seq[ri]
                    for ri in close_resindices
                    if ri in resindex_to_seq
                ))
        return []
    except ImportError:
        print("[na_mpnn_utils] ProDy not available for interface detection; returning empty")
        return []


def _make_symmetry_groups(binder_chain_map: Dict[int, str], binder_length: int) -> Tuple[str, str]:
    groups: List[str] = []
    weights: List[str] = []
    chain_letters = [binder_chain_map[k] for k in sorted(binder_chain_map.keys())]
    for pos in range(1, binder_length + 1):
        group = ",".join([f"{cl}{pos}" for cl in chain_letters])
        groups.append(group)
        weights.append(",".join(["1.0" for _ in chain_letters]))
    return "|".join(groups), "|".join(weights)


def _parse_fasta(fasta_path: str) -> List[str]:
    """Parse NA-MPNN FASTA output.

    NA-MPNN uses non-standard encoding (lowercase for DNA, special chars for RNA).
    We extract the binder chain sequence (first chain before ':').
    """
    sequences = []
    with open(fasta_path, "r") as f:
        lines = f.readlines()
    seq = None
    overall = None
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if seq is not None:
                sequences.append((overall, seq))
                seq = None
            try:
                parts = line.split(",")
                overall = float(parts[4].split("=")[-1])
            except Exception:
                overall = -1.0
        else:
            seq = line
    if seq is not None:
        sequences.append((overall, seq))

    return [seq[1].split(":")[0] for seq in sequences]


def nampnn_seq_to_boltz(seq: str, polymer_type: str) -> str:
    """Convert NA-MPNN one-letter sequence to Boltz-compatible sequence.

    NA-MPNN encoding:
      RNA: b=A, d=C, h=G, u=U
      DNA: a=DA, c=DC, g=DG, t=DT
    """
    result = []
    for c in seq:
        if polymer_type == "rna":
            result.append(na_constants.NA_MPNN_RNA_REVERSE.get(c, "N"))
        elif polymer_type == "dna":
            result.append(na_constants.NA_MPNN_DNA_REVERSE.get(c, "N"))
        else:
            result.append(c)
    return "".join(result)


def perform_tied_nampnn_redesign(
    design_dir: str,
    state_results: List[Tuple[dict, list, int]],
    polymer_type: str = "rna",
    num_seqs: int = 8,
    motif_indices: Optional[List[int]] = None,
    temperature: float = 0.1,
    seed: int = 0,
):
    """Run tied NA-MPNN redesign across multiple states.

    Parallel to perform_tied_lmpnn_redesign but for NA polymers.
    """
    assert polymer_type in ("rna", "dna"), f"NA-MPNN requires rna or dna, got {polymer_type}"

    # 1) Pick best sample per state
    best_idx_by_state: Dict[int, int] = {}
    for output, struct_list, state_idx in state_results:
        best_idx_by_state[state_idx] = _select_best_sample_idx(output)

    # 2) Collect PDB paths
    pdb_paths_by_state: Dict[int, str] = {}
    for state_idx, best_j in best_idx_by_state.items():
        pdb_path = _existing_best_pdb_path(design_dir, state_idx, best_j)
        if not os.path.exists(pdb_path):
            pdb_path = _existing_best_pdb_path(design_dir, state_idx, 0)
        pdb_paths_by_state[state_idx] = pdb_path

    # 3) Build composite PDB
    nampnn_dir = os.path.join(design_dir, "nampnn")
    composite_pdb_path = os.path.join(nampnn_dir, "composite.pdb")
    composite_pdb_path, binder_chain_map, binder_length = _build_composite_pdb(
        pdb_paths_by_state, composite_pdb_path
    )

    # 4) Compute fixed residues (interface + motif)
    fixed_tokens: List[str] = []
    for _, chain_letter in binder_chain_map.items():
        interface_res = _get_na_interface_residues(composite_pdb_path, chain_letter, cutoff=6.0)
        fixed_tokens.extend([f"{chain_letter}{i+1}" for i in interface_res])

    if motif_indices:
        motif_tokens: List[str] = []
        for _, chain_letter in binder_chain_map.items():
            motif_tokens.extend([f"{chain_letter}{i+1}" for i in motif_indices])
        fixed_tokens.extend(motif_tokens)

    fixed_residues = " ".join(fixed_tokens)

    # 5) Build symmetry groups
    sym_res, sym_wts = _make_symmetry_groups(binder_chain_map, binder_length)

    # 6) Locate NA-MPNN checkpoint
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    nampnn_root = os.path.join(repo_root, "NA-MPNN")
    checkpoint_path = os.path.join(nampnn_root, "models", "design_model", "s_19137.pt")

    if not os.path.exists(checkpoint_path):
        # Try alternate locations
        checkpoint_path = os.path.join(repo_root, "NA-MPNN", "model_params", "na_mpnn.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"NA-MPNN checkpoint not found. Looked in {nampnn_root}/models/design_model/ "
                f"and {nampnn_root}/model_params/. Please download the model weights."
            )

    # 7) Run NA-MPNN via subprocess (to avoid import path conflicts)
    chains_to_design = ",".join([binder_chain_map[k] for k in sorted(binder_chain_map.keys())])
    out_folder = nampnn_dir
    os.makedirs(out_folder, exist_ok=True)

    # Determine omit_AA: suppress all protein AA when designing NA only
    omit_aa = "ARNDCQEGHILKMFPSTWYVX"  # keep only NA tokens active

    import subprocess
    cmd = [
        sys.executable,
        os.path.join(nampnn_root, "inference", "run.py"),
        "--model_type", "na_mpnn",
        "--checkpoint_na_mpnn", checkpoint_path,
        "--pdb_path", composite_pdb_path,
        "--out_folder", out_folder,
        "--mode", "design",
        "--chains_to_design", chains_to_design,
        "--fixed_residues", fixed_residues,
        "--symmetry_residues", sym_res,
        "--symmetry_weights", sym_wts,
        "--omit_AA", omit_aa,
        "--design_na_only", "1",
        "--parse_na_only", "1",
        "--na_shared_tokens", "0",
        "--number_of_batches", str(num_seqs),
        "--temperature", str(temperature),
        "--output_pdbs", "0",
        "--output_sequences", "1",
        "--seed", str(seed),
    ]

    print(f"[NA-MPNN] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=nampnn_root)
    if result.returncode != 0:
        print(f"[NA-MPNN] STDERR: {result.stderr[:2000]}")
        print(f"[NA-MPNN] STDOUT: {result.stdout[:2000]}")
        raise RuntimeError(f"NA-MPNN failed with return code {result.returncode}: {result.stderr[:500]}")

    # 8) Parse outputs
    base = os.path.splitext(os.path.basename(composite_pdb_path))[0]
    fasta_path = os.path.join(out_folder, "seqs", f"{base}.fa")

    top_sequences = _parse_fasta(fasta_path) if os.path.exists(fasta_path) else []

    # Convert NA-MPNN encoding to Boltz-compatible sequence
    converted_sequences = []
    for seq in top_sequences:
        converted_sequences.append(nampnn_seq_to_boltz(seq, polymer_type))

    return converted_sequences, fasta_path, best_idx_by_state
