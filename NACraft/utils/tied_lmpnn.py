import os
import json
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

from Bio.PDB import PDBParser, PDBIO
import numpy as np

from .ligandmpnn_utils import get_protein_ligand_interface_all_atom


def _select_best_sample_idx(output: dict) -> int:
    # return index of best sample by ptm score if available
    # otherwise just take the first sample
    try:
        ptm = output.get("ptm", None)
        idx = np.argmax(ptm.cpu().numpy())
        print("Using ptm to select best structure per state.")
        print(f"Best sample index: {idx}")
    except Exception:
        print("Defaulting to sample 0 as best structure per state.")
        idx = 0
        pass
    
    return idx


def _existing_best_pdb_path(design_dir: str, state_idx: int, sample_idx: int) -> str:
    # mydesign.py writes state{state}_sample{j}.pdb; we reference that
    return os.path.join(design_dir, f"state{state_idx}_sample{sample_idx}.pdb")


def _parse_structure(pdb_path: str):
    parser = PDBParser(QUIET=True)
    # strucid must be non-empty
    strucid = os.path.basename(pdb_path)[:4] or "1xxx"
    return parser.get_structure(strucid, pdb_path)


def _is_protein_res(residue) -> bool:
    hetflag = residue.id[0]
    return hetflag == " " and residue.resname not in ("HOH",)


def _has_non_protein(context_structure) -> bool:
    for model in context_structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " " and residue.resname != "HOH":
                    return True
    return False


def _write_structure(structure, out_path: str):
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)


def _build_composite_pdb(pdb_paths_by_state: Dict[int, str], out_path: str) -> Tuple[str, Dict[int, str], int]:
    # Assign binder chain letters spaced by 3 to allow other context chains between
    chain_sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    binder_chain_map: Dict[int, str] = {}

    # Create an empty structure by copying the first then clearing chains
    first_state = next(iter(pdb_paths_by_state))
    base_struct = _parse_structure(pdb_paths_by_state[first_state])
    model = base_struct[0]

    # Remove all chains from model; we'll repopulate
    for chain in list(model):
        model.detach_child(chain.id)

    # Build composite
    binder_length = None
    shift_step = 100.0
    state_index = 0
    used_chain_letters = set()

    for state_idx, pdb_path in pdb_paths_by_state.items():
        src = _parse_structure(pdb_path)
        src_model = src[0]

        # Assign binder chain letter for this state
        binder_letter = chain_sequence[(state_index * 3) % len(chain_sequence)]
        while binder_letter in used_chain_letters:
            state_index += 1
            binder_letter = chain_sequence[(state_index * 3) % len(chain_sequence)]
        used_chain_letters.add(binder_letter)
        binder_chain_map[state_idx] = binder_letter

        # Compute translation
        shift = np.array([state_index * shift_step, 0.0, 0.0])

        # Copy chains; change binder 'A' to binder_letter, others to next available letters
        next_letter_idx = (state_index * 3 + 1) % len(chain_sequence)

        for chain in src_model:
            new_chain = chain.copy()
            # translate coords
            for residue in new_chain:
                for atom in residue:
                    atom.set_coord(atom.get_coord() + shift)

            if chain.id == "A":
                new_id = binder_letter
                if binder_length is None:
                    binder_length = sum(1 for r in new_chain if _is_protein_res(r))
            else:
                # allocate a unique chain id for context
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


def _get_fixed_residues_union(composite_pdb_path: str, binder_chain_map: Dict[int, str], model_type: str, cutoff: float = 6.0) -> str:
    fixed_tokens: List[str] = []
    non_protein_target = True if model_type == "ligand_mpnn" else False

    # basically aggregate the fixed indices across all chains
    for _, chain_letter in binder_chain_map.items():
        indices = get_protein_ligand_interface_all_atom(
            composite_pdb_path,
            cutoff=cutoff,
            non_protein_target=non_protein_target,
            binder_chain=chain_letter,
            target_chains="all",
        )
    
    # go through each state and add the fixed indices to that list
    for _, chain_letter in binder_chain_map.items():
        fixed_tokens.extend([f"{chain_letter}{i+1}" for i in indices])
    return " ".join(fixed_tokens)


def _make_symmetry_groups(binder_chain_map: Dict[int, str], binder_length: int) -> Tuple[str, str]:
    # Groups like: A1,D1,G1|A2,D2,G2|...; weights all 1.0
    groups: List[str] = []
    weights: List[str] = []
    chain_letters = [binder_chain_map[k] for k in sorted(binder_chain_map.keys())]
    for pos in range(1, binder_length + 1):
        group = ",".join([f"{cl}{pos}" for cl in chain_letters])
        groups.append(group)
        weights.append(",".join(["1.0" for _ in chain_letters]))
    return "|".join(groups), "|".join(weights)


def _detect_model_type(composite_pdb_path: str) -> str:
    struct = _parse_structure(composite_pdb_path)
    return "ligand_mpnn" if _has_non_protein(struct) else "soluble_mpnn"


def _parse_fasta(fasta_path: str) -> List[str]:
    sequences = []
    with open(fasta_path, "r") as f:
        lines = f.readlines()
    seq = None
    overall = None
    ligandc = None
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if seq is not None:
                sequences.append((overall, ligandc, seq))
                seq = None
            parts = line.split(",")
            try:
                overall = float(parts[4].split("=")[-1])
                ligandc = float(parts[5].split("=")[-1])
            except Exception:
                overall = -1.0
                ligandc = -1.0
        else:
            seq = line
    if seq is not None:
        sequences.append((overall, ligandc, seq))

    return [seq[2].split(":")[0] for seq in sequences]



def perform_tied_lmpnn_redesign(
    design_dir: str,
    state_results: List[Tuple[dict, list, int]],
    num_seqs: int = 8,
    motif_indices: Optional[List[int]] = None,
):
    # 1) pick best sample per state (fallback 0)
    best_idx_by_state: Dict[int, int] = {}
    for output, struct_list, state_idx in state_results:
        best_idx_by_state[state_idx] = _select_best_sample_idx(output)

    # 2) collect PDB paths for chosen samples
    # TODO: after removing the pdb saving from earlier, generate temp pdbs here

    pdb_paths_by_state: Dict[int, str] = {}
    for state_idx, best_j in best_idx_by_state.items():
        pdb_path = _existing_best_pdb_path(design_dir, state_idx, best_j)
        if not os.path.exists(pdb_path):
            # fallback to sample 0
            pdb_path = _existing_best_pdb_path(design_dir, state_idx, 0)
        pdb_paths_by_state[state_idx] = pdb_path

    # 3) build composite PDB
    lmpnn_dir = os.path.join(design_dir, "lmpnn")
    composite_pdb_path = os.path.join(lmpnn_dir, "composite.pdb")
    composite_pdb_path, binder_chain_map, binder_length = _build_composite_pdb(
        pdb_paths_by_state, composite_pdb_path
    )

    # 4) select model type
    model_type = _detect_model_type(composite_pdb_path)

    # 5) compute fixed residues union
    fixed_residues = _get_fixed_residues_union(
        composite_pdb_path, binder_chain_map, model_type, cutoff=6.0
    )

    # 5.5) optional step - fix motif residues
    if motif_indices:
        motif_tokens: List[str] = []
        for _, chain_letter in binder_chain_map.items():
            motif_tokens.extend([f"{chain_letter}{i+1}" for i in motif_indices])
        fixed_residues = (fixed_residues + " " + " ".join(motif_tokens)).strip()

    # 6) build symmetry groups
    sym_res, sym_wts = _make_symmetry_groups(binder_chain_map, binder_length)

    # 7) run LigandMPNN once
    from LigandMPNN.run import main as lmpnn_main

    out_folder = lmpnn_dir
    os.makedirs(out_folder, exist_ok=True)

    chains_to_design = ",".join([binder_chain_map[k] for k in sorted(binder_chain_map.keys())])

    # resolve checkpoint paths from repo
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    model_params_dir = os.path.join(repo_root, "LigandMPNN", "model_params")
    checkpoint_ligand = os.path.join(model_params_dir, "ligandmpnn_v_32_010_25.pt")
    checkpoint_soluble = os.path.join(model_params_dir, "solublempnn_v_48_020.pt")
    checkpoint_protein = os.path.join(model_params_dir, "proteinmpnn_v_48_020.pt")

    config = SimpleNamespace(
        model_type=model_type,
        checkpoint_protein_mpnn=checkpoint_protein,
        checkpoint_ligand_mpnn=checkpoint_ligand,
        checkpoint_soluble_mpnn=checkpoint_soluble,
        pdb_path=composite_pdb_path,
        pdb_path_multi="",
        fixed_residues=fixed_residues,
        fixed_residues_multi="",
        redesigned_residues="",
        redesigned_residues_multi="",
        bias_AA="",
        bias_AA_per_residue="",
        bias_AA_per_residue_multi="",
        omit_AA="",
        omit_AA_per_residue="",
        omit_AA_per_residue_multi="",
        symmetry_residues=sym_res,
        symmetry_weights=sym_wts,
        homo_oligomer=0,
        out_folder=out_folder,
        file_ending="",
        zero_indexed=0,
        seed=0,
        batch_size=1,
        number_of_batches=int(num_seqs),
        temperature=0.1,
        save_stats=0,
        ligand_mpnn_use_atom_context=1,
        ligand_mpnn_cutoff_for_score=8.0,
        ligand_mpnn_use_side_chain_context=0,
        chains_to_design=chains_to_design,
        parse_these_chains_only="",
        transmembrane_buried="",
        transmembrane_interface="",
        global_transmembrane_label=0,
        parse_atoms_with_zero_occupancy=0,
        verbose=0,
        fasta_seq_separation=":",
        dont_write_backbones=0, # this means that we DO write the backbones 
    )

    lmpnn_main(config)

    # 8) parse outputs
    base = os.path.splitext(os.path.basename(composite_pdb_path))[0]
    fasta_path = os.path.join(out_folder, "seqs", f"{base}.fa")

    #TODO: also return the indices of the structure we sampled from
    top_sequences = _parse_fasta(fasta_path) if os.path.exists(fasta_path) else []

    return top_sequences, fasta_path, best_idx_by_state


