import numpy as np
from . import protein, residue_constants
from . import na_constants
from copy import deepcopy


def get_motif_scaffold_templates(paths, target_length=None, polymer_type="protein"):

    specs = [load_motif_spec(path) for path in paths]
    spec = merge_motif_specs(specs)
    masks = sample_motif_mask(spec)
    full_motif_mask = masks['sequence']
    motif_groups = masks['group']
    motif_templates = []

    is_na = polymer_type in ("rna", "dna")

    for i, path in enumerate(paths):
        motif_idx = i + 1
        motif_mask = motif_groups == motif_idx
        with open(path) as f:
            prot = protein.from_pdb_string(f.read(), polymer_type=polymer_type)

        # Select reference atom based on polymer type
        if is_na:
            ref_atom_name = na_constants.NA_CENTER_ATOM  # C1'
            # Find the atom index — NA atoms are not in residue_constants.atom_order
            # We use the position from prot directly by name
            ref_pos = np.zeros((len(motif_mask), 3))
            motif_indices = np.where(motif_mask)[0]

            # Use na_atom_positions dict if available (from NA-aware parsing)
            if hasattr(prot, 'na_atom_positions') and prot.na_atom_positions is not None:
                for j, midx in enumerate(motif_indices):
                    if midx < len(prot.na_atom_positions):
                        atom_pos = prot.na_atom_positions[midx]
                        if ref_atom_name in atom_pos:
                            ref_pos[midx] = atom_pos[ref_atom_name]
            else:
                # Fallback: use atom_positions array and scan for C1'
                _fill_na_ref_pos(prot, ref_pos, motif_indices, ref_atom_name)

            seq = [na_constants.boltz_index_to_letter(
                na_constants.UNK_TOKEN_ID_MAP.get(polymer_type, 27), polymer_type
            )] * len(motif_mask)
            for idx, aatype_val in zip(motif_mask.nonzero()[0], prot.aatype):
                seq[idx] = _aatype_to_letter(aatype_val, polymer_type)
        else:
            CB = residue_constants.atom_order["CB"]
            CA = residue_constants.atom_order["CA"]

            ref_pos = np.zeros((len(motif_mask), 3))
            motif_indices = np.where(motif_mask)[0]
            ref_pos[motif_indices] = prot.atom_positions[:, CB]

            gly_in_motif = prot.aatype == residue_constants.restype_order["G"]
            ref_pos[motif_indices[gly_in_motif]] = prot.atom_positions[gly_in_motif, CA]

            seq = ['X'] * len(motif_mask)
            for idx, aatype_val in zip(motif_mask.nonzero()[0], prot.aatype):
                seq[idx] = residue_constants.restypes[aatype_val]

        motif_templates.append({
            'full_motif_mask': full_motif_mask,
            'motif_mask': motif_mask,
            'cb_pos': ref_pos,  # name kept for backward compat; holds C1' for NA
            'motif_seq': ''.join(seq)
        })
        
    if target_length is not None and target_length > len(full_motif_mask):
        print(f"[motif_utils] Padding motif templates from {len(full_motif_mask)} → {target_length}")
        pad_len = target_length - len(full_motif_mask)
        padded_templates = []
        for motif in motif_templates:
            padded = motif.copy()
            padded['full_motif_mask'] = np.concatenate(
                [motif['full_motif_mask'], np.zeros(pad_len, dtype=bool)]
            )
            padded['motif_mask'] = np.concatenate(
                [motif['motif_mask'], np.zeros(pad_len, dtype=bool)]
            )
            padded['cb_pos'] = np.concatenate(
                [motif['cb_pos'], np.zeros((pad_len, 3))], axis=0
            )
            padded['motif_seq'] = motif['motif_seq'] + 'X' * pad_len
            padded_templates.append(padded)
        motif_templates = padded_templates

    return motif_templates
        

def merge_motif_specs(specs):
    
    def recompute_total_lengths(structures):
        min_total = 0
        max_total = 0
        for s in structures:
            if s["type"] == "motif":
                L = s["end_index"] - s["start_index"] + 1
                min_total += L
                max_total += L
            elif  s["type"] == "scaffold":
                min_total += s["min_length"]
                max_total += s["max_length"]
        
        return min_total, max_total

    def merge_two(spec1, spec2):
        trailing = spec1['structures'][-1]
        leading = spec2['structures'][0]
        
        assert trailing["type"] == "scaffold" and leading["type"] == "scaffold", "only valid merge when leading/trailing are scaffold segments"
        
        min_len = max(trailing["min_length"], leading["min_length"])
        max_len = min(trailing["max_length"], leading["max_length"])
        if min_len > max_len:	# edge case
            max_len = max(trailing["max_length"], leading["max_length"])
   
        new_pad = {'type': 'scaffold','min_length': min_len, 'max_length': max_len}
        
        # getting last motif group identifier
        last_motif_group = None
        for struc in spec1['structures']:
            if struc["type"]=="motif":
                last_motif_group = struc["group"]

        # setting new motif group identifiers to last + 1
        next_motif_group = chr(ord(last_motif_group) + 1)
        spec2_structs = deepcopy(spec2["structures"])
        for struc in spec2_structs[1:]:
            if struc["type"]=="motif":
                struc["group"]=next_motif_group
    
        new_scaffold_template= spec1['structures'][:-1] + [new_pad] + spec2_structs[1:]
        
        min_total,max_total = recompute_total_lengths(new_scaffold_template)
        return {
            'name': spec1['name'] + '+' + spec2['name'],
            'structures': new_scaffold_template,
            'min_total_length': min_total,
            'max_total_length': max_total
        }
        
    merged = specs[0]
    for i in range(len(specs)-1):
        merged = merge_two(merged, specs[i+1])
    return merged


def load_motif_spec(filepath):
    """
    Load motif specification file.

    Args:
        filepath:
            Path to the PDB file for motif specification.

    Returns:
        A dictionary of motif specifications containing
            -	name:
                Name of the motif scaffolding problem
            -	structures:
                A list of dictionaries, each of which defines either 
                    -	a motif segment, containing information on the chain and 
                        residue index range that the motif structure is coming 
                        from, as well as the motif group that this segment belongs
                    -	a scaffold segment, containing information on the maximum 
                        and minimum number of residues for the segment
            -	min_total_length:
                Minimum number of residues for the generated structure
            -	max_total_length:
                Maximum number of residues for the generated structure.
    """
    with open(filepath) as file:
        structures = []
        for line in file:
            if line.startswith('REMARK 999 INPUT'):
                if line[18] == ' ':
                    structures.append({
                        'type': 'scaffold',
                        'min_length': int(line[19:23]),
                        'max_length': int(line[23:27])
                    })
                else:
                    structures.append({
                        'type': 'motif',
                        'chain': line[18],
                        'start_index': int(line[19:23]),
                        'end_index': int(line[23:27]),
                        'group': line[28] if len(line) > 28 and line[28] != ' ' else 'A'
                    })
            if line.startswith('REMARK 999 NAME'):
                name = line[18:]
            if line.startswith('REMARK 999 MINIMUM TOTAL LENGTH'):
                min_total_length = int(line[37:])
            if line.startswith('REMARK 999 MAXIMUM TOTAL LENGTH'):
                max_total_length = int(line[37:])
    return {
        'name': name,
        'structures': structures,
        'min_total_length': min_total_length,
        'max_total_length': max_total_length
    }

def sample_motif_mask(spec):
    """
    Sample a motif configuration from a dictionary of specifications.

    Args:
        spec:
            A dictionary of motif specifications containing
                -	name:
                    Name of the motif scaffolding problem
                -	structures:
                    A list of dictionaries, each of which defines either 
                        -	a motif segment, containing information on the chain and 
                            residue index range that the motif structure is coming 
                            from, as well as the motif group that this segment belongs
                        -	a scaffold segment, containing information on the maximum 
                            and minimum number of residues for the segment
                -	min_total_length:
                    Minimum number of residues for the generated structure
                -	max_total_length:
                    Maximum number of residues for the generated structure.

    Returns:
        A dictionary of masks including
            -	sequence:
                A residue-level mask to indicate which residue contains conditional 
                sequence information
            -	structure: 
                A pair residue-residue mask to indicate which pair of residues contains
                conditional structural information
            -	group:
                Residue-level group indices to indicate which group each residue belongs to 
                (0 indicates scaffold and each positive integer indicates a motif group).
    """
    success = False
    while not success:

        # Define
        total_length = 0
        motif_sequence_mask = []
        motif_groups = []

        # Generate
        for structure in spec['structures']:
            if structure['type'] == 'scaffold':
                scaffold_length = np.random.randint(structure['min_length'], structure['max_length'] + 1)
                motif_sequence_mask.extend([0] * scaffold_length)
                motif_groups.extend([0] * scaffold_length)
                total_length += scaffold_length
            else:
                motif_length = structure['end_index'] - structure['start_index'] + 1
                motif_sequence_mask.extend([1] * motif_length)
                motif_groups.extend([ord(structure['group']) - ord('A') + 1] * motif_length)
                total_length += motif_length

        # Validate
        if total_length >= spec['min_total_length'] and \
            total_length <= spec['max_total_length']:
            success = True

    # Create motif structure mask
    motif_structure_mask = np.zeros((total_length, total_length))
    num_groups = np.max(motif_groups)
    for i in range(1, 1 + num_groups):
        motif_group_sequence_mask = np.equal(motif_groups, i)
        motif_structure_mask += motif_group_sequence_mask[:, np.newaxis] * motif_group_sequence_mask[np.newaxis, :]

    return {
        'sequence': np.array(motif_sequence_mask).astype(bool),
        'structure': np.array(motif_structure_mask).astype(bool),
        'group': np.array(motif_groups).astype(int)
    }

def save_motif_pdb(spec_filepath, mask, pdb_filepath):
    """
    Save motif information as a PDB file.

    Args:
        spec_filepath:
            Path to motif specification file.
        mask:
            A residue-level mask to indicate which residue is a motif residue
        pdb_filepath:
            Output PDB filepath.
    """

    def pad_left(string, length):
        assert len(string) <= length
        return ' ' * (length - len(string)) + string

    # Parse residue index in motif spec file
    spec = load_motif_spec(spec_filepath)
    residue_index_spec = []
    for structure in spec['structures']:
        if structure['type'] == 'motif':
            for i in range(structure['start_index'], structure['end_index'] + 1):
                residue_index_spec.append((
                    structure['chain'],
                    i,
                    structure['group']
                ))

    # Parse residue index in motif pdb file
    residue_index_pdb = [i + 1 for i, elt in enumerate(mask) if elt]
    assert len(residue_index_pdb) == len(residue_index_spec)

    # Create residue index map
    residue_index_map = dict([
        (
            '{}_{}'.format(elt[0], elt[1]),
            (residue_index_pdb[i], elt[2])
        )
        for i, elt in enumerate(residue_index_spec)
    ])

    # Parse records in motif spec file
    with open(spec_filepath) as file:
        lines = [line for line in file if line.startswith('ATOM')]

    # Update residue index
    updated_lines = []
    for i, line in enumerate(lines):
        chain = line[21]
        residue_index = int(line[22:26])
        key = '{}_{}'.format(chain, residue_index)
        updated_residue_index = residue_index_map[key][0]
        updated_group = residue_index_map[key][1]
        updated_line = line[:21] + 'A' + str(updated_residue_index).rjust(4) + line[26:72] + updated_group.ljust(4) + line[76:]
        updated_lines.append(updated_line)

    # Save
    with open(pdb_filepath, 'w') as file:
        file.write(''.join(updated_lines))


def _aatype_to_letter(aatype_val, polymer_type):
    """Convert aatype integer to display letter based on polymer type."""
    if polymer_type == "rna":
        return na_constants.RNA_TOKEN_ID_TO_LETTER.get(aatype_val, "N")
    elif polymer_type == "dna":
        return na_constants.DNA_TOKEN_ID_TO_LETTER.get(aatype_val, "N")
    else:
        return residue_constants.restypes[aatype_val] if aatype_val < 20 else "X"


def _fill_na_ref_pos(prot, ref_pos, motif_indices, ref_atom_name):
    """Fill reference atom positions for NA motifs by scanning atom_positions.

    Falls back to residue_constants atom_order if the atom name exists there,
    otherwise uses the first backbone atom position as fallback.
    """
    for j, midx in enumerate(motif_indices):
        if midx < prot.atom_positions.shape[0]:
            # Try to find the atom in the standard atom_order
            if ref_atom_name in residue_constants.atom_order:
                atom_idx = residue_constants.atom_order[ref_atom_name]
                pos = prot.atom_positions[midx, atom_idx]
                if prot.atom_mask[midx, atom_idx] > 0.5:
                    ref_pos[midx] = pos
                    continue
            # Fallback: use C1' coordinate from na_ref_atoms ordering
            # or the first non-zero position
            for k in range(prot.atom_positions.shape[1]):
                if prot.atom_mask[midx, k] > 0.5:
                    ref_pos[midx] = prot.atom_positions[midx, k]
                    break
