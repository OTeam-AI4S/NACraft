"""Nucleic acid constants for NACraft.

Defines NA-specific constants that complement residue_constants.py (protein).
All token indices align with Boltz-1's unified 33-token vocabulary.
"""

import numpy as np

# ============================================================================
# Boltz-1 Unified Token Vocabulary (33 tokens)
# ============================================================================
# Full token list for reference:
#   0: <pad>, 1: -
#   2-21: Protein AA (ALA..VAL), 22: UNK
#   23-26: RNA (A,G,C,U), 27: N (unknown RNA)
#   28-31: DNA (DA,DG,DC,DT), 32: DN (unknown DNA)
NUM_TOKENS = 33

# ============================================================================
# NA Base Types
# ============================================================================
RNA_BASES = ["A", "G", "C", "U"]
DNA_BASES = ["DA", "DG", "DC", "DT"]

RNA_BASE_LETTERS = ["A", "G", "C", "U"]  # 1-letter for sequence display
DNA_BASE_LETTERS = ["A", "G", "C", "T"]  # 1-letter for sequence display

# ============================================================================
# Token Index Mappings
# ============================================================================
# RNA base → Boltz token index
RNA_TOKEN_IDS = {"A": 23, "G": 24, "C": 25, "U": 26}
RNA_UNKNOWN_TOKEN_ID = 27  # "N"

# DNA base → Boltz token index
DNA_TOKEN_IDS = {"DA": 28, "DG": 29, "DC": 30, "DT": 31}
DNA_UNKNOWN_TOKEN_ID = 32  # "DN"

# DNA letter → Boltz token index (user-facing: A→DA, G→DG, C→DC, T→DT)
DNA_LETTER_TO_TOKEN_ID = {"A": 28, "G": 29, "C": 30, "T": 31}

# Reverse: Boltz token index → display letter
RNA_TOKEN_ID_TO_LETTER = {23: "A", 24: "G", 25: "C", 26: "U", 27: "N"}
DNA_TOKEN_ID_TO_LETTER = {28: "A", 29: "G", 30: "C", 31: "T", 32: "N"}

# ============================================================================
# Active Token Ranges (which token indices are valid for each modality)
# ============================================================================
RNA_ACTIVE_INDICES = list(range(23, 27))   # [23, 24, 25, 26] = A, G, C, U
DNA_ACTIVE_INDICES = list(range(28, 32))   # [28, 29, 30, 31] = DA, DG, DC, DT

# Protein active indices (for reference)
PROTEIN_ACTIVE_INDICES = list(range(2, 22))  # 20 standard AA

# ============================================================================
# Invalid Token Indices (to be suppressed during design)
# ============================================================================
_all_indices = set(range(33))

PROTEIN_INVALID_INDICES = sorted(
    _all_indices - set(PROTEIN_ACTIVE_INDICES) - {1}  # keep gap for protein
)
# Original protein invalid: [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
# Note: index 6 (CYS) was also suppressed in original SwitchCraft. We preserve that.
PROTEIN_INVALID_INDICES = [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

RNA_INVALID_INDICES = sorted(_all_indices - set(RNA_ACTIVE_INDICES))
DNA_INVALID_INDICES = sorted(_all_indices - set(DNA_ACTIVE_INDICES))

# ============================================================================
# Molecular Type Values (Boltz convention)
# ============================================================================
MOL_TYPE_PROTEIN = 0
MOL_TYPE_DNA = 1
MOL_TYPE_RNA = 2
MOL_TYPE_NONPOLYMER = 3

MOL_TYPE_MAP = {
    "protein": MOL_TYPE_PROTEIN,
    "dna": MOL_TYPE_DNA,
    "rna": MOL_TYPE_RNA,
}

# Polymer key for Boltz YAML batch construction
POLYMER_KEY_MAP = {
    "protein": "protein",
    "dna": "dna",
    "rna": "rna",
}

# Unknown token ID per modality
UNK_TOKEN_ID_MAP = {
    "protein": 22,  # UNK
    "rna": 27,      # N
    "dna": 32,      # DN
}

# ============================================================================
# Reference Atoms (from Boltz const.py)
# ============================================================================
# C1' is the center atom for all nucleotides (used in distance matrices)
NA_CENTER_ATOM = "C1'"

# RNA frame: (P, C4', C1') triplet for base orientation
NA_FRAME_ATOMS = ("P", "C4'", "C1'")

# Reference atoms per NA residue type (from Boltz const.py ref_atoms)
NA_REF_ATOMS = {
    # RNA
    "A": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
          "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6",
          "N1", "C2", "N3", "C4"],
    "G": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
          "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6",
          "N1", "C2", "N2", "N3", "C4"],
    "C": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
          "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4",
          "C5", "C6"],
    "U": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
          "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4",
          "C5", "C6"],
    "N": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
          "C2'", "O2'", "C1'"],
    # DNA
    "DA": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
           "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1",
           "C2", "N3", "C4"],
    "DG": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
           "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1",
           "C2", "N2", "N3", "C4"],
    "DC": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
           "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "DT": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
           "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
    "DN": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
           "C2'", "C1'"],
}

# Backbone atoms common to all nucleotides
NA_BACKBONE_ATOMS = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
                     "C3'", "O3'", "C2'", "C1'"]

# Sugar atoms (ribose/deoxyribose)
NA_SUGAR_ATOMS = ["C5'", "C4'", "O4'", "C3'", "C2'", "C1'"]

# RNA has additional O2' (2'-hydroxyl)
RNA_SUGAR_ATOMS = NA_SUGAR_ATOMS + ["O2'"]

# ============================================================================
# NA Residue Name Recognition (for PDB parsing)
# ============================================================================
RNA_RESIDUE_NAMES = {"A", "G", "C", "U"}  # 1-letter in PDB
DNA_RESIDUE_NAMES = {"DA", "DG", "DC", "DT"}  # 2-letter in PDB
NA_RESIDUE_NAMES = RNA_RESIDUE_NAMES | DNA_RESIDUE_NAMES

# Map PDB residue name → Boltz token index
NA_RESNAME_TO_TOKEN_ID = {}
for name in RNA_RESIDUE_NAMES:
    NA_RESNAME_TO_TOKEN_ID[name] = RNA_TOKEN_IDS[name]
for name in DNA_RESIDUE_NAMES:
    NA_RESNAME_TO_TOKEN_ID[name] = DNA_TOKEN_IDS[name]

# Map PDB residue name → display letter
NA_RESNAME_TO_LETTER = {
    "A": "A", "G": "G", "C": "C", "U": "U",
    "DA": "A", "DG": "G", "DC": "C", "DT": "T",
}

# ============================================================================
# NA Torsion Angle Definitions
# ============================================================================
# Backbone torsion angles for nucleic acids (defined by 4 consecutive atoms)
# alpha: O3'[i-1] - P - O5' - C5'
# beta:  P - O5' - C5' - C4'
# gamma: O5' - C5' - C4' - C3'
# delta: C5' - C4' - C3' - O3'
# epsilon: C4' - C3' - O3' - P[i+1]
# zeta:  C3' - O3' - P[i+1] - O5'[i+1]
# chi:   glycosidic bond torsion (base-specific atoms)

NA_BACKBONE_TORSIONS = {
    "alpha":   ["O3'", "P", "O5'", "C5'"],       # involves prev residue's O3'
    "beta":    ["P", "O5'", "C5'", "C4'"],
    "gamma":   ["O5'", "C5'", "C4'", "C3'"],
    "delta":   ["C5'", "C4'", "C3'", "O3'"],
    "epsilon": ["C4'", "C3'", "O3'", "P"],         # involves next residue's P
    "zeta":    ["C3'", "O3'", "P", "O5'"],         # involves next residue's P, O5'
}

# Glycosidic bond chi angle (different atoms for purines vs pyrimidines)
NA_CHI_TORSIONS = {
    # Purines (A, G, DA, DG): O4' - C1' - N9 - C4
    "A":  ["O4'", "C1'", "N9", "C4"],
    "G":  ["O4'", "C1'", "N9", "C4"],
    "DA": ["O4'", "C1'", "N9", "C4"],
    "DG": ["O4'", "C1'", "N9", "C4"],
    # Pyrimidines (C, U, DC, DT): O4' - C1' - N1 - C2
    "C":  ["O4'", "C1'", "N1", "C2"],
    "U":  ["O4'", "C1'", "N1", "C2"],
    "DC": ["O4'", "C1'", "N1", "C2"],
    "DT": ["O4'", "C1'", "N1", "C2"],
}

# ============================================================================
# NA-MPNN Encoding Mapping
# ============================================================================
# NA-MPNN uses non-standard one-letter codes for nucleic acids
# RNA: b=A, d=C, h=G, u=U
# DNA: a=DA, c=DC, g=DG, t=DT
NA_MPNN_RNA_MAP = {"A": "b", "C": "d", "G": "h", "U": "u"}  # Boltz name → NA-MPNN letter
# NA-MPNN may also output DNA-style lowercase (a,c,g,t) for RNA when na_shared_tokens=0
NA_MPNN_RNA_REVERSE = {"b": "A", "d": "C", "h": "G", "u": "U", "a": "A", "c": "C", "g": "G", "t": "U"}
NA_MPNN_DNA_MAP = {"A": "a", "C": "c", "G": "g", "T": "t"}  # 1-letter → NA-MPNN letter

NA_MPNN_DNA_REVERSE = {v: k for k, v in NA_MPNN_DNA_MAP.items()}

# ============================================================================
# Watson-Crick Base Pairing
# ============================================================================
RNA_WC_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
DNA_WC_PAIRS = {("DA", "DT"), ("DT", "DA"), ("DG", "DC"), ("DC", "DG")}

# ============================================================================
# NA Distance/Geometry Constants
# ============================================================================
# Typical inter-nucleotide distances (Angstroms)
NA_PHOSPHATE_INTER_DISTANCE = 6.7    # P[i] to P[i+1] along backbone
NA_C1P_INTER_DISTANCE = 5.4          # C1'[i] to C1'[i+1] typical
NA_BASE_PAIR_DISTANCE = 10.4         # C1'[i] to C1'[j] for WC pair
NA_BASE_STACKING_DISTANCE = 3.4      # vertical distance between stacked bases

# ============================================================================
# Helper Functions
# ============================================================================
def get_invalid_indices(polymer_type: str) -> list:
    """Get the list of invalid token indices for the given polymer type."""
    if polymer_type == "rna":
        return RNA_INVALID_INDICES
    elif polymer_type == "dna":
        return DNA_INVALID_INDICES
    else:
        return PROTEIN_INVALID_INDICES

def get_active_indices(polymer_type: str) -> list:
    """Get the list of active token indices for the given polymer type."""
    if polymer_type == "rna":
        return RNA_ACTIVE_INDICES
    elif polymer_type == "dna":
        return DNA_ACTIVE_INDICES
    else:
        return PROTEIN_ACTIVE_INDICES

def get_mol_type(polymer_type: str) -> int:
    """Get the Boltz mol_type integer for the given polymer type."""
    return MOL_TYPE_MAP[polymer_type]

def get_polymer_key(polymer_type: str) -> str:
    """Get the Boltz YAML polymer key for batch construction."""
    return POLYMER_KEY_MAP[polymer_type]

def get_unk_token_id(polymer_type: str) -> int:
    """Get the unknown token ID for the given polymer type."""
    return UNK_TOKEN_ID_MAP[polymer_type]

def seq_to_boltz_indices(seq: str, polymer_type: str) -> list:
    """Convert a sequence string to Boltz token indices.

    Args:
        seq: Sequence string (e.g., "ACGU" for RNA, "ACGT" for DNA)
        polymer_type: "rna" or "dna"

    Returns:
        List of Boltz token indices
    """
    if polymer_type == "rna":
        mapping = {c: RNA_TOKEN_IDS[c] for c in "AGCU"}
        mapping["N"] = RNA_UNKNOWN_TOKEN_ID
    elif polymer_type == "dna":
        mapping = {"A": 28, "G": 29, "C": 30, "T": 31}
        mapping["N"] = DNA_UNKNOWN_TOKEN_ID
    else:
        raise ValueError(f"Use residue_constants for protein, got {polymer_type}")

    return [mapping.get(c, mapping.get("N")) for c in seq.upper()]

def boltz_index_to_letter(idx: int, polymer_type: str) -> str:
    """Convert a Boltz token index to a display letter."""
    if polymer_type == "rna":
        return RNA_TOKEN_ID_TO_LETTER.get(idx, "N")
    elif polymer_type == "dna":
        return DNA_TOKEN_ID_TO_LETTER.get(idx, "N")
    else:
        return "X"
