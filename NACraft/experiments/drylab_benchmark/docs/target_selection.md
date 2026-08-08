# Target Selection

The primary benchmark uses post-2023 protein-RNA and protein-DNA complexes from the PDB.
The target manifest should record:

- `target_id`
- `release_date`
- `polymer_type`
- `na_length`
- `protein_chains`
- `na_chains`
- `structure_path`
- `native_na_sequence`, when available for similarity-guided design

## Eligibility Constraints

The canonical NA-20 target set is selected from filtered RNA and DNA candidate
manifests using these constraints:

- exclude ODesign-visible RNA/DNA targets from the primary benchmark
- require one protein chain and one nucleic-acid chain
- require an unambiguous native RNA or DNA sequence
- require target-protein length <=400 amino acids
- require global uniqueness of `target_id`
- require global uniqueness of `protein_sequence`
- avoid oversized structures in the generated set

## Length Bins

RNA:

- 10-30 nt: 2 targets
- 31-50 nt: 2 targets
- 51-75 nt: 2 targets
- 76-100 nt: 2 targets
- 101-150 nt: 2 targets

DNA:

- 5-20 nt: 2 targets
- 21-40 nt: 2 targets
- 41-60 nt: 2 targets
- 61-80 nt: 2 targets
- 81-100 nt: 2 targets

If a bin has insufficient eligible targets, fill from adjacent lengths and record `selection_reason`.

The current NA-20 manifest has 20 targets, 20 unique target-protein sequences,
no multi-chain protein sequence concatenation, maximum target-protein length 395
aa and maximum downloaded PDB size below 1 MB.

## Hotspots

Hotspots are computed from the native complex, but native nucleic-acid coordinates are not passed to design.
The default rule is:

1. Compute all protein residues with any heavy atom within 5 A of any native nucleic-acid heavy atom.
2. Sort residues by contact count, then by minimum nucleic-acid distance.
3. Select the top four residues as site-guided hotspots.
4. Store both the full native interface patch and the selected hotspot list.

## Paired-State Targets

Conformation-selective design uses a separate paired-state manifest rather than
the NA-20 single-target manifest. The manifest should distinguish:

- target-selective pairs, where positive and negative contexts share
  the same target sequence but differ in conformation;
- paralog-specific pairs, such as EGFR versus HER2, where targets are related
  but have different sequences;
- PTM-specific pairs, such as phosphohistidine-like versus unmodified target
  contexts.

The first target-selective case should use EGFR extracellular-domain
holo-like/active-like and apo/inactive-like structures. EGFR-versus-HER2 and
phosphohistidine-like target-state discrimination can be included as secondary
specificity cases, but they should not be labelled as holo/apo conformation
selection.
