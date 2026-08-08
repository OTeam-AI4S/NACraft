# Dry-lab benchmark specification

## Analysis units

The target is the primary statistical unit. Candidate-level distributions are
used for diversity, threshold success, and best-of-N analyses, but paired
method comparisons are first summarized per target or target-length cell.

## NA-12

- Six RNA and six DNA protein-binding targets.
- De novo and similarity-guided NACraft modes.
- ODesign comparison where matched AF3-rescored outputs are available.
- 100 optimized parents and two NA-MPNN children per parent and mode.
- 300 AF3-validated candidates per target-mode cell.

## Target-selective benchmark

- Positive target: EGFR domain III (`1YY9`).
- Alternative target: HER2 domain III (`1N8Z`).
- RNA and DNA at 30, 40, and 50 nt.
- `LigandContactLoss` for EGFR and `AntiLigandContactLoss` for HER2.
- AF3 validation against both targets.

Selectivity is reported as paired confidence, not inferred from a single
complex. The success region requires HER2 ipTM < 0.5, EGFR ipTM > 0.5, and an
EGFR-HER2 ipTM margin greater than 0.1.

## Protein-antigen benchmark

- B7-H3, PD-L1, CD3delta, TNFR1, and FGFR2.
- RNA lengths of 20, 30, 40, and 50 nt.
- 100 optimized parents plus two NA-MPNN children per parent.

## Similarity-weight ablation

- Weights: 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0.
- Ten optimized parents and four NA-MPNN children per parent.
- Manuscript analysis is restricted to NA-12 and recomputed after filtering.

## Validation

AF3 is the common validator for NACraft and ODesign. Each candidate uses the
same target-protein MSA/template cache. Designed RNA/DNA chains do not receive
an evolutionary MSA. The collected record includes structure paths, ipTM,
pLDDT, iPAE, and target RMSD where an input target structure is available.

Boltz-1 supplies NACraft's optimization gradients and is therefore not used as
the independent final benchmark model.
