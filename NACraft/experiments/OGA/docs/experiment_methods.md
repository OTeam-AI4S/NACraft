# OGA target-correction experiment methods

## Target definition

The protein target is defined from
`Human O-GlcNAcase 5VVO.cif`. The canonical polymer sequence is read directly
from `_entity_poly.pdbx_seq_one_letter_code_can`. It contains 504 amino acids
and occurs twice in the biological assembly. The two theoretical protein
sequences are identical even though the deposited coordinates resolve 437
residues in chain A and 429 residues in chain B.

Every predicted complex therefore contains two complete 504-residue OGA
proteins and one independently supplied RNA. Missing crystallographic
coordinates are treated as unresolved template positions, not as sequence
deletions. The A3 aptamer is used only as provenance for the A3-guided RNA
population and is not included as a target entity.

## Parent population

The corrected parent manifest was reconstructed from the normalized wet-lab
report and cross-checked against the original de novo and A3-guided sequence
JSON files. It contains exactly 600 stable records: 200 de novo, 200
A3-guided, and 200 previous-redesign parents. Sequence lengths are calculated
from the RNA strings rather than trusted from historical length columns.
Every record includes its provenance, sequence hash, and historical metrics
marked as originating from the invalid monomer target.

## Protein presearch

The AlphaFold 3 data pipeline was run once for the exact 504-residue protein
sequence. The resulting cache contains paired and unpaired MSAs and four
template records. Lookup is by exact full sequence; both identical protein
copies reuse this one entry. The corrected configuration has no fallback to
the historical 437-residue cache.

## Boltz parent refolding

Each parent RNA is predicted with two fixed-sequence OGA504 protein entities.
Five Boltz structures are generated per parent and stored as both PDB and
mmCIF, together with the per-sample confidence arrays. A completion record
contains the RNA sequence, protein sequence hash, copy count, sample count,
metrics, and elapsed time. An existing completion record is a safe-resume
sentinel.

These parent structures are the backbone source for subsequent RNA-only
NA-MPNN redesign. Backbone selection, five-child redesign, corrected AF3
validation, joint ranking, and wet-lab comparison follow the approved design
specification and are recorded separately as those stages execute.

## Independent AF3 replication

AF3 validation uses five separate predictor invocations with seeds 101, 211,
307, 401, and 503. Each invocation writes one independently sampled structure
and metric record. Before acceptance, the runner verifies that the structure
contains chains with residue counts RNA-length, 504, and 504. Results produced
by the historical parser behavior that copied one top-ranked model into five
array positions are marked `legacy_tiled` and excluded from aggregation.

For the previous wet-lab 20 comparison, corrected RNA–OGA chain-pair ipTM is
compared with historical monomer-complex ipTM. Corrected global ipTM is
retained but explicitly marked non-comparable because it also contains the
high-confidence OGA–OGA dimer interface.

## Reproducibility and storage

Scripts and manuscript-facing methods are version controlled. Raw MSAs,
templates copied from predictor output, predicted structures, serialized
model tensors, metrics, and scheduler logs remain under
`NACraft/experiments/OGA/work/` by default and are not committed to Git.
Target and sequence hashes in the operational manifests link the external
artifacts to this experiment definition.
