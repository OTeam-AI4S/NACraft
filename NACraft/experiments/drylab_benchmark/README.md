# NACraft dry-lab benchmark

This package contains reusable code for constructing protein-binding RNA/DNA
benchmarks, generating NACraft and ODesign inputs, submitting Slurm jobs,
collecting AF3 outputs, and computing candidate-level metrics. Large datasets
and generated results are stored outside Git.

## Released inputs

```text
configs/
|-- na12/de_novo/              12 de novo design configurations
|-- na12/similarity_guided/    12 similarity-guided configurations
|-- protein_antigen/           5 targets at 20, 30, 40 and 50 nt
|-- target_selective/          EGFR-versus-HER2 RNA/DNA configurations
`-- templates/                 config-generation template
targets/
|-- na12/                      12 native complexes and target manifest
|-- protein_antigen/           5 protein structures, sequences and hotspots
`-- target_selective/          EGFR and HER2 structures and manifest
```

All host paths stored in the released manifests are repository-relative.
Model assets, AF3 databases, presearch outputs and generated candidates remain
untracked.

## Released experiment blocks

### NA-12

NA-12 contains six RNA-binding and six DNA-binding protein targets. The current
analysis compares de novo and similarity-guided NACraft designs and uses
ODesign as an external generation baseline.

For each target and NACraft mode:

- 100 sequences are optimized directly;
- two NA-MPNN variants are generated per optimized parent;
- all 300 sequences are independently validated with AF3.

Historical analysis functions retain a few `na20` identifiers for compatibility
with the source tables from which NA-12 was selected. The released design
configs and target manifest contain only the final 12 targets.

### Target-selective design

The retained target-selective experiment discriminates EGFR domain III
(`1YY9`) from HER2 domain III (`1N8Z`) for RNA and DNA aptamers of 30, 40, and
50 nt. Each candidate is validated against both proteins. A candidate is
selective when:

```text
HER2 ipTM < 0.5
EGFR ipTM > 0.5
EGFR ipTM > HER2 ipTM + 0.1
```

The two earlier EGFR paired-context cases are not part of the released
analysis because their optimization inputs did not distinguish the contexts.

### Protein targets

The focused RNA benchmark covers B7-H3, PD-L1, CD3delta, TNFR1, and FGFR2 at
20, 30, 40, and 50 nt. Internal `antibody` identifiers are retained only for
compatibility with collected tables; the experimental targets are proteins,
not antibodies.

### Ablations

- similarity-loss weights: 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0;
- optimized parent versus NA-MPNN redesign;
- full optimization versus early-stop trajectories.

The released benchmark does not include random-mutant, seed-only,
anti-bind-only, or AF3-gradient baselines.

## Metrics

AF3 is the common independent validator. The primary metric is ipTM; secondary
metrics include aptamer/interface pLDDT, iPAE, intended-site contacts,
protein-aligned nucleic-acid RMSD, and target-protein RMSD. Benchmark summaries
retain every parse-valid candidate. Fixed ipTM thresholds are reported as
success rates, while joint filters are reserved for downstream candidate
triage.

## Pipeline modules

```text
scripts/query_pdb_post2023_na_targets.py  query candidate PDB entries
scripts/build_target_manifest.py          build a structure manifest
scripts/filter_manifest_candidates.py     enforce target/QC constraints
scripts/select_na20_targets.py            construct the original discovery pool
scripts/compute_hotspots.py                derive native interface patches
scripts/generate_nacraft_configs.py        render NACraft YAML files
scripts/build_odesign_inputs.py            render ODesign inputs
scripts/submit_nacraft_designs.py          generate NACraft command lists
scripts/submit_odesign_runs.py             generate ODesign command lists
scripts/run_af3_predict_only.py             validate one candidate with AF3
scripts/collect_af3_metrics*.py            collect confidence and RMSD metrics
```

The retained NA-12 filtering and source-data tables are handled by the
benchmark workflow. Manuscript statistics and figures are generated in the
separate external dry-lab analysis workspace and are not part of this release.

## Example construction commands

```bash
python NACraft/experiments/drylab_benchmark/scripts/build_target_manifest.py \
  --structures-dir data/drylab_benchmark/structures \
  --output data/drylab_benchmark/processed/target_manifest.csv

python NACraft/experiments/drylab_benchmark/scripts/compute_hotspots.py \
  --manifest data/drylab_benchmark/processed/target_manifest.csv \
  --output-manifest data/drylab_benchmark/processed/target_manifest_hotspots.csv \
  --hotspot-dir data/drylab_benchmark/hotspots

python NACraft/experiments/drylab_benchmark/scripts/generate_nacraft_configs.py \
  --manifest data/drylab_benchmark/processed/target_manifest_hotspots.csv \
  --output-dir data/drylab_benchmark/configs/nacraft \
  --clean-output
```

Production inference must be submitted through Slurm. Array builders in this
package intentionally do not impose a task throttle; configure partitions,
resource requests, and node exclusions for the target cluster.

For example, run one released NA-12 configuration from the repository root:

```bash
python NACraft/main.py \
  --config NACraft/experiments/drylab_benchmark/configs/na12/de_novo/8X0N.yaml \
  --num_designs 1 \
  --ligandmpnn_seqs 2 \
  --outpath outputs/na12/8X0N
```

## Tests

```bash
python -m unittest discover -s NACraft/experiments/drylab_benchmark/tests
```

The tests cover manifest filtering, hotspot ranking, config generation, AF3
metric parsing, target RMSD, and command generation. They do not run GPU model
inference.
