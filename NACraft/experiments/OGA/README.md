# OGA504 dimer experiment

This directory contains the released OGA RNA-aptamer workflow. The target is
the canonical 504-residue engineered human O-GlcNAcase entity from PDB 5VVO,
used twice in every modeled complex. Coordinate coverage is incomplete
(437 and 429 resolved residues in chains A and B), but the theoretical
504-residue sequence must not be truncated.

Generated data are written to `work/` by default and are not tracked. All
scripts accept explicit input/output paths; the Slurm wrappers derive source
paths from their own location and use environment variables for cluster
configuration.

## Inputs

The canonical 5VVO structure is included under `inputs/`. Add the audited
parent tables and sequence collections, or provide equivalent paths:

```text
inputs/
|-- Human_O-GlcNAcase_5VVO.cif
|-- parent_candidates.csv
|-- denovo_sequences.json
|-- similarity_guided_sequences.json
`-- wetlab_name_mapping.csv
```

The parent table contains the 600 audited RNA parents used by this workflow:
200 de novo, 200 similarity-guided, and 200 previous-redesign candidates.
Candidate tables and wet-lab mappings are experiment data and are not bundled
with the source release.

Prepare the exact target manifest, parent manifests, FASTA files, and AF3
presearch input:

```bash
python NACraft/experiments/OGA/scripts/oga_pipeline.py prepare \
  --source-cif NACraft/experiments/OGA/inputs/Human_O-GlcNAcase_5VVO.cif \
  --parent-table NACraft/experiments/OGA/inputs/parent_candidates.csv \
  --denovo-json NACraft/experiments/OGA/inputs/denovo_sequences.json \
  --similarity-json NACraft/experiments/OGA/inputs/similarity_guided_sequences.json \
  --wetlab-mapping NACraft/experiments/OGA/inputs/wetlab_name_mapping.csv \
  --out-root NACraft/experiments/OGA/work
```

The command rejects a target that is not the OGA504 dimer and verifies the
expected parent populations before writing any production manifest.

## MSA and templates

Run AF3 presearch once for the exact 504-residue protein sequence. The
resulting `presearch/out/index.json` must contain that sequence as an exact
key. Prepare chain-specific 5VVO templates with:

```bash
python NACraft/experiments/OGA/scripts/prepare_5vvo_templates.py \
  --source-cif NACraft/experiments/OGA/work/targets/Human_O-GlcNAcase_5VVO.cif \
  --target-manifest NACraft/experiments/OGA/work/targets/target_manifest.json \
  --output-dir NACraft/experiments/OGA/work/presearch/out/5vvo_only
```

AF3 receives the same protein MSA for both OGA chains, a chain-specific 5VVO
protein template, and no RNA template.

## Cluster configuration

Create the log directory before submitting because Slurm resolves `#SBATCH`
log paths at submission time:

```bash
cd NACraft/experiments/OGA
mkdir -p logs work
export OGA_RUN_ROOT="work"
export OGA_ASSET_ROOT="work"
export NACRAFT_PYTHON="python"
export NACRAFT_BOLTZ_CACHE="../../../model_assets/boltz"
```

For AF3 jobs, additionally configure the local installation:

```bash
export AF3_CODE_DIR="../../../third_party/alphafold3"
export AF3_MODEL_DIR="../../../model_assets/af3_models"
export AF3_PRESEARCH_DIR="work/presearch/out"
export AF3_SIF_PATH="../../../model_assets/containers/alphafold3.sif"
export APPTAINER_BIN="apptainer"
```

Cluster-specific partitions, accounts, memory limits, and node exclusions
should be supplied with `sbatch` options or edited locally. The released
scripts do not encode a username, node name, Conda path, or array throttle.

## Parent refolding

```bash
sbatch --array=0-59 scripts/slurm_parent_boltz.sh
```

Each worker processes its manifest shard. The runner validates the exact
target sequence and reusable presearch entry, generates five Boltz refolds per
parent, and writes resumable `complete.json` or `failed.json` records.

## NA-MPNN redesign

```bash
sbatch --dependency=aftercorr:<BOLTZ_JOB_ID> --array=0-59 \
  scripts/slurm_nampnn.sh
```

NA-MPNN reads the saved Boltz structures, fixes both protein chains, and
redesigns only the RNA sequence. The number of workers and children is
controlled by the manifest and command options.

## AF3 validation

Parent and child populations are validated separately with five independent
AF3 seeds. MSA and template data are reused; the AF3 data pipeline is not
rerun for each candidate.

```bash
sbatch --array=0-59 \
  --export=ALL,OGA_AF3_MODE=parent,OGA_AF3_NUM_WORKERS=60 \
  scripts/slurm_af3_validate.sh

sbatch --dependency=afterok:<NAMPNN_JOB_ID> --array=0-299 \
  --export=ALL,OGA_AF3_MODE=child,OGA_AF3_NUM_WORKERS=300 \
  scripts/slurm_af3_validate.sh
```

Every replicate must contain RNA plus two 504-residue OGA chains before its
metrics are accepted. RNA-OGA chain-pair ipTM and OGA-dimer confidence are
recorded separately.

## Aggregate and rank

```bash
sbatch --dependency=afterok:<CHILD_AF3_JOB_ID> \
  scripts/slurm_aggregate_rank.sh
```

Ranking combines AF3 interface metrics, catalytic-pocket contacts, sequence
diversity, and target RMSD. Candidate tables retain source paths and seed-level
records for auditability.

## Refinement batch

The released refinement workflow combines 30 new de novo parents (10 each at
20, 40, and 60 nt) with 30 similarity-guided parents initialized from five
ranked sequences. Prepare its 60-parent manifest with:

```bash
python NACraft/experiments/OGA/scripts/prepare_refinement_batch.py \
  --target-manifest NACraft/experiments/OGA/work/targets/target_manifest.json \
  --top-selection NACraft/experiments/OGA/work/analysis/top_candidates.csv \
  --out-root NACraft/experiments/OGA/work/refinement
```

Then point both run and asset roots at the prepared locations and submit the
dependency graph:

```bash
cd NACraft/experiments/OGA
export OGA_RUN_ROOT="work/refinement"
export OGA_ASSET_ROOT="work"
bash scripts/submit_refinement.sh
```

After both AF3 arrays finish, extract the seed-level target-RMSD evidence in
24 shards. The default native structure is the released 5VVO input; override
`OGA_NATIVE_STRUCTURE` only when validating against another reference.

```bash
extract_id=$(sbatch --parsable --array=0-23 \
  --export=ALL,OGA_RUN_ROOT="${OGA_RUN_ROOT}",OGA_ASSET_ROOT="${OGA_ASSET_ROOT}" \
  scripts/slurm_refinement_target_rmsd_extract.sh)
```

To rank the refinement candidates together with an earlier campaign, point
`OGA_BASE_EVIDENCE` to that campaign's
`oga_target_rmsd_seed_evidence.csv` and submit the combined rank job after all
shards complete:

```bash
export OGA_BASE_EVIDENCE="work/analysis/oga_target_rmsd_seed_evidence.csv"
sbatch --dependency="afterok:${extract_id}" \
  --export=ALL,OGA_RUN_ROOT="${OGA_RUN_ROOT}",OGA_ASSET_ROOT="${OGA_ASSET_ROOT}",OGA_BASE_EVIDENCE="${OGA_BASE_EVIDENCE}" \
  scripts/slurm_refinement_combined_rank.sh
```

`OGA_COMBINED_ANALYSIS_ROOT`, `OGA_RMSD_SHARDS`, and
`OGA_CANDIDATE_PREFIX` optionally override the output location, shard count,
and namespace used to distinguish refinement candidates.

The generic refinement wrappers replace the former node-specific A100, A800,
and 4090 scripts. Resource overrides belong in `sbatch` arguments or local
copies, not in the released workflow.

## Validation

These checks do not run model inference:

```bash
python -m unittest NACraft/experiments/OGA/tests/test_oga_pipeline.py
python -m py_compile NACraft/experiments/OGA/scripts/*.py
bash -n NACraft/experiments/OGA/scripts/slurm_common.sh \
  NACraft/experiments/OGA/scripts/slurm_parent_boltz.sh \
  NACraft/experiments/OGA/scripts/slurm_nampnn.sh \
  NACraft/experiments/OGA/scripts/slurm_af3_validate.sh \
  NACraft/experiments/OGA/scripts/slurm_aggregate_rank.sh
```

Additional scientific details are retained in
[`docs/experiment_methods.md`](docs/experiment_methods.md). Generated result
reports are kept in the external experiment workspace and are not versioned
in this release.
