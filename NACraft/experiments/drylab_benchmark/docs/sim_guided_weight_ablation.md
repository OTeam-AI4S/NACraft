# Similarity-guided loss-weight ablation

This experiment replaces the earlier de novo versus similarity-guided ablation
as the dedicated analysis of the sequence-similarity prior.

## Scope

- Benchmark set: the random/NA-20 dry-lab benchmark targets.
- Design mode: NACraft similarity-guided design.
- Ablated term: `SequenceSimilarityLoss.strength`.
- Weights: `0.0, 0.2, 0.4, 0.6, 0.8, 1.0`.
- Per target and per weight:
  - 10 optimized NACraft sequences;
  - 4 NA-MPNN redesigns per optimized sequence;
  - 50 total candidates for downstream AF3 validation.

## Objective

For each target, NACraft optimizes a binding/contact objective while applying a
variable sequence-similarity prior:

```yaml
losses:
  - type: LigandContactLoss
    state: 0
  - type: SequenceSimilarityLoss
    target_sequence: <native_na_sequence or seed_sequence>
    strength: <ablation weight>
```

`LigandContactLoss` uses its default strength of `1.0`. The ablated
`SequenceSimilarityLoss` strength is the only changed coefficient.

## Output layout

Configs are written to:

```text
data/drylab_benchmark/configs/nacraft_sim_guided_weight_ablation/
```

Candidate outputs are written to:

```text
data/drylab_benchmark/candidates/nacraft_sim_guided_weight_ablation/
```

Each weight uses a distinct directory name: `sim_guided_w000`,
`sim_guided_w020`, `sim_guided_w040`, `sim_guided_w060`,
`sim_guided_w080` and `sim_guided_w100` to avoid collisions.

## Execution

Generate configurations with
`scripts/generate_sim_guided_weight_ablation_configs.py`, then submit the
resulting command list through
`slurm/submit_sim_guided_weight_ablation_array.sh`. Commands use
`--skip_existing`, so interrupted arrays can be resubmitted without replacing
completed designs. Cluster-specific partitions, accounts and node constraints
must be supplied locally; the released workflow does not set an array
throttle.
