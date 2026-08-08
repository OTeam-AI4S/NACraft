# Experiment workflows

The experiment code is kept beside the core implementation while large input
datasets and generated results remain outside Git.

| Directory | Purpose |
|---|---|
| `drylab_benchmark/` | released target structures and configs, target discovery, Slurm submission, AF3 collection, and benchmark metrics |
| `OGA/` | corrected OGA504 dimer design, NA-MPNN redesign, AF3 validation, and ranking |

Each workflow has its own README. By default, generated data are written below
the repository-relative `data/` or experiment-local `work/` directories;
environment variables and command-line options can redirect them on a cluster.
Production model inference is submitted with Slurm, whereas plotting and
tabulation scripts run directly in the configured analysis environment.
Manuscript-only dry-lab analysis and rendered figures are maintained outside
this release.

The released target structures and exact design YAML files are versioned
under `drylab_benchmark/targets/` and `drylab_benchmark/configs/`. AF3
databases, MSA/template caches, model checkpoints and generated candidates are
not versioned.

- [`drylab_benchmark/README.md`](drylab_benchmark/README.md)
- [`drylab_benchmark/presearch_bundle/README.md`](drylab_benchmark/presearch_bundle/README.md)
- [`OGA/README.md`](OGA/README.md)
