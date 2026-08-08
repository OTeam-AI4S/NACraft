# Metrics

## Primary Metrics

- AF3 ipTM.
- AF3 aptamer-chain pLDDT.

Report best-of-400 and top-10 mean per target.

Confidence success:

```text
ipTM > 0.5 and aptamer pLDDT > 70
```

## Secondary Metrics

- AF3 iPAE/interface PAE, lower is better.
- Target C-alpha RMSD between the AF3-refold target and its input structure.
- Protein-aligned nucleic-acid phosphate RMSD.

Candidate-level result tables save the target metric as `target_RMSD`, with
matched C-alpha count, sequence identity, chain pairing and any calculation
error in adjacent provenance columns. Group summaries report its mean, median
and maximum.

Geometry success:

```text
protein-aligned NA phosphate RMSD < 5 A
```

## Ranking

Candidate ranking uses AF3 confidence only:

```text
score = ipTM + 0.01 * pLDDT_aptamer - 0.05 * iPAE
```

If iPAE is missing:

```text
score = ipTM + 0.01 * pLDDT_aptamer
```

Target RMSD may be applied as a hard structural-validity gate before ranking.
Neither target RMSD nor nucleic-acid RMSD is added to the confidence ranking
score.

## RMSD Implementation Note

`target_RMSD` pairs protein residues by local sequence alignment and computes
one global Kabsch superposition over matched C-alpha atoms. Manifest chain and
residue-range annotations define the input target region. The production RMSD
path should run in an environment with numpy, enabling Kabsch alignment of predicted protein atoms to native protein atoms before evaluating nucleic-acid representative atoms.
When numpy is unavailable, the lightweight helper falls back to centroid translation for smoke tests and protocol validation.

## Conformation-Selective Metrics

For paired-state NACraft designs, score the same candidate sequence against the
positive and negative target states with the same AF3 predict-only protocol.

Primary selectivity margin:

```text
Delta ipTM = ipTM_positive - ipTM_negative
```

Positive and negative AF3 contexts also save `positive_target_RMSD`,
`negative_target_RMSD` and `max_target_RMSD`; both target structures should be
valid before interpreting the selectivity margin.

Secondary contact margin, when predicted structures are parsed consistently:

```text
Delta C = C_positive - C_negative
```

Suggested computational success rule:

```text
ipTM_positive > 0.5
aptamer pLDDT_positive > 70
Delta ipTM > 0.15
ipTM_negative < 0.5
```

These criteria are used for triage only and should not be phrased as measured
binding specificity.
