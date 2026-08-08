# ODesign Comparison

Only the nucleic-acid design parts of ODesign are used for comparison.

Protocol points matched by NACraft:

- post-2023 protein-RNA and protein-DNA targets;
- native nucleic-acid binder removed from the design input;
- nucleic-acid length equals native binder length;
- 400 candidates per target;
- AF3 predict-only validation;
- ODesign-style RMSD threshold retained as a secondary metric.

Important differences:

- NACraft primary ranking uses AF3 ipTM and aptamer pLDDT, optionally penalized by iPAE.
- RMSD is not used for ranking because it requires the native nucleic-acid geometry.
- NACraft reports both de novo and similarity-guided design modes.
- Conformation-selective, paralog-specific and PTM-specific paired-state designs are NACraft-only follow-up experiments because ODesign does not expose an equivalent positive/negative context objective.
- NACraft ablations test the contribution of bind loss and polymer-specific handling.

Visible ODesign targets are treated as a secondary direct-comparison subset:

- RNA: `7WKP`, `7YEW`, `7YGL`, `9C7A`, `8TG4`
- DNA: `7XVN`, `7YSF`, `7YUK`, `8PMF`
