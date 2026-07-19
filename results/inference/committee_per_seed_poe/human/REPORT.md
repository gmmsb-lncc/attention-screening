# Per-seed PoE committee — corpus `human`

**Protocol**: PoE = geometric mean of 4 calibrated probs per seed; thr = geometric mean of 4 thresholds per seed. Dedupe by `(seq_id, chembl_id)`. Aggregated mean ± σ over 5 canonical seeds {42, 123, 456, 789, 1024}.

| system | MCC | AUROC | F1 | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| dtkinase | 0.4484 ± 0.0127 | 0.8053 ± 0.0086 | 0.6174 ± 0.0190 | 0.7626 ± 0.0041 | 0.6509 ± 0.0196 | 0.5898 ± 0.0467 |
| drugban | 0.4866 ± 0.0144 | 0.8369 ± 0.0084 | 0.6642 ± 0.0094 | 0.7627 ± 0.0099 | 0.6170 ± 0.0187 | 0.7205 ± 0.0276 |
| graphban | 0.4669 ± 0.0211 | 0.8256 ± 0.0127 | 0.6527 ± 0.0142 | 0.7521 ± 0.0092 | 0.6004 ± 0.0136 | 0.7156 ± 0.0263 |
| conplex | 0.4305 ± 0.0227 | 0.8076 ± 0.0127 | 0.5673 ± 0.0102 | 0.7637 ± 0.0107 | 0.7093 ± 0.0579 | 0.4762 ± 0.0351 |
| **Committee PoE** | 0.5323 ± 0.0092 | 0.8613 ± 0.0063 | 0.6696 ± 0.0094 | 0.8007 ± 0.0049 | 0.7288 ± 0.0238 | 0.6206 ± 0.0278 |
