# Per-seed PoE committee — corpus `all`

**Protocol**: PoE = geometric mean of 4 calibrated probs per seed; thr = geometric mean of 4 thresholds per seed. Dedupe by `(seq_id, chembl_id)`. Aggregated mean ± σ over 5 canonical seeds {42, 123, 456, 789, 1024}.

| system | MCC | AUROC | F1 | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| dtkinase | 0.4350 ± 0.0162 | 0.7979 ± 0.0079 | 0.6228 ± 0.0121 | 0.7490 ± 0.0071 | 0.6282 ± 0.0116 | 0.6178 ± 0.0190 |
| drugban | 0.4745 ± 0.0093 | 0.8328 ± 0.0053 | 0.6597 ± 0.0084 | 0.7567 ± 0.0070 | 0.6225 ± 0.0172 | 0.7032 ± 0.0355 |
| graphban | 0.4733 ± 0.0240 | 0.8315 ± 0.0105 | 0.6658 ± 0.0130 | 0.7442 ± 0.0172 | 0.5940 ± 0.0250 | 0.7585 ± 0.0167 |
| conplex | 0.4278 ± 0.0261 | 0.8032 ± 0.0132 | 0.5601 ± 0.0073 | 0.7591 ± 0.0116 | 0.7283 ± 0.0527 | 0.4572 ± 0.0245 |
| **Committee PoE** | 0.5412 ± 0.0080 | 0.8611 ± 0.0028 | 0.6791 ± 0.0107 | 0.8014 ± 0.0028 | 0.7421 ± 0.0138 | 0.6267 ± 0.0256 |
