# Per-seed PoE committee — corpus `non_human`

**Protocol**: PoE = geometric mean of 4 calibrated probs per seed; thr = geometric mean of 4 thresholds per seed. Dedupe by `(seq_id, chembl_id)`. Aggregated mean ± σ over 5 canonical seeds {42, 123, 456, 789, 1024}.

| system | MCC | AUROC | F1 | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| dtkinase | 0.4834 ± 0.0106 | 0.7782 ± 0.0125 | 0.7797 ± 0.0032 | 0.7420 ± 0.0058 | 0.7199 ± 0.0098 | 0.8506 ± 0.0116 |
| drugban | 0.5291 ± 0.0301 | 0.8326 ± 0.0097 | 0.7937 ± 0.0142 | 0.7648 ± 0.0157 | 0.7510 ± 0.0257 | 0.8437 ± 0.0411 |
| graphban | 0.5029 ± 0.0611 | 0.8095 ± 0.0142 | 0.7849 ± 0.0364 | 0.7494 ± 0.0286 | 0.7250 ± 0.0086 | 0.8589 ± 0.0779 |
| conplex | 0.4600 ± 0.0203 | 0.8271 ± 0.0021 | 0.7603 ± 0.0153 | 0.7315 ± 0.0089 | 0.7297 ± 0.0108 | 0.7952 ± 0.0451 |
| **Committee PoE** | 0.5178 ± 0.0301 | 0.8429 ± 0.0118 | 0.7908 ± 0.0145 | 0.7593 ± 0.0140 | 0.7411 ± 0.0151 | 0.8485 ± 0.0344 |
