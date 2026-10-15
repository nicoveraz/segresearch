| setting | rule | computed | final |
|---|---|---|---|
| trained, 12M | entropy | 5.8 ± 0.9 | 46.6 ± 4.0 |
| trained, 12M | typed uncertainty | 6.2 ± 0.5 | 18.9 ± 7.4 |
| trained, 12M | typed + answer cue | 5.9 ± 0.7 | 25.2 ± 20.6 |
| trained, 50M | entropy | 20.7 ± 7.9 | 59.9 ± 1.3 |
| trained, 50M | typed uncertainty | 8.9 ± 1.1 | 35.1 ± 21.1 |
| trained, 50M | typed + answer cue | 9.5 ± 0.2 | 65.7 ± 2.1 |
| 12M trained with novelty, rule switched at inference | novelty (as trained) | 12.1 ± 0.1 | 73.6 ± 6.6 |
| 12M trained with novelty, rule switched at inference | entropy | 3.2 ± 0.8 | 0.0 ± 0.0 |
| 12M trained with novelty, rule switched at inference | typed uncertainty | 2.8 ± 0.8 | 0.5 ± 0.6 |
| BLT-1B, layout changed at inference (10%) | entropy (BLT default rule) | 19.0 | 51.4 |
| BLT-1B, layout changed at inference (10%) | novelty (GSM8K-fitted table) | 56.5 | 19.4 |
| BLT-1B, layout changed at inference (10%) | novelty + answer cue | 54.8 | 65.6 |
| BLT-1B, layout changed at inference (10%) | typed uncertainty | 48.7 | 32.0 |
| BLT-1B, layout changed at inference (10%) | typed + answer cue | 48.4 | 64.6 |

: What did not work, and inference-time layouts (exact match, %). Trained rows: 3 seeds; switched rows: 2 seeds; BLT-1B: 294 GSM8K test problems [@vera2026easy].
