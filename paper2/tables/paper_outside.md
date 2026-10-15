| domain | rule | eval patch rate | bits/byte | computed values | copied values |
|---|---|---|---|---|---|
| prose | entropy10 | 10.1% | 1.832 ± 0.031 | – | – |
| prose | novelty | 10.3% | 1.770 ± 0.044 | – | – |
| code | entropy_tnov | 5.9% | 1.711 ± 0.024 | – | – |
| code | entropy10 | 7.6% | 1.649 ± 0.004 | – | – |
| code | novelty | 7.7% | 1.684 ± 0.006 | – | – |
| code | entropy_t14 | 9.5% | 1.607 ± 0.006 | – | – |
| code | entropy_t17 | 11.7% | 1.566 ± 0.021 | – | – |
| code | novelty_t10 | 13.1% | 1.541 ± 0.005 | – | – |
| traces | entropy10 | 10.0% | 1.127 ± 0.033 | 55.1 ± 8.6 | 56.9 ± 5.4 |
| traces | novelty | 10.3% | 1.157 ± 0.014 | 50.2 ± 0.2 | 61.0 ± 3.5 |
| traces | novelty_net | 10.0% | 1.196 ± 0.001 | 47.9 ± 0.3 | 36.6 ± 0.3 |

: Outside math (12M, 2 seeds; mean ± sd). Code: entropy at four budgets (entropy_tnov, entropy10, entropy_t14, entropy_t17) and novelty at two (novelty, novelty_t10). Traces: exact match (%) on computed and copied printed values; novelty_net is the 128-byte novelty predictor (§6).
