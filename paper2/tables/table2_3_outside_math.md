# Table 2. Outside math (12M, bits per byte, 2 seeds; eval patch rate in parentheses)

| domain | rule | eval rate | bpb | bpb on new bytes | bpb on copied bytes |
|---|---|---|---|---|---|
| prose | entropy10 | 10.1% | 1.832 ± 0.031 | 1.861 ± 0.032 | 1.611 ± 0.027 |
| prose | novelty | 10.3% | 1.770 ± 0.044 | 1.797 ± 0.044 | 1.556 ± 0.042 |
| code | entropy_tnov | 5.9% | 1.711 ± 0.024 | 1.970 ± 0.026 | 1.264 ± 0.019 |
| code | entropy10 | 7.6% | 1.649 ± 0.004 | 1.906 ± 0.003 | 1.207 ± 0.005 |
| code | novelty | 7.7% | 1.684 ± 0.006 | 1.929 ± 0.006 | 1.261 ± 0.006 |
| code | entropy_t14 | 9.5% | 1.607 ± 0.006 | 1.864 ± 0.007 | 1.165 ± 0.003 |
| code | entropy_t17 | 11.7% | 1.566 ± 0.021 | 1.818 ± 0.022 | 1.132 ± 0.019 |
| code | novelty_t10 | 13.1% | 1.541 ± 0.005 | 1.782 ± 0.007 | 1.127 ± 0.002 |

# Table 3. Code exact match of whole tokens (%), equal eval rate, 2 seeds

| target | n | entropy10 | novelty | difference ± binomial SE |
|---|---|---|---|---|
| copied identifiers | 31,238 | 24.3 | 24.1 | -0.2 ± 0.2 |
| new identifiers | 60,690 | 15.0 | 15.8 | +0.9 ± 0.1 |
| new numbers | 3,477 | 14.2 | 13.4 | -0.8 ± 0.6 |
| copied values after = / return / ( | 6,514 | 21.4 | 19.6 | -1.8 ± 0.5 |
| new values after = / return / ( | 5,536 | 16.0 | 15.7 | -0.3 ± 0.5 |
