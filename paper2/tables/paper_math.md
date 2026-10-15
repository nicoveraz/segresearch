| size | target | Entropy | Dependence | Syntax + entropy | **Novelty** | Novelty (corpus) |
|---|---|---|---|---|---|---|
| 1M | computed | 4.4 ± 0.7 | 2.8 ± 2.4 | 4.6 ± 1.1 | 3.7 ± 0.6 | – |
|  | final | 7.0 ± 2.0 | 3.6 ± 3.2 | 45.5 ± 17.5 | 12.6 ± 10.5 | – |
|  | bits/byte | 2.093 ± 0.012 | 3.037 ± 1.458 | 2.109 ± 0.014 | 2.125 ± 0.018 | – |
| 12M | computed | 5.8 ± 0.9 | 7.7 ± 2.4 | 12.9 ± 2.4 | 10.1 ± 3.9 | 10.2 ± 2.7 |
|  | final | 46.6 ± 4.0 | 56.7 ± 2.4 | 76.3 ± 2.3 | 71.5 ± 0.7 | 69.8 ± 1.4 |
|  | bits/byte | 1.657 ± 0.013 | 1.673 ± 0.031 | 1.645 ± 0.022 | 1.674 ± 0.029 | 1.619 ± 0.003 |
| 50M | computed | 20.7 ± 7.9 | 35.1 ± 1.8 | – | 57.5 ± 1.1 | 57.3 ± 2.3 |
|  | final | 59.9 ± 1.3 | 78.2 ± 0.9 | – | 89.5 ± 1.0 | 86.4 ± 1.1 |
|  | bits/byte | 1.416 ± 0.009 | 1.420 ± 0.007 | – | 1.446 ± 0.005 | 1.436 ± 0.009 |

: Math: exact match (%) on in-line computed results and final answers, and bits per byte, at a ~11% evaluation patch rate. Mean ± sd over 3 seeds (syntax + entropy at 50M not run with clipping). 50M runs use gradient clipping for every rule.
