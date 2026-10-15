| rule | eval rate | enrichment B | computed-result starts covered | computed (50M) | final (50M) |
|---|---|---|---|---|---|
| Entropy | 11.2% | 1.61 | 1% | 20.7 | 59.9 |
| Dependence | 10.9% | 1.02 | 37% | 35.1 | 78.2 |
| Syntax + entropy | 12.1% | 1.52 | 100% | – | – |
| Novelty | 11.9% | 1.13 | 100% | 57.5 | 89.5 |
| Novelty (corpus) | 11.4% | 1.03 | 99% | 57.3 | 86.4 |

: Boundary enrichment (mean surprisal at patch starts / mean surprisal; surprisal from the patcher's byte model) on the math evaluation set at the 50M thresholds, coverage of computed-result starts, and 50M accuracy (%, 3 seeds). Mean surprisal at computed-result starts: 3.34 bits; all bytes: 3.12 bits.
