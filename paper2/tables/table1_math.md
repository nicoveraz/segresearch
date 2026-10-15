# Table 1. Math: computed-result / final-answer exact match (%) and bits per byte, ~10% patches

Mean ± sd over seeds (3 unless marked). 1M and 12M unclipped; 50M with gradient clipping (norm 1.0) for every rule. Same corpus (SHA-256 pinned), same 1000 computed / 660 final targets at every size.

## Computed results

| size | Entropy (BLT) | Dependence table (paper 1) | Syntax + entropy (hand-written) | **Novelty** | Novelty, corpus-fitted |
|---|---|---|---|---|---|
| 1M | 4.4 ± 0.7 | 2.8 ± 2.4 | 4.6 ± 1.1 | 3.7 ± 0.6 | – |
| 12M | 5.8 ± 0.9 | 7.7 ± 2.4 | 12.9 ± 2.4 | 10.1 ± 3.9 | 10.2 ± 2.7 |
| 50M | 20.7 ± 7.9 | 35.1 ± 1.8 | – | 57.5 ± 1.1 | 57.3 ± 2.3 |

## Final answers

| size | Entropy (BLT) | Dependence table (paper 1) | Syntax + entropy (hand-written) | **Novelty** | Novelty, corpus-fitted |
|---|---|---|---|---|---|
| 1M | 7.0 ± 2.0 | 3.6 ± 3.2 | 45.5 ± 17.5 | 12.6 ± 10.5 | – |
| 12M | 46.6 ± 4.0 | 56.7 ± 2.4 | 76.3 ± 2.3 | 71.5 ± 0.7 | 69.8 ± 1.4 |
| 50M | 59.9 ± 1.3 | 78.2 ± 0.9 | – | 89.5 ± 1.0 | 86.4 ± 1.1 |

## Bits per byte

| size | Entropy (BLT) | Dependence table (paper 1) | Syntax + entropy (hand-written) | **Novelty** | Novelty, corpus-fitted |
|---|---|---|---|---|---|
| 1M | 2.093 ± 0.012 | 3.037 ± 1.458 | 2.109 ± 0.014 | 2.125 ± 0.018 | – |
| 12M | 1.657 ± 0.013 | 1.673 ± 0.031 | 1.645 ± 0.022 | 1.674 ± 0.029 | 1.619 ± 0.003 |
| 50M | 1.416 ± 0.009 | 1.420 ± 0.007 | – | 1.446 ± 0.005 | 1.436 ± 0.009 |

## 50M without clipping (reference; nov10 was only run clipped)

| rule | computed | final | bpb |
|---|---|---|---|
| Entropy (BLT) | 13.9 ± 3.1 | 59.0 ± 1.3 | 1.439 ± 0.005 |
| Dependence table (paper 1) | 35.9 ± 3.4 | 78.9 ± 1.1 | 1.420 ± 0.007 |
| Syntax + entropy (hand-written) | 43.3 ± 9.3 | 85.5 ± 0.3 | 1.431 ± 0.014 |
