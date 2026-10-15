# Table 4. Where boundaries fall (held-out bytes; ~10% rate unless noted)

Coverage = share of the category that starts a patch; share = fraction of all boundaries in the category (chance = the category's share of bytes).

## math

| predictor | line/statement start: cov / share (chance) | after '='/'return '/'(': cov / share (chance) | token start: cov / share (chance) | mid-identifier: cov / share (chance) |
|---|---|---|---|---|
| novelty_table (rate 9.5%) | 75% / 9% (1%) | 26% / 5% (2%) | 15% / 32% (20%) | 5% / 27% (47%) |
| novelty_net (rate 10.0%) | 38% / 4% (1%) | 21% / 4% (2%) | 13% / 26% (20%) | 8% / 37% (47%) |
| novelty_oracle (rate 9.0%) | 52% / 6% (1%) | 15% / 3% (2%) | 16% / 35% (20%) | 6% / 32% (47%) |
| gain_table (rate 9.9%) | 27% / 3% (1%) | 29% / 6% (2%) | 10% / 19% (20%) | 11% / 51% (47%) |
| gain_net (rate 10.0%) | 11% / 1% (1%) | 13% / 3% (2%) | 12% / 23% (20%) | 10% / 45% (47%) |

## code

| predictor | line/statement start: cov / share (chance) | after '='/'return '/'(': cov / share (chance) | token start: cov / share (chance) | mid-identifier: cov / share (chance) |
|---|---|---|---|---|
| novelty_table (rate 10.2%) | 0% / 0% (2%) | 0% / 0% (2%) | 28% / 30% (11%) | 12% / 57% (50%) |
| novelty_net (rate 10.1%) | 9% / 2% (2%) | 2% / 0% (2%) | 13% / 15% (11%) | 14% / 68% (50%) |
| novelty_oracle (rate 7.6%) | 18% / 6% (2%) | 12% / 3% (2%) | 16% / 23% (11%) | 7% / 49% (50%) |
| gain_table (rate 10.0%) | 1% / 0% (2%) | 9% / 1% (2%) | 10% / 11% (11%) | 14% / 71% (50%) |
| gain_net (rate 10.1%) | 7% / 2% (2%) | 12% / 2% (2%) | 10% / 11% (11%) | 12% / 57% (50%) |

## H-Net on code (public checkpoints; chunk start at the byte or the preceding space)

| model / stage | chunk rate | statement starts | values after = / return / ( | other token starts | inside identifiers |
|---|---|---|---|---|---|
| hnet_2stage_XL s0 | 64.1% | 59% (100%) | 80% (86%) | 60% (94%) | 40% (40%) |
| hnet_2stage_XL s1 | 26.5% | 33% (45%) | 28% (35%) | 12% (65%) | 4% (4%) |
| hnet_1stage_XL s0 | 50.8% | 33% (100%) | 41% (65%) | 21% (88%) | 18% (18%) |
