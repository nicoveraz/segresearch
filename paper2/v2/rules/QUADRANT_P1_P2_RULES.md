<!-- Extract of general_rule.md from the author's private working repository: lines 119-136 at 3de575a, the
file's last commit (lines 66-83 at 36886d3, where the rules were fixed on 2026-10-23; the section is identical at both): the decision rules for P1 and P2 of the quadrant test
(results/blt_quadrant.json, results/blt_quadrant_verdicts.json). The rest of that file is not shipped. -->

### Decision rules for P1-P2, fixed before the quadrant results (written 2026-10-23 while `realblt_quadrant.py` was at 50/300 problems; no result read)

Measure: gain = per-unit bits with the unit's patch starts removed minus with them added (rest of the entropy
layout fixed), BLT-1B, 10%. Intervals: 95% bootstrap over problems, for the paired difference where two kinds are
compared (same problems).
- **P1 holds** if (a) gain(computed) exceeds gain(id_new), gain(id_copy), gain(copied) and gain(word_rep), each
  difference's interval above 0; and (b) the upper interval bounds of gain(id_new) and of each copy kind are below
  25% of gain(computed). **P1 fails** if any of those kinds has a lower bound at or above 50% of gain(computed).
  Otherwise inconclusive. Primary unit: per unit (one start = one decision); per byte reported alongside.
- **final** (a copy that ends a segment) and **word_new** carry no P1 prediction: final answers are the known
  exception (open problems), and new words mix learnable and unlearnable content.
- **P2** uses an *independent* learnability measure, not BLT-1B's own loss (which shares a term with the gain and
  would favour "easy" units by regression to the mean, review 3): Qwen2.5-1.5B's surprisal per byte of the same new
  word in the same context. **P2 holds** if gain per unit in the easiest third exceeds the hardest third (interval
  of the difference above 0); fails if the difference's upper bound is at or below 0. The split by BLT-1B's own loss (`bits_full`), already in the script,
  is reported but not used for P2.
- The design was changed once after a 6-problem smoke test (forced vs entropy -> with vs without starts at the
  unit), before these rules and before any full-run result; disclosed with the result.
