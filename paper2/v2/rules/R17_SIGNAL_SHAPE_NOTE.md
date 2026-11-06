# R17: the shape of the patcher's signal around a number (descriptive; written 2026-10-28 before it ran)

**Question (PI).** Not the signal's level at one byte, but how it moves byte by byte around a number: does BLT-1B's
patcher show computed values differently from copies before the number (the detector's score over the preceding
words) or inside it (entropy on the second and later digits, where a copy becomes predictable)?

**What is computed.** R12's data and candidates; on the 275 test problems of each format (prose rewrites, equation
text), around every number that starts at a candidate, from 8 bytes before its first digit to 4 bytes after:
- the patcher's entropy for each byte (offset 0 = the first digit's entropy, the value BLT's rule thresholds);
- the score of R12's label-free probe (auto_prose / auto_orig, refitted here exactly as R12, on each format's train
  problems) applied to the state predicting each byte (offset 0 = the decision state R12 uses; other offsets are
  outside the probe's training positions and shown as description only);
- grouped by kind: computed value, other new number, copy (value seen earlier); word starts as a reference.
Reported: mean and 95% interval (bootstrap over problems) per kind and offset; inside numbers only while the byte is
still part of the number. Plus two separations (AUC, computed vs copy): the probe score at offsets -4 to 0, and
entropy at offsets 0 to +2 (numbers with at least 3 digits for +2).

**No verdicts.** This is description. A patching rule suggested by it (for example a start at the second digit
when entropy stays high) gets its own decision rule before it is tested.

Script: realblt_profiles.py -> results/blt_profiles.json.

## Results (2026-10-28; results/blt_profiles.json, .log, .png; description, no verdicts)

Test problems; counts per kind (prose / equation text): computed 711 / 711, other new numbers 418 / 644, copies
1,956 / 2,133, words 20,389 / 9,202.
- **The detector's score jumps at the decision point; it does not build up over the words before.** Prose: about -5
  to -7 (log-odds) from 8 bytes before to 1 byte before for every kind, then +0.8 (computed), +1.2 (other new), -1.9
  (copies) at the first digit. Separation of computed values from copies (AUC): 0.61 one byte before, 0.84 at the
  decision point (equation text: 0.81, 0.96).
- **Entropy separates results from copies only inside the number.** At the first digit the kinds look alike (prose
  2.31 [2.26, 2.35] computed vs 2.18 [2.14, 2.23] copies, AUC 0.56; equation text AUC 0.40). From the second digit
  copies become predictable and results do not: prose +1 byte 1.89 [1.83, 1.94] vs 1.06 [1.01, 1.11], +2 bytes 1.56
  vs 0.59; AUC 0.84 at +1 and 0.90 at +2 (equation text 0.76, 0.82). No training is involved: it is the patcher's own
  entropy.
- Other new numbers (introduced in the solution, not marked as computed) sit between the two.

**Reading.** Two shapes. The learned detector is a last-byte signal: it fires once the phrase before the number is
complete. Entropy carries a training-free signal in its persistence: uncertainty that stays high after the first
digit marks a value the model cannot recover from the context (a result); uncertainty that collapses marks a copy.
It arrives one byte late for a start at the first digit. Whether a start at the second digit, where entropy stays
high, recovers part of the gain is untested and needs its own decision rule.
