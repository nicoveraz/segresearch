# Paper outline (issue #25)

**Title:** Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy patching misses

**Scope:** math (GSM8K, MATH). Code as a small-effect contrast; logic and program traces as limits. Every number below
comes from `results/tables.md`, `results/stats.md` or `results/figures/` (rebuilt from the registry).

## Claim

Byte-level models that start patches where next-byte entropy is high (BLT, Scratchpad Patching) skip positions whose
*type* is predictable but whose *value* must be computed, such as the number after `=`. Under tight patch budgets this
costs most computed-result accuracy and much final-answer accuracy. A boundary-dependence signal, measured from the
model's own losses without labels, recovers most of it.

## Sections, evidence, status

| # | Section | Evidence | Status |
|---|---|---|---|
| 1 | Introduction: BLT allocates global compute by entropy; entropy is about the next byte, not about what the decoder needs | — | to write |
| 2 | Setup: BLT-lite harness (cross-attention pooling, windowed local decoder), GSM8K + MATH corpus, exact match given the true prefix; BLT-1B with inference-time layouts | README, `literature.md` (BLT rules: global threshold in the release, monotonic in the paper) | to write |
| 3 | **The blind spot in a trained 1B model.** BLT-1B at 10-15% budgets: entropy covers 21-41% of computed results; a boundary after `= ` at equal patch count: +25.0 points at 15% (p = 1e-44), +26.6 at 10% | `tables.md` BLT-1B (train thresholds); `stats.md` §3 | done |
| 4 | **Training at tight budgets.** D=128, 10/15/20%, 3 seeds: hand-written > dependence > entropy at every budget; dependence at 10% beats entropy at 20% on final answers (+23.4) | `tables.md` tight budgets; `stats.md` §2; `figures/acc_vs_compute` | done |
| 5 | **Label-free fix.** Boundary dependence (loss change from one added boundary, per 2-byte context): BLT-1B entdep +38.2 over entropy on computed results; trained models dep10 67% vs 10% final answers | `tables.md`, `stats.md` | done; note: the best combination differs (entdep on BLT-1B, dep alone in training) |
| 6 | **Dose-response: computed vs copied.** Code identifiers +2-4 (significant), proof steps small, copied trace values +3-7; computed values need the skill and long-range context (traces null at window 32; window 8 suggestive) | `tables.md` BLT code/trace/logic; `stats.md` | done |
| 7 | **Scratchpad Patching.** Entropy trigger = random scratchpads across 3 seeds (5.4% vs 5.4%); answer-start scratchpads help on average but unstable (49/31/8%) | `tables.md` scratchpad16 | seeds 3-4 running |
| 8 | **Compute.** BLT-1B at 10%: 61% of default forward FLOPs (45% with a lookup rule); small harness 76% | `flops.py`; `figures/blt1b_acc_vs_compute` | done |
| 9 | Negative results and lessons: toy effect mostly architectural (sum pooling + patch-local decoder); Rho-1 and learned chunking do not help; two-seed effects that vanished | `notes.md`, `program.md` standards | to write (or move to #24) |
| 10 | **Adapting BLT-1B to the budget (#11).** LoRA, 1500 steps, 3 runs per rule, each under its own layout: entropy 27.6%, hand-written 57.0%, entdep 70.0% on computed results (entdep - entropy +42.4, p = 2e-237) | `stats.md` §3 | done |
| 11 | Limits: small trained models; teacher-forced scoring (end to end, even the fine-tuned BLT-1B gets 2% under every layout, #17); LoRA, not full training; dependence tables do not transfer across formats (#5) | issues | to write |

## Figures and tables

1. Diagram: where entropy places boundaries in a GSM8K line vs where dependence does (one example, BLT-1B). *To make.*
2. BLT-1B: computed-result exact vs budget, per layout (train thresholds). *To make from `items/`.*
3. Accuracy vs compute per byte, trained models (exists: `results/figures/acc_vs_compute.svg`); add a BLT-1B panel.
4. Table: dose-response across target types (computed results, final answers, code identifiers, proof steps, trace values).
5. Table: Scratchpad Patching triggers (after seeds 3-4).

## Gaps before submission

- **#17** end-to-end accuracy: at the floor with BLT-1B even after fine-tuning (2% for every layout); needs a stronger model (#12). State it as a limit.
- **#13** full novelty search right before submission (quick re-check 2026-10-02: nothing found).
- Figures 1, 2 and the BLT-1B compute panel.

**Venue:** TMLR (rolling) or the next ICML / ACL ARR / COLM cycle.
