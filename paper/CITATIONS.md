# Citation ledger

Why this file exists (from textca's ledger): a prior-art gate there refuted 13 of 74 extracted claims for
overreaching their own sources, so summaries are not citable. Every work cited in `manuscript.md` gets an entry:
where it is cited, what the manuscript claims about it, the supporting quote, and the **basis** of the check.

**Bases.** `LOCAL FULL TEXT`: the PDF was fetched to this machine, extracted with `pdftotext -layout`, and the
quote was read there. `FULL TEXT (2026-10-01)`: the source was read in the literature check (`../literature.md`)
but the quote was not recorded; upgrade it to LOCAL FULL TEXT before submission. `ABSTRACT PAGE`: only the arXiv
abstract page was checked. `DATASET` / `CODE`: cited for a dataset or released artifact, not for a claim.

**Status: 16 works; 0 at LOCAL FULL TEXT, 2 at FULL TEXT (2026-10-01), 12 at ABSTRACT PAGE, 2 DATASET.**
Not ready for submission: every entry carrying a claim must reach LOCAL FULL TEXT with its quote.

---

| Key | Cited in | Manuscript's claim | Basis | Quote |
|---|---|---|---|---|
| `pagnoni2025blt` | §1, §2 | BLT starts patches where a small model's next-byte entropy crosses a threshold; it describes a global-threshold rule and an approximate monotonic rule, and uses the monotonic one for its main BLT-Entropy model. The released BLT-1B and the official code default to the global threshold (`monotonicity: false`, 1.335). | FULL TEXT (2026-10-01) + CODE (HF config, `bytelatent/data/patcher.py`) | TODO |
| `zheng2026scratchpad` | §2, §9 | Scratchpad Patching inserts transient global steps inside patches, triggered by an absolute next-byte entropy threshold; it names stale patch context "patch lag"; its trigger ablation (appendix E.2) compares on bits per byte, code pass@1 and multiple-choice tasks, found entropy best, and reports no math answer accuracy. | FULL TEXT (2026-10-01; appendix B.3, E.2) | TODO |
| `kallini2026fastblt` | §2 | Fast BLT speeds up decoding without changing where boundaries go. | ABSTRACT PAGE | TODO |
| `nawrot2023dynamic` | §2 | Entropy spikes used as boundaries in dynamic pooling. | ABSTRACT PAGE | TODO |
| `slagle2024spacebyte` | §1, §2 | SpaceByte starts patches at word boundaries. | ABSTRACT PAGE | TODO |
| `hwang2025hnet` | §1, §2 | H-Net learns boundaries (dynamic chunking). | ABSTRACT PAGE | TODO |
| `videau2025aunet` | §2 | AU-Net derives boundaries another way. | ABSTRACT PAGE | TODO |
| `kallini2024mrt5` | §2 | MrT5 learns which bytes to keep. | ABSTRACT PAGE | TODO |
| `goriely2025bytespan` | §2 | ByteSpan derives boundaries from information. | ABSTRACT PAGE | TODO |
| `singh2024tokenization` | §2 | Number segmentation in static tokenizers affects arithmetic. | ABSTRACT PAGE | TODO |
| `meister2026tokeval` | §2 | Same, including digit place-value alignment. | ABSTRACT PAGE | TODO |
| `lin2024rho1` | §2 | Rho-1 selects training tokens by excess loss against a reference model. | ABSTRACT PAGE | TODO |
| `raposo2024mod` | §2 | Mixture-of-Depths routes compute per token with a learned router. | ABSTRACT PAGE | TODO |
| `haltiuk2026disentangling` | not cited | `literature.md`: a position paper; cite only as a hypothesis if at all. | ABSTRACT PAGE | — |
| `cobbe2021gsm8k` | §3 | GSM8K. | DATASET | — |
| `hendrycks2021math` | §3 | MATH. | DATASET | — |

**The novelty claim** (§1: no prior report of entropy-based patching, scratchpads or learned chunking failing at
computed outputs) rests on the searches in `../literature.md` (2026-10-01, quick re-check 2026-10-02). Re-run
it right before submission (#13).
