# Email to the BLT authors (draft; send only after paper 2 is on arXiv)

To: Artidoro Pagnoni (artidoro at cs.washington.edu)
Cc: Srinivasan Iyer (sviyer at meta.com)
Subject: Entropy patching misses computed results; a cheap reuse-based patcher fixes it (BLT follow-up)

---

Dear Dr. Pagnoni and Dr. Iyer,

I'm an independent researcher working on byte-level language models, building on BLT (by training I'm a physician,
running an emergency department in Chile; this is self-funded work on the side). I wanted to share two
results that I think bear on how BLT allocates compute, and to ask whether they would be of interest to your group.

1. Entropy patching rarely starts a patch at in-line computed results (e.g. the number after "="). These bytes look
   ordinary to the small entropy model (average surprisal), so a surprisal-driven rule passes over them; on the
   released BLT-1B this is visible as low accuracy on computed results under tight patch budgets
   (arXiv:2610.11790).

2. A label-free alternative, starting a patch where the upcoming bytes are unlikely to repeat recent context (a
   2-byte lookup table fitted in seconds, LZ-style), fixes this in math and the gain grows with scale. At 50M
   parameters, at the same patch budget (3 seeds each): computed-result accuracy 57% vs 21% for entropy, final
   answers 90% vs 60%; at 12M 72% vs 47% on final answers; no difference at 1M. Fitted on the raw pretraining
   corpus it keeps the gain. On prose it modestly lowers bits per byte; on code it ties entropy
   (arXiv:XXXX.XXXXX).

My evidence stops at 50M trained from scratch, plus inference-only layouts on BLT-1B, so the open question is
whether it holds at the scales BLT targets. The patcher is a drop-in replacement for the entropy model (about 30
lines, causal, no extra network), and the code, configs and a preregistered scale-up plan are public:
[repository link].

If this is useful, I would be glad to help in whatever way fits: preparing a patch for the BLT codebase, running
specific checks you'd like to see, or simply hearing whether you've seen something similar or find it useful.

Thank you for releasing BLT and its weights; this work would not have been possible otherwise.

Best regards,
[Your name], MD
Independent researcher · Puerto Montt, Chile · nicovera@quetru.cl
[arXiv paper 2 link] · [repository link]

---

## Notes before sending
- Fill in the arXiv ID, repository link, name and affiliation.
- Check the authors' current affiliations and addresses (Holtzman is now at the University of Chicago; Pagnoni's
  listed address is at UW).
- Keep it short: one email, one follow-up after ~2 weeks at most if there is no reply.
- Optional parallel route: a short GitHub issue on facebookresearch/blt offering the drop-in patcher (a
  contribution, not a request), linking the paper.
- Possible second email later: SOMBRERO authors (Aleph Alpha), framed around their boundary-enrichment metric and our
  finding that high enrichment goes with the worst computed accuracy.
