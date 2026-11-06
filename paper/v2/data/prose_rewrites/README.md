# Data card: GSM8K solutions rewritten as prose

Used in section 5 of version 2 of *Easy to anticipate, hard to compute* (arXiv:2610.11790v2) to test whether BLT-1B's
blind spot depends on how computed values are written. Not meant as a training set.

## What it is

The same GSM8K problems in two writings of the solution:

- **orig**: the question, the worked solution with calculator annotations removed, and "The final answer is N" (the
  BLT-1B format of `realblt_budget.problem()`);
- **prose**: the same question, the solution rewritten by an LLM as explanatory prose with no `=` and no arithmetic
  symbol between numbers, every computed result of the original kept in order with the same digits, and the same
  final-answer line.

A computed value is a number right after `= ` in the original solution (digits only; a `$` before it is not part of
it). Problems whose equations have no space after `=` have no computed values by this definition and serve only as
training text.

| File | Problems | With computed values | Computed values |
|---|---|---|---|
| `train.jsonl` | 1,382 of GSM8K train problems 0-1,499 | 1,099 | 3,597 |
| `test.jsonl` | 275 of GSM8K test problems 0-299 | 221 | 711 |

## Source and generation

- Source: GSM8K (openai/grade-school-math), the files `realtext.py` downloads. GSM8K is released under the MIT
  licence; its copyright and permission notice is in `LICENSE-GSM8K` and covers the questions, solutions and answers
  copied here.
- Writer: Claude Opus 5.5 (model id `claude-opus-5-5`; every record's `served_by` field says so) through the Anthropic
  API, on 2026-10-26: a pilot of 20 standard requests, the rest as one Message Batch, failures retried once with
  standard requests. Settings: effort "low", at most 4,000 output tokens, output constrained to the JSON schema
  `{"solution": string}`. Cost: about $7.6 in all.
- Prompt (system message, verbatim):

  ```
  You rewrite worked solutions to grade-school math problems as natural, fluent explanatory prose, the way a teacher would explain the solution in writing.

  Rules:
  - Do not use "=" or any arithmetic symbol (+, -, *, /, x, ×, ÷) between numbers anywhere in the solution; describe each operation in words. Write fractions in words or as decimals.
  - Keep every computed result you are given, in the given order, written exactly as given (same digits, commas and decimal point; a $ sign before money is fine). Wrap each one in double square brackets the first time you state it, like [[18]]; mark nothing else. You may mention other numbers from the problem.
  - Vary how results are introduced, as natural writing does; do not reuse one fixed phrase.
  - Do not end with a separate final-answer line; one is added afterwards.
  - Plain ASCII text in one or more paragraphs; no lists, headings, LaTeX or markdown.

  Return JSON: {"solution": "<the rewritten solution>"}.
  ```

- User message: `Problem:\n<question>\n\nOriginal solution:\n<solution without annotations>\n\nComputed results, in
  order: <values, comma-separated, or "(none)">`.
- The generation script is in the author's private working repository; the rewrites were made with its version at
  commit 1b7d410 (the data were committed at 152c2df, on 2026-10-26). Its later changes, up to 513e14c, leave the
  prompt and settings unchanged. The prompt, settings and checks above are complete.

## Checks (automatic, before any model was run on the data)

A rewrite is kept only if: the output is valid JSON; the solution is ASCII; the values marked `[[...]]` equal the
original's computed values, in order (a leading `$` ignored); after the markers are removed there is no `=` and no
operator between two numbers (regular expression `\d\s*[-+*/x×÷]\s*\d`); and both writings fit BLT-1B's 1,000-token
limit (999 bytes plus BOS). Each failure was retried once, then the problem was dropped from both writings. Most
drops were rewrites too long for the token limit.

## Fields

`split`, `idx` (the GSM8K index in that split), `question`, `solution` (the original, annotations removed), `answer`,
`values` (the computed values, as strings), `orig` and `prose` (the two full texts), `orig_spans` and `prose_spans`
(character offset of each value's first digit in that text), `final_start_orig` and `final_start_prose` (offset of the
final answer), `marked` (the model's solution with its `[[...]]` markers), `served_by`.

## Known limits

One writer model and one prompt. A rewrite changes wording, where values sit and how hard they are, not only the cue
before them; BLT-1B's default layout gets 77.9% of the values right in the original writing and 56.8% in prose. Prose
spends about 1.43 times as many bytes per value. Values are marked only at their first statement.
