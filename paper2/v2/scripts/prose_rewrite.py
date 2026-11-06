"""R9 (paper2v2/R9_PROSE_RULES.md): rewrite GSM8K solutions as natural prose without equations, with Claude Opus 5.5.
Every computed result of the original (the number after '= ') is kept, in order, with the same digits, and marked
[[...]] at its first statement; the markers give the value positions and are removed before scoring.

    .venv-api/bin/python paper2v2/prose_rewrite.py pilot          first 20 train problems, standard requests, cost
    .venv-api/bin/python paper2v2/prose_rewrite.py batch          the rest (train up to 1,500, test 300), Batches API
    .venv-api/bin/python paper2v2/prose_rewrite.py collect        fetch the batch, validate, retry failures once
    .venv-api/bin/python paper2v2/prose_rewrite.py fresh          R20: GSM8K test problems 300-1,318, Batches API
    .venv-api/bin/python paper2v2/prose_rewrite.py fresh-collect  fetch them, validate, retry once (not "too long")
Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile, plus ANTHROPIC_WORKSPACE_ID for a
key not scoped to a workspace); this script never reads or
prints them. Output: ~/.cache/segresearch-prose/{train,test}.jsonl; cost log results/prose_rewrite_cost.json.
"""
import json
import os
import re
import sys
import time

import anthropic

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSM = os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")
GSM_TRAIN = os.path.expanduser("~/.cache/segresearch-gsm8k/train.jsonl")
OUT = os.path.expanduser("~/.cache/segresearch-prose")
COST = os.path.join(HERE, "results", "prose_rewrite_cost.json")
MODEL, EFFORT, MAX_TOKENS = "claude-opus-5-5", "low", 4000
PRICE = {"standard": (4.0, 20.0), "batch": (2.0, 10.0)}          # $ per million input / output tokens
N_TRAIN, N_TEST, N_PILOT, BUDGET = 1500, 300, 20, 30.0
FRESH, FRESH_CAP = (300, 1319), 8.0                              # R20: the test problems R9 did not rewrite; $ cap
MAX_BYTES = 999                                                  # BLT-1B scripts: 1,000 tokens including BOS
VAL = re.compile(r"= \$?(\d[\d,]*(?:\.\d+)?)")
SYSTEM = """You rewrite worked solutions to grade-school math problems as natural, fluent explanatory prose, the way a \
teacher would explain the solution in writing.

Rules:
- Do not use "=" or any arithmetic symbol (+, -, *, /, x, ×, ÷) between numbers anywhere in the solution; describe \
each operation in words. Write fractions in words or as decimals.
- Keep every computed result you are given, in the given order, written exactly as given (same digits, commas and \
decimal point; a $ sign before money is fine). Wrap each one in double square brackets the first time you state \
it, like [[18]]; mark nothing else. You may mention other numbers from the problem.
- Vary how results are introduced, as natural writing does; do not reuse one fixed phrase.
- Do not end with a separate final-answer line; one is added afterwards.
- Plain ASCII text in one or more paragraphs; no lists, headings, LaTeX or markdown.

Return JSON: {"solution": "<the rewritten solution>"}."""
SCHEMA = {"type": "object", "properties": {"solution": {"type": "string"}}, "required": ["solution"],
          "additionalProperties": False}


def items(split):
    if split == "fresh":
        lines, base = list(open(GSM))[FRESH[0]:FRESH[1]], FRESH[0]       # idx stays the GSM8K test index
    else:
        path, n = (GSM_TRAIN, N_TRAIN) if split == "train" else (GSM, N_TEST)
        lines, base = list(open(path))[:n], 0
    out = []
    for i, l in enumerate(lines):
        p = json.loads(l)
        sol, ans = p["answer"].rsplit("#### ", 1)
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        vals = [m.group(1) for m in VAL.finditer(sol)]
        out.append({"split": split, "idx": base + i, "question": p["question"], "solution": sol, "answer": ans.strip(),
                    "values": vals})
    return out


def user_msg(it):
    return (f"Problem:\n{it['question']}\n\nOriginal solution:\n{it['solution']}\n\n"
            f"Computed results, in order: {', '.join(it['values']) if it['values'] else '(none)'}")


def params(it):
    return {"model": MODEL, "max_tokens": MAX_TOKENS, "system": SYSTEM,
            "messages": [{"role": "user", "content": user_msg(it)}],
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}}}


def validate(it, text):
    """(record, None) or (None, reason). The record holds both formats and the value spans (byte offsets)."""
    try:
        sol = json.loads(text)["solution"].strip()
    except Exception:
        return None, "bad json"
    if not sol.isascii():
        return None, "non-ascii"
    marks = re.findall(r"\[\[(.*?)\]\]", sol)
    if [m.lstrip("$") for m in marks] != it["values"]:
        return None, "marked values differ"
    clean, spans, pos = "", [], 0
    for m in re.finditer(r"\[\[(.*?)\]\]", sol):
        clean += sol[pos:m.start()]
        v = m.group(1).lstrip("$")
        clean += m.group(1)[:len(m.group(1)) - len(v)]               # a '$' inside the marker stays in the text
        spans.append(len(clean)); clean += v; pos = m.end()
    clean += sol[pos:]
    if "=" in clean:
        return None, "has ="
    if re.search(r"\d\s*[-+*/x×÷]\s*\d", clean):
        return None, "operator between numbers"
    head = it["question"] + "\n"
    prose = head + clean + "\nThe final answer is " + it["answer"]
    orig = head + it["solution"] + "\nThe final answer is " + it["answer"]
    if len(prose.encode()) > MAX_BYTES or len(orig.encode()) > MAX_BYTES:
        return None, "too long"
    o_sp = [len(head) + m.start(1) for m in VAL.finditer(it["solution"])]
    p_sp = [len(head) + s for s in spans]
    for t, sp in ((orig, o_sp), (prose, p_sp)):
        assert [t[s:s + len(v)] for s, v in zip(sp, it["values"])] == it["values"], (it["idx"], "span check")
    return {**it, "orig": orig, "prose": prose, "orig_spans": o_sp, "prose_spans": p_sp,
            "final_start_orig": len(orig) - len(it["answer"]), "final_start_prose": len(prose) - len(it["answer"]),
            "marked": sol}, None


def text_of(msg):
    return next((b.text for b in msg.content if b.type == "text"), "")


def cost_of(usage, tier):
    pin, pout = PRICE[tier]
    tin = usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    return tin * pin / 1e6 + usage.output_tokens * pout / 1e6, tin, usage.output_tokens


def log_cost(key, d):
    c = json.load(open(COST)) if os.path.exists(COST) else {}
    c[key] = d; json.dump(c, open(COST, "w"), indent=1)


def direct(client, todo, tag):
    """Standard requests (pilot, retries). Returns (records, failures, cost dict)."""
    recs, fails, tot = [], [], [0.0, 0, 0]
    for it in todo:
        try:
            msg = client.beta.messages.create(**params(it), betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        except anthropic.APIStatusError as e:
            fails.append((it, f"api {e.status_code}: {e.message[:200]}"))
            if "usage limits" in e.message:                     # the account's spend limit: every later request fails too
                print(f"  stopping retries: {e.message[:200]}", flush=True)
                fails += [(x, "not tried (usage limit)") for x in todo[todo.index(it) + 1:]]; break
            continue
        c, ti, to = cost_of(msg.usage, "standard"); tot[0] += c; tot[1] += ti; tot[2] += to
        if msg.stop_reason == "refusal":
            fails.append((it, "refusal")); continue
        r, why = validate(it, text_of(msg))
        if r:
            r["served_by"] = msg.model; recs.append(r)
        else:
            fails.append((it, why))
        print(f"  {tag} {it['split']}-{it['idx']}: {'ok' if r else why} | ${c:.4f} ({ti} in, {to} out)", flush=True)
    return recs, fails, {"n": len(todo), "usd": tot[0], "input_tokens": tot[1], "output_tokens": tot[2]}


def save(recs):
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test", "fresh"):
        path = os.path.join(OUT, f"{split}.jsonl")
        have = {json.loads(l)["idx"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}
        for r in recs:
            if r["split"] == split:
                have[r["idx"]] = r
        with open(path, "w") as f:
            for k in sorted(have):
                f.write(json.dumps(have[k]) + "\n")


def main(mode):
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")              # for a key not scoped to a workspace
    client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)
    if mode == "pilot":
        todo = items("train")[:N_PILOT]
        recs, fails, cost = direct(client, todo, "pilot")
        save(recs)
        per = cost["usd"] / max(1, cost["n"])
        n_rest = N_TRAIN + N_TEST - N_PILOT
        proj = per * n_rest * PRICE["batch"][1] / PRICE["standard"][1] + cost["usd"]
        cost.update({"valid": len(recs), "failures": [(f["idx"], w) for f, w in fails], "usd_per_rewrite": per,
                     "projected_total_usd_with_batch": proj})
        log_cost("pilot", cost)
        print(f"PILOT {len(recs)}/{len(todo)} valid; ${cost['usd']:.3f} total, ${per:.4f} per rewrite; "
              f"projected total with the rest batched: ${proj:.2f} (limit ${BUDGET:.0f})")
        for f, w in fails:
            print(f"  failed {f['split']}-{f['idx']}: {w}")
    elif mode == "batch":
        done = {(json.loads(l)["split"], json.loads(l)["idx"]) for s in ("train", "test")
                for l in (open(os.path.join(OUT, f"{s}.jsonl")) if os.path.exists(os.path.join(OUT, f"{s}.jsonl")) else [])}
        todo = [it for it in items("train")[N_PILOT:] + items("test") if (it["split"], it["idx"]) not in done]
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        b = client.messages.batches.create(requests=[Request(custom_id=f"{it['split']}-{it['idx']}",
                                                             params=MessageCreateParamsNonStreaming(**params(it)))
                                                     for it in todo])
        log_cost("batch_id", {"id": b.id, "n": len(todo), "created": time.strftime("%Y-%m-%d %H:%M")})
        print(f"BATCH {b.id}: {len(todo)} requests")
    elif mode == "collect":
        bid = json.load(open(COST))["batch_id"]["id"]
        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            print(f"  {b.processing_status}: {b.request_counts.processing} processing", flush=True); time.sleep(120)
        byid = {f"{it['split']}-{it['idx']}": it for it in items("train") + items("test")}
        recs, fails, tot = [], [], [0.0, 0, 0]
        for res in client.messages.batches.results(bid):
            it = byid[res.custom_id]
            if res.result.type != "succeeded":
                fails.append((it, res.result.type)); continue
            msg = res.result.message
            c, ti, to = cost_of(msg.usage, "batch"); tot[0] += c; tot[1] += ti; tot[2] += to
            if msg.stop_reason == "refusal":
                fails.append((it, "refusal")); continue
            r, why = validate(it, text_of(msg))
            if r:
                r["served_by"] = msg.model; recs.append(r)
            else:
                fails.append((it, why))
        save(recs)
        log_cost("batch", {"n": len(recs) + len(fails), "usd": tot[0], "input_tokens": tot[1], "output_tokens": tot[2],
                           "valid": len(recs), "failures": [(f["split"], f["idx"], w) for f, w in fails]})
        print(f"BATCH {len(recs)} valid, {len(fails)} failed; ${tot[0]:.2f}")
        retry = [f for f, w in fails if w not in ("expired", "canceled")]   # an expired batch is resubmitted by hand
        if len(retry) > 200:                                                # (the author chose batch pricing)
            print(f"{len(retry)} failures: too many to retry at standard prices without asking; stopping"); sys.exit(1)
        if retry:
            r2, f2, c2 = direct(client, retry, "retry")
            save(r2)
            log_cost("retry", {**c2, "valid": len(r2), "failures": [(f["split"], f["idx"], w) for f, w in f2]})
            print(f"RETRY {len(r2)}/{len(retry)} valid; ${c2['usd']:.2f}; dropped {len(f2)}")
    elif mode == "fresh":
        c = json.load(open(COST))
        if "fresh_batch_id" in c:
            print(f"fresh batch already submitted: {c['fresh_batch_id']['id']}"); sys.exit(1)
        todo = items("fresh")
        per = c["batch"]["usd"] / c["batch"]["n"]                           # R9's measured batched cost per rewrite
        if per * len(todo) > FRESH_CAP:
            print(f"projected ${per * len(todo):.2f} exceeds the ${FRESH_CAP:.0f} cap; stopping"); sys.exit(1)
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        b = client.messages.batches.create(requests=[Request(custom_id=f"fresh-{it['idx']}",
                                                             params=MessageCreateParamsNonStreaming(**params(it)))
                                                     for it in todo])
        log_cost("fresh_batch_id", {"id": b.id, "n": len(todo), "created": time.strftime("%Y-%m-%d %H:%M"),
                                    "projected_usd": per * len(todo)})
        print(f"FRESH BATCH {b.id}: {len(todo)} requests, projected ${per * len(todo):.2f}")
    elif mode == "fresh-collect":
        bid = json.load(open(COST))["fresh_batch_id"]["id"]
        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            print(f"  {b.processing_status}: {b.request_counts.processing} processing", flush=True); time.sleep(120)
        byid = {f"fresh-{it['idx']}": it for it in items("fresh")}
        recs, fails, tot = [], [], [0.0, 0, 0]
        for res in client.messages.batches.results(bid):
            it = byid[res.custom_id]
            if res.result.type != "succeeded":
                fails.append((it, res.result.type)); continue
            msg = res.result.message
            c, ti, to = cost_of(msg.usage, "batch"); tot[0] += c; tot[1] += ti; tot[2] += to
            if msg.stop_reason == "refusal":
                fails.append((it, "refusal")); continue
            r, why = validate(it, text_of(msg))
            if r:
                r["served_by"] = msg.model; recs.append(r)
            else:
                fails.append((it, why))
        save(recs)
        log_cost("fresh_batch", {"n": len(recs) + len(fails), "usd": tot[0], "input_tokens": tot[1], "output_tokens": tot[2],
                                 "valid": len(recs), "failures": [(f["split"], f["idx"], w) for f, w in fails]})
        print(f"FRESH BATCH {len(recs)} valid, {len(fails)} failed; ${tot[0]:.2f}")
        retry = [f for f, w in fails if w not in ("expired", "canceled", "too long")]   # "too long" rarely fits on retry (R9)
        room = FRESH_CAP - tot[0]
        retry = retry[:max(0, int(room / 0.0089 * 0.8))]                    # R9's standard cost per retry, 20% margin
        if retry:
            r2, f2, c2 = direct(client, retry, "retry")
            save(r2)
            log_cost("fresh_retry", {**c2, "valid": len(r2), "failures": [(f["split"], f["idx"], w) for f, w in f2]})
            print(f"RETRY {len(r2)}/{len(retry)} valid; ${c2['usd']:.2f}; dropped {len(f2)}")


if __name__ == "__main__":
    main(sys.argv[1])
