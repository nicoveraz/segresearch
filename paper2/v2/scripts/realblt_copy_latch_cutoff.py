"""O6c confirmatory test (followup/O6c_CUTOFF_RULES.md). PROPOSED, NOT APPROVED: under the conversion as released
(window 0), does L8 beat SIV+PO<620, the byte rule SIV+PO applied only at token positions t < 620, on new F1-template
texts from GSM8K train idx 6253 on? Every command that reads those problems refuses unless O6C_FILED names a filing
commit of the O6c rule (filed_check): an ancestor of HEAD that introduced the heading FILED_LINE below and changed
text only, from which this script, the filed O6b code it imports (realblt_copy_latch_fresh.py, pinned at 4026bec) and
every repository module they import are unchanged. Each job then runs once per filing, after the job before it, whose
output must be committed and pushed (gate). Its readings are quoted whole in paper 2 v2's Section 6 (paper2v2/PLAN.md,
Decision 2026-10-31).

Imported unchanged from realblt_copy_latch_fresh.py: F1's builder and O6's filter, the masks, the implementation
checks, the matched-rate fit, the layouts, the scorer, the readouts, the bootstrap and the class rule. Added here: the
cutoff mask and its layout, the builder for idx >= 6253, O6c's verdict, and O6c's guard and job gate.

    PY="../segresearch/.venv-blt/bin/python -B"
    $PY realblt_copy_latch_cutoff.py selftest                     bytes and dev problems only; no model
    O6C_FILED=<commit> $PY realblt_copy_latch_cutoff.py blocks     C0: the new texts' indices, IDs and hashes (no model)
    O6C_FILED=<commit> $PY realblt_copy_latch_cutoff.py score      C1: patcher under window 0, checks, then 5 layouts
    O6C_FILED=<commit> $PY realblt_copy_latch_cutoff.py verdict    the class, its reading and the rows beside it
    O6C_SMOKE=1 $PY realblt_copy_latch_cutoff.py blocks|score|verdict   10 fit + 10 test dev problems (train idx
                     0-299) -> *_smoke.*: a code check only, never used for a verdict or a prediction
"""
import sys

sys.dont_write_bytecode = True                  # importing the rule modules must not write bytecode into the repo

import collections
import datetime
import hashlib
import itertools
import json
import os
import re
import subprocess
import tempfile

import numpy as np

import realblt_copy_latch_fresh as F            # the filed O6b code, unchanged; it sets HF_HUB_OFFLINE=1
from realblt_copy_latch_fresh import L, blt_layouts, blt_light, blt_load, torch, AutoTokenizer, MODEL, GSM_TRAIN, _seg

SMOKE = os.environ.get("O6C_SMOKE") == "1"
TAG = "_smoke" if SMOKE else ""
STATUS = ("O6c confirmatory code, PROPOSED (not approved); smoke runs use dev problems and are code checks only"
          if SMOKE or not os.environ.get("O6C_FILED") else "O6c confirmatory code, run under its filing")
RULES = "followup/O6c_CUTOFF_RULES.md"
FILED_LINE = re.compile(r"^## O6c confirmatory rule, filed \((\d{4}-\d{2}-\d{2})\)$", re.M)   # '... filed (YYYY-MM-DD)'
FILED_SINCE = datetime.date(2026, 10, 31)       # an O6c filing heading's date may not be earlier
PIN = re.compile(r"realblt_copy_latch_cutoff\.py, at commit ([0-9a-f]{7,40})")    # the Code section's pin
O6B_CODE, O6B_PIN = "realblt_copy_latch_fresh.py", "4026bec"     # the filed O6b code, which must stay as pinned
CUT = 620                                       # SIV+PO<620 masks starts at token positions t < CUT only
NEW0 = 6253                                     # the new problems: GSM8K train idx >= 6253 (O6b's J0 stopped at 6252)
SEED = 6301                                     # F1's ticket draws for O6c, fixed before any text
SIZES, SMOKE_SIZES, MIN_TEST = (300, 600), (10, 10), 400      # fit, test; fewer than 400 test texts: no class
MASKS = F.RULE_MASKS + ("SIVPO620",)
NAMES = ["D", "D-all", "D.L8", "D.SIVPO", "D.SIVPO620"]
PAIRS = [("D-all", "D"), ("D.L8", "D"), ("D.SIVPO", "D"), ("D.SIVPO620", "D"), ("D.L8", "D.SIVPO"),
         ("D.L8", "D.SIVPO620")]
PRIMARY = "D.L8 vs D.SIVPO620"
OUTPUTS = ("blocks", "score", "verdict")
BINS = ((0, 600), (600, 700), (700, 2000))      # text length in tokens, as O6b's F5 by-length readout


def out_path(name, ext="json"):
    return os.path.join(F.RES, f"blt_copy_latch_cutoff_{name}{TAG}.{ext}")


# ---------------------------------------------------------------- the cutoff rule
def cut_mask(m):
    """SIV+PO<620 from SIV+PO's mask over tokens (True = no start): the same below token CUT, nothing from it on"""
    return m & (np.arange(len(m)) < CUT)


def add_masks(rows):
    """The filed masks (L8, L8c, V_8, SIV, PO, SIV+PO, L8b; with the subset asserts), then SIV+PO<620"""
    F.add_masks(rows)
    for r in rows:
        r["SIVPO620"] = cut_mask(r["SIVPO"])


def layouts(r, fit, r_d):
    """The filed layouts for D, D-all, D.L8 and D.SIVPO, and D.SIVPO620: SIV+PO<620's mask, refitted at D's train
    rate on the fit texts, as every X.R layout is"""
    lay = F.layouts(r, [n for n in NAMES if n != "D.SIVPO620"], fit, r_d)
    lay["D.SIVPO620"] = blt_layouts.layout("entsivpo620", r_d, L.sig_of(r, ("SIVPO620",)), fit)
    return lay


# ---------------------------------------------------------------- the new texts (no model)
def pool_rows():
    """Smoke: dev problems, GSM8K train idx 0-299. Otherwise the new problems, idx >= NEW0 in file order, and only once
    the O6c rule is filed (provenance)."""
    if SMOKE:
        with open(GSM_TRAIN) as f:
            for i, line in enumerate(itertools.islice(f, 300)):
                yield i, json.loads(line)
        return
    provenance()
    with open(GSM_TRAIN) as f:
        for i, line in enumerate(f):
            if i >= NEW0:
                yield i, json.loads(line)


def take_texts(rows, tok, sizes=None):
    """F1's template on the problems in file order (F.make: realblt_quadrant.build, ticket drawn with SEED): the first
    kept are the fit texts, the next up to the test size the test texts. A problem is kept if its text passes O6's
    filter; rejected problems keep their draws. Returns (block record, {'fit': rows, 'test': rows})."""
    keep = F.keeper(tok)
    rng = np.random.default_rng(SEED)
    nf, nt = sizes or (SMOKE_SIZES if SMOKE else SIZES)
    got, visited, first, last = [], 0, None, None
    for idx, p in rows:
        visited += 1; first = idx if first is None else first; last = idx
        b, units, info = F.make(p, rng, "F1")
        if not keep(b):
            continue
        got.append({"idx": idx, **info, "n_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "b": b,
                    "units": units})
        if len(got) >= nf + nt:                                     # stop before the pool yields another row
            break
    if len(got) < nf:
        sys.exit(f"the pool ran out before {nf} fit texts ({len(got)} kept)")
    texts = {"fit": got[:nf], "test": got[nf:]}
    block = {"seed": SEED, "visited": visited, "first_idx": first, "last_idx": last,
             "pool_ended": len(got) < nf + nt}
    for split, rs in texts.items():
        block[split] = [{k: v for k, v in r.items() if k not in ("b", "units")} for r in rs]
    return block, texts


def rebuild(tok):
    """The texts again, checked against the committed block file (indices, IDs and text hashes). Outside smoke runs the
    block file must be committed, unmodified, and written under the same filing commit."""
    stored = json.load(open(out_path("blocks")))
    if not SMOKE:
        rel = os.path.relpath(out_path("blocks"), _seg.HERE)
        if F.git("ls-files", "--error-unmatch", rel).returncode or F.git("diff", "--quiet", "HEAD", "--", rel).returncode:
            sys.exit(f"refused: {rel} must be committed, unmodified, before the score job")
        if stored.get("code", {}).get("filed_commit") != provenance()["filed_commit"]:
            sys.exit("refused: the block file was written under another filing commit")
    block, texts = take_texts(pool_rows(), tok)
    for split in ("fit", "test"):
        assert block[split] == stored["block"][split], f"{split} differs from the committed block file"
    return texts


# ---------------------------------------------------------------- the guard, provenance and the job gate
def code_paths():
    """This script, the filed O6b code and every repository module the process has loaded"""
    return sorted(set(F.code_paths()) | {os.path.relpath(os.path.realpath(__file__), os.path.realpath(_seg.HERE))})


def filed_headings(text):
    """O6c's filing heading lines: FILED_LINE matches outside code fences and HTML comments, dated FILED_SINCE or
    later (as the filed O6b guard reads its own heading)"""
    text, keep, fence = re.sub(r"<!--.*?-->", "", text, flags=re.S), [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        elif not fence:
            keep.append(line)
    out = []
    for m in FILED_LINE.finditer("\n".join(keep)):
        try:
            if datetime.date.fromisoformat(m.group(1)) >= FILED_SINCE:
                out.append(m.group(0))
        except ValueError:
            pass
    return out


def filed_check(filed):
    """(commit, None) if `filed` names an O6c filing commit, else (None, why). As the filed O6b guard, a filing commit:
    - is an ancestor of HEAD and has one parent;
    - introduced an O6c filing heading line, one its parent's O6c rules file lacks;
    - changes no covered file and no submodule against its parent: it changes the text only;
    - holds the covered files and the submodule commits unchanged from the code commit its Code section pins, with
      the filed O6b code unchanged from O6B_PIN;
    and the working tree's covered files are unchanged since (files in a submodule against the commit it records)."""
    c = F.git("rev-parse", "--verify", "--quiet", f"{filed}^{{commit}}").stdout.strip() if filed else ""
    if not c:
        return None, f"O6C_FILED={filed!r} does not name a commit"
    if F.git("merge-base", "--is-ancestor", c, "HEAD").returncode != 0:
        return None, f"{c} is not an ancestor of HEAD"
    parents = F.git("rev-list", "--parents", "-n", "1", c).stdout.split()[1:]
    if len(parents) != 1:
        return None, f"{c} has {len(parents)} parents; a filing commit has one"
    text = F.git("show", f"{c}:{RULES}").stdout
    new = collections.Counter(filed_headings(text))
    new -= collections.Counter(filed_headings(F.git("show", f"{parents[0]}:{RULES}").stdout))
    if not new:
        return None, (f"{c} introduced no heading line '## O6c confirmatory rule, filed (YYYY-MM-DD)' (outside code "
                      f"fences and comments, dated {FILED_SINCE} or later)")
    repos = F.by_repo(c, code_paths())
    top = repos.get("", (c, []))[1] + [sub for sub in repos if sub]
    if F.git("diff", "--quiet", parents[0], c, "--", *top).returncode != 0:
        return None, f"{c} changes code against its parent; a filing commit changes the text only"
    m = PIN.search(text, max(0, text.rfind("\n### Code\n")))
    pin = F.git("rev-parse", "--verify", "--quiet", f"{m.group(1)}^{{commit}}").stdout.strip() if m else ""
    if not pin:
        return None, f"the O6c rules file at {c} pins no code commit ('realblt_copy_latch_cutoff.py, at commit ...')"
    if F.git("diff", "--quiet", pin, c, "--", *top).returncode != 0:
        return None, f"the code at {c} differs from the code commit its rules file pins, {pin}"
    if F.git("diff", "--quiet", O6B_PIN, c, "--", O6B_CODE).returncode != 0:
        return None, f"{O6B_CODE} at {c} differs from O6b's pin, {O6B_PIN}"
    for sub, (rev, ps) in repos.items():
        missing = set(ps) - set(F.git("ls-tree", "-r", "--name-only", rev, "--", *ps, sub=sub).stdout.split())
        if missing or F.git("diff", "--quiet", rev, "--", *ps, sub=sub).returncode != 0:
            where = f"submodule {sub} at {rev}" if sub else c
            return None, f"the code differs from {where}: {sorted(missing) or 'git diff ' + rev}"
    return c, None


PROVENANCE = {}


def provenance():
    """What every output records: HEAD, each code file's sha256, the submodules' commits, the O6c rules file's sha256,
    the threshold mode, the model record and the filing commit. Outside smoke runs the filing is checked once per
    process, and nothing is recorded, read or scored unless it passes."""
    if not PROVENANCE:
        if sys.flags.optimize:
            sys.exit("refused: python -O (or PYTHONOPTIMIZE) drops the rule's asserts; run without it")
        if blt_load.WINDOW != 512:
            sys.exit("refused: SEGR_BLT_WINDOW is set; the score job switches the windows off itself, after this check")
        if blt_layouts.MODE != "train":
            sys.exit(f"refused: thresholds must be fitted in train mode (blt_layouts.MODE is {blt_layouts.MODE!r})")
        paths = code_paths()
        rec = {"head": F.git("rev-parse", "HEAD").stdout.strip(), "filed_commit": None,
               "files_sha256": {f: F.sha256(os.path.join(_seg.HERE, f)) for f in paths},
               "submodules": {s: F.git("rev-parse", "HEAD", sub=s).stdout.strip() for s in F.by_repo("HEAD", paths) if s},
               "script_sha256": F.sha256(os.path.realpath(__file__)),
               "rules_sha256": F.sha256(os.path.join(_seg.HERE, RULES)), "threshold_mode": blt_layouts.MODE,
               **F.model_record()}
        if not SMOKE:
            c, why = filed_check(os.environ.get("O6C_FILED", ""))
            if why:
                sys.exit(f"refused: {why}. No new problem was read or scored.")
            rec["filed_commit"] = c
        PROVENANCE.update(rec)
    return PROVENANCE


def read_out(name):
    """A job's output, or None. Outside smoke runs only an output written under the current filing commit counts."""
    if not os.path.exists(out_path(name)):
        return None
    x = json.load(open(out_path(name)))
    return x if SMOKE or x.get("code", {}).get("filed_commit") == provenance()["filed_commit"] else None


def failures(name):
    """How many committed versions of a job's output, in the history behind HEAD, record a failed implementation check"""
    rel = os.path.relpath(out_path(name), _seg.HERE)
    n = 0
    for c in F.git("log", "--format=%H", "HEAD", "--", rel).stdout.split():
        try:
            x = json.loads(F.git("show", f"{c}:{rel}").stdout)
        except ValueError:
            continue
        n += any("implementation check failed" in s for s in x.get("stop", []))
    return n


def gate(name):
    """Why command `name` may not run now, or None (as the filed O6b gate). Smoke runs are not gated. It fetches first;
    each job runs once per filing (its output may not exist under the filing, and no commit after the filing commit,
    on HEAD or its upstream, may have touched its output file); the score job refuses after two failed checks; every
    output that exists is committed and unmodified, none tracked is missing, and HEAD is pushed and not behind its
    upstream; the score job needs the block file and `verdict` the score output, both under this filing."""
    if SMOKE:
        return None
    code = provenance()
    if F.git("fetch", "--quiet").returncode != 0:
        return "git fetch failed: the gate needs the remote's history"
    if read_out(name) is not None:
        return f"{name} already ran under the filing commit {code['filed_commit']}; a redo needs a re-filing commit"
    rel = os.path.relpath(out_path(name), _seg.HERE)
    for span in (f"{code['filed_commit']}..HEAD", f"{code['filed_commit']}..@{{upstream}}"):
        if F.git("log", "--format=%H", span, "--", rel).stdout.strip():
            return f"a commit in {span} touched {rel}: {name} already ran under this filing"
    if name == "score" and failures("score") >= 2:
        return "the score job's implementation checks failed twice: the test has stopped"
    for o in OUTPUTS:
        r = os.path.relpath(out_path(o), _seg.HERE)
        tracked = F.git("ls-files", "--error-unmatch", r).returncode == 0
        if tracked and not os.path.exists(out_path(o)):
            return f"{r} is tracked but missing from disk"
        if os.path.exists(out_path(o)) and (not tracked or F.git("diff", "--quiet", "HEAD", "--", r).returncode):
            return f"{r} must be committed and unmodified before the next job"
    if F.git("rev-parse", "--verify", "--quiet", "@{upstream}").returncode != 0:
        return "HEAD has no upstream branch: set one, and push every output before the next job"
    if F.git("rev-list", "--count", "@{upstream}..HEAD").stdout.strip() != "0":
        return "HEAD has commits that are not pushed: push every output before the next job"
    if F.git("rev-list", "--count", "HEAD..@{upstream}").stdout.strip() != "0":
        return "HEAD is behind its upstream: bring it up to date before the next job"
    if name == "score" and read_out("blocks") is None:
        return "the block file is missing under this filing: run blocks first"
    if name == "verdict" and read_out("score") is None:
        return "the score job has not run under this filing"
    return None


# ---------------------------------------------------------------- jobs
def cmd_blocks():
    why = gate("blocks")
    if why:
        sys.exit(f"refused: {why}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    block, texts = take_texts(pool_rows(), tok)
    out = {"status": STATUS, "pool": "dev, GSM8K train idx 0-299 (smoke)" if SMOKE else f"GSM8K train idx >= {NEW0}",
           "code": provenance(), "block": block,
           "n_tokens": {s: [r["n_bytes"] + 1 for r in rs] for s, rs in texts.items()}}
    if not SMOKE:                                                   # after filing only: the pool file as read
        with open(GSM_TRAIN, "rb") as f:
            out["pool_file"] = {"path": GSM_TRAIN, "sha256": F.sha256(GSM_TRAIN), "lines": sum(1 for _ in f)}
    json.dump(out, open(out_path("blocks"), "w"), indent=1)
    print(json.dumps({k: v for k, v in block.items() if k not in ("fit", "test")} |
                     {"fit": len(block["fit"]), "test": len(block["test"])}, indent=1))
    print("wrote", out_path("blocks"), "- commit and push it before the score job")


def discordance(rows, a, b):
    """The texts where layouts a and b differ on id_copy exact, for the copies a lost and those it kept: how many, how
    many with L8 firing at c1 and with b starting a patch at c1, those crossed, and the texts' problem indices"""
    out = {"lost": collections.Counter(), "kept": collections.Counter()}
    joint, idx = collections.Counter(), {"lost": [], "kept": []}
    for r in rows:
        x, y = (r["layouts"][n]["kinds"]["id_copy"][1] for n in (a, b))
        if x != y:
            d = "kept" if x > y else "lost"
            side = out[d]
            side["copies"] += 1
            side["L8 fired at c1" if r["fires_c1"]["L8"] else "L8 did not fire at c1"] += 1
            side[f"{b} starts at c1" if r["starts_c1"][b] else f"{b} does not start at c1"] += 1
            joint[f"{a} {d} the copy; L8 {'fires' if r['fires_c1']['L8'] else 'does not fire'} at c1; {b} "
                  f"{'starts' if r['starts_c1'][b] else 'does not start'} at c1"] += 1
            idx[d].append(r["idx"])
    return {**{k: dict(v) for k, v in out.items()}, "joint": dict(joint), "texts (problem idx)": idx}


def by_length(rows):
    """Exact copies per layout and L8's firing at c1, by text length (O6b's F5 bins) and by the copy's position"""
    out = {}
    groups = [(f"tokens {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r["n_tokens"] < hi) for lo, hi in BINS]
    groups += [(f"copy starts before token {CUT}", lambda r: r["c1"] < CUT),
               (f"copy starts at token {CUT} or later", lambda r: r["c1"] >= CUT)]
    for name, keep in groups:
        sub = [r for r in rows if keep(r)]
        if sub:
            out[name] = {"n": len(sub), "L8_fires_c1": float(np.mean([r["fires_c1"]["L8"] for r in sub])),
                         **{n: float(np.mean([r["layouts"][n]["kinds"]["id_copy"][1] for r in sub])) for n in NAMES}}
    return out


def score(C, model, tok, dev):
    """D, D-all, D.L8, D.SIVPO and D.SIVPO620 on the test texts, thresholds refitted on the fit texts (as F.score_cell,
    with the cutoff layout added)"""
    fit_rows, test = C["fit"], C["test"]
    fit, r_d = F.cell_fit(fit_rows, MASKS)
    thr = {f"{kd}@{r_d:.4f}": fit._threshold(kd, r_d) for kd in ("entropy", "entl8", "entsivpo", "entsivpo620")}
    print(f"{len(fit_rows)} fit / {len(test)} test; D's train rate {r_d:.4f}", flush=True)
    rows = []
    t0 = F.time.time()
    for i, r in enumerate(test):
        ids = tok(r["b"].decode(), return_tensors="pt").input_ids.to(dev)
        lay = layouts(r, fit, r_d)
        sc = F.scorer(model, dev, ids, r["units"])
        c = F.copy_tokens(r)
        rows.append({"idx": r["idx"], "layouts": {n: sc(lay[n]) for n in NAMES}, "n_tokens": len(r["b"]) + 1,
                     "c1": int(c[0]), "copy_len": len(c), "distance": r.get("distance"),
                     "fires_c1": {m: bool(r[m][c[0]]) for m in ("L8", "SIV", "PO", "SIVPO", "SIVPO620")},
                     "starts_c1": {n: bool(int(c[0]) in set(lay[n].tolist())) for n in NAMES}})
        if (i + 1) % 50 == 0 or SMOKE:
            print(f"  {i + 1} texts, {F.time.time() - t0:.0f}s", flush=True)
    res = {"train_rate_default": r_d, "thresholds_nats": thr, **F.summarize(rows, NAMES, PAIRS)}
    res["fires_c1"] = {m: float(np.mean([r["fires_c1"][m] for r in rows])) for m in rows[0]["fires_c1"]} if rows else {}
    res["by_length"] = by_length(rows)
    res["discordance"] = {PRIMARY: discordance(rows, "D.L8", "D.SIVPO620"),
                          "D.L8 vs D.SIVPO": discordance(rows, "D.L8", "D.SIVPO")}
    return res, rows


def cmd_score():
    """C1: the released conversion (window 0) on the new texts: the patcher, the implementation checks, then the five
    layouts. A failed check stops the test before any scoring; a redo needs a re-filing the PI approves."""
    L.alone()
    why = gate("score")
    if why:
        sys.exit(f"refused: {why}")
    code = provenance()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    C = rebuild(tok)
    blt_load.WINDOW = 0                                             # as SEGR_BLT_WINDOW=0 sets it; load() reads it
    model = blt_light.load(MODEL, dev).eval()
    if not (model.model.patcher.config.sliding_window is None and model.config.local_attention_window_len is None):
        sys.exit("refused: the released conversion's model still has windows")
    L.patcher_pass(C, tok, model, dev)
    F.save(C, out_path("patcher", "npz"))
    for rows in C.values():
        add_masks(rows)                                             # a failed subset assert ends the job here
    chk = F.checks(C)
    out = {"status": STATUS, "code": code, "window": 0, "patcher_sha256": F.sha256(out_path("patcher", "npz")),
           "checks": chk}
    if not chk["passed"] and not SMOKE:                             # a bug, not a result: nothing is scored
        out["stop"] = ["implementation check failed (window 0)" + (" again (second failure)" if failures("score")
                                                                   else "")]
        json.dump(out, open(out_path("score"), "w"), indent=1)
        sys.exit("O6c: an implementation check failed under window 0; nothing was scored. Commit and push this output. "
                 "A redo needs a re-filing the PI approves. " + json.dumps(chk))
    res, rows = score(C, model, tok, dev)
    out.update(res)
    out["stop"] = []
    json.dump({**out, "rows": rows}, open(out_path("score"), "w"), indent=1)
    print(json.dumps({k: (round(v["id_copy"]["exact"][0] * 100, 1), [round(x * 100, 1) for x in v["id_copy"]["exact"][1]])
                      for k, v in res["pairs"].items() if "id_copy" in v}, indent=1))
    print("wrote", out_path("score"), "- commit and push it before `verdict`")


# ---------------------------------------------------------------- the verdict
READINGS = {     # pre-registered (followup/O6c_CUTOFF_RULES.md, Readings), quoted whole
    "tie": ("Under the released conversion, a byte rule that stops at token 620, a position fixed from dev texts, is "
            "not classed as differing from L8 (a class needs a difference of at least 2 points with its 95% interval "
            "on that side of 0; this is not an equivalence test). L8's model-reading gate shows no advantage over a "
            "fixed position cutoff on this template. With O6b's windowed result, no setting tested shows that reading "
            "the model is needed."),
    "L8 better": ("L8's gate helps beyond a byte rule that stops at token 620, a position fixed from dev texts, under "
                  "the released conversion, on this template; reading the model earns something there."),
    "the cutoff better": ("A byte rule that stops at token 620, a position fixed from dev texts, beats L8 under the "
                          "released conversion, on this template."),
}
STOPPED = "The implementation failed under window 0; nothing was tested."
UNDERPOWERED = "Fewer than {min} test texts ({n}): the numbers are reported with no class."


def pts(v):
    f = lambda x: "0.00" if x == 0 else f"{x * 100:+.2f}"
    return f"{f(v[0])} [{f(v[1][0])}, {f(v[1][1])}]"


def verdict_from(S):
    """O6c's class, its reading and the rows reported beside it, from the score output S"""
    if S is None:
        return {"class": "not reached", "reading": "not reached: the score job has not run", "sentences": []}
    if S.get("stop"):
        return {"class": "not reached", "reading": STOPPED, "stop": S["stop"], "sentences": [STOPPED]}
    n = S["n"]
    prim = F.pt(S["pairs"][PRIMARY])
    cls = F.classify(prim, F.MARGIN["exact"], "the cutoff") if n >= MIN_TEST else "no class"
    reading = READINGS[cls] if cls in READINGS else UNDERPOWERED.format(min=MIN_TEST, n=n)
    d = S["discordance"][PRIMARY]
    lost = d.get("lost", {})
    split = {"n": lost.get("copies", 0), "a": lost.get("L8 did not fire at c1", 0)}
    split["b"] = split["n"] - split["a"]
    vs_sivpo = F.pt(S["pairs"]["D.L8 vs D.SIVPO"])
    vs_cls = F.classify(vs_sivpo, F.MARGIN["exact"]) if n >= MIN_TEST else "no class"
    beside = {"D.L8 - D.SIVPO (O6b's released row, replicated)": {"diff": vs_sivpo, "class": vs_cls},
              "D-all - D": F.pt(S["pairs"]["D-all vs D"]),
              "D.SIVPO620 - D": F.pt(S["pairs"]["D.SIVPO620 vs D"]),
              "by length": S["by_length"], "discordant split (primary)": split, "discordant texts (primary)": d,
              "discordant texts (D.L8 vs D.SIVPO)": S["discordance"].get("D.L8 vs D.SIVPO")}
    sentences = [reading,
                 f"Primary: D.L8 - D.SIVPO<620 on exact copies, {pts(prim)} points on {n} texts (class: {cls}; at "
                 f"least 2 points with the 95% interval on that side of 0, or a tie).",
                 f"Beside it: D.L8 - D.SIVPO {pts(vs_sivpo)} points (class under O6b's rule: {vs_cls}); D-all - D "
                 f"{pts(beside['D-all - D'])} points; D.SIVPO<620 - D {pts(beside['D.SIVPO620 - D'])} points; of "
                 f"the {split['n']} copies L8 lost to the cutoff, L8 did not fire at c1 on {split['a']} and fired on "
                 f"{split['b']}."]
    return {"class": cls, "reading": reading, "primary": prim, "n": n, "beside": beside, "sentences": sentences}


def cmd_verdict():
    why = gate("verdict")
    if why:
        sys.exit(f"refused: {why}")
    S = read_out("score")
    code = provenance()
    V = {"status": STATUS, "code": code, **verdict_from(S)}
    if not SMOKE:
        V["rules file changes since filing"] = F.git("diff", "--stat", code["filed_commit"], "HEAD", "--", RULES).stdout
    json.dump(V, open(out_path("verdict"), "w"), indent=1, default=str)
    print(json.dumps(V, indent=1, default=str))
    print(f"O6c result ({'filing ' + code['filed_commit'] if code['filed_commit'] else 'smoke run, dev problems'})")
    print("\n".join(V["sentences"]))
    print("wrote", out_path("verdict"))


# ---------------------------------------------------------------- self-test (bytes only, dev problems; no model)
def guard_selftest():
    """The O6c guard in a scratch repository: it accepts a filing commit whose code is the pinned code, with the O6b
    code at its own pin, and refuses the rest"""
    global code_paths, O6B_PIN
    here, paths_of, o6b_pin = _seg.HERE, code_paths, O6B_PIN
    with tempfile.TemporaryDirectory() as tmp:
        sh = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, cwd=tmp,
                                       capture_output=True, text=True).stdout.strip()
        put = lambda path, text: open(os.path.join(tmp, path), "w").write(text)

        def commit(rules=None):
            if rules is not None:
                put(RULES, rules)
            sh("add", "-A"); sh("commit", "-qm", "c")
            return sh("rev-parse", "HEAD")
        sh("init", "-q", "-b", "main"); os.makedirs(os.path.join(tmp, "followup"))
        put("code.py", "x = 1\n"); put(O6B_CODE, "o6b = 1\n")
        pin_b = commit("# O6c rules\n")
        pinned = f"# O6c rules\n### Code\n- realblt_copy_latch_cutoff.py, at commit {pin_b}\n"
        heading = "## O6c confirmatory rule, filed (2026-11-02)\n"
        got = {"no heading": commit(pinned)}
        got["O6b's heading"] = commit(pinned + "## Confirmatory rule, filed (2026-11-02)\n")
        commit(pinned)
        got["dated before 2026-10-31"] = commit(pinned + "## O6c confirmatory rule, filed (2026-10-30)\n")
        commit(pinned)
        put("code.py", "x = 9\n"); got["a filing commit that also edits code"] = commit(pinned + heading)
        put("code.py", "x = 1\n"); commit(pinned)
        filed = commit(pinned + heading + "Approved: the PI's words.\n")
        try:
            _seg.HERE, code_paths, O6B_PIN = tmp, (lambda: ["code.py", O6B_CODE]), pin_b
            refused = {k: filed_check(c)[1] for k, c in got.items()}
            ok = filed_check(filed)
            put("code.py", "x = 2\n"); refused["code changed after filing"] = filed_check(filed)[1]
            put("code.py", "x = 1\n")
            put(O6B_CODE, "o6b = 2\n"); fix = commit(pinned)
            refused["the O6b code changed from its pin"] = filed_check(commit(pinned.replace(pin_b, fix) + heading))[1]
            put(O6B_CODE, "o6b = 1\n"); commit(pinned)
            refused["a pin that is not its code"] = filed_check(commit(pinned.replace(pin_b, fix) + heading))[1]
        finally:
            _seg.HERE, code_paths, O6B_PIN = here, paths_of, o6b_pin
    assert ok == (filed, None) and all(refused.values()), (ok, refused)
    return {"accepts": "the filing commit", "refuses": refused}


def cmd_selftest():
    tok = AutoTokenizer.from_pretrained(MODEL)
    res = {}
    # 1. the builder on dev problems (train idx 0-299): deterministic, F1's template (build itself), the ID twice, and
    #    the copy's first token 34-40 tokens before the end of the text
    rows = list(itertools.islice(((i, json.loads(l)) for i, l in enumerate(open(GSM_TRAIN))), 300))
    b1, t1 = take_texts(rows, tok, (100, 150))
    b2, t2 = take_texts(rows, tok, (100, 150))
    assert b1 == b2 and len(b1["fit"]) == 100 and len(b1["test"]) == 150
    texts = t1["fit"] + t1["test"]
    tails = set()
    for r in texts:
        (a0, e0), (a1, e1) = r["units"]["id_new"][0], r["units"]["id_copy"][0]
        assert r["b"][a0:e0] == r["b"][a1:e1] == r["id"].encode() and r["b"].count(r["id"].encode()) == 2
        tails.add(len(r["b"]) + 1 - int(F.copy_tokens(r)[0]))
    res["builder"] = {"dev texts": len(texts), "visited": b1["visited"], "deterministic": True,
                      "copy's first token before the end (tokens)": [min(tails), max(tails)]}
    # 2. the cutoff mask: SIV+PO below token 620, nothing from 620 on; equal to SIV+PO on texts of 620 tokens or fewer
    long_n = 0
    for r in texts:
        m = F.masks_bytes(r["b"])["SIVPO"]
        c = cut_mask(m)
        n = len(r["b"]) + 1
        assert (c[:CUT] == m[:CUT]).all() and not c[CUT:].any()
        assert (c == m).all() or n > CUT
        long_n += n > CUT
    joined = texts[0]["b"] + b"\n" + texts[1]["b"] + b"\n" + texts[2]["b"]
    m = F.masks_bytes(joined)["SIVPO"]
    assert len(joined) + 1 > CUT and m[CUT:].any() and not cut_mask(m)[CUT:].any()
    res["cutoff mask"] = (f"{len(texts)} dev texts ({long_n} longer than {CUT} tokens) and one joined text of "
                          f"{len(joined) + 1} tokens: SIV+PO below {CUT}, nothing from {CUT} on")
    # 3. the cutoff rule is causal: token t reads bytes up to t-2 only, so a prefix of p bytes fixes the mask at tokens
    #    0..p+1 (SIV+PO's own functions, as masks_bytes calls them), and there it equals the full text's mask
    k = 0
    for r in texts[:20] + [{"b": joined}]:
        full = cut_mask(F.masks_bytes(r["b"])["SIVPO"])
        for cut in sorted({len(r["b"]) // 3, len(r["b"]) // 2, min(len(r["b"]) - 1, CUT - 1), len(r["b"]) - 1}):
            pre = cut_mask(L.final_rules.siv(r["b"][:cut], cut + 2, kmin=5) | L.final_rules.po(r["b"][:cut], cut + 2))
            assert (pre == full[:cut + 2]).all(), (r.get("idx"), cut)
            k += 1
    res["causal"] = f"{k} prefixes of 21 texts: a p-byte prefix's mask equals the full text's mask at tokens 0..p+1"
    # 4. the layout: with stand-in patcher outputs, D.SIVPO620 equals D.SIVPO where no fit or scored text reaches token
    #    620, and from 620 on it can start a patch where SIV+PO's mask holds
    rng = np.random.default_rng(0)

    def stand_in(b):
        n = len(b) + 1
        ent = rng.gamma(2.0, 0.6, n).astype(np.float32)
        return {"b": b, "units": {"id_copy": []}, "ent": ent, "arg": np.zeros(n, np.int16), "sur": np.zeros(n - 1),
                "D": np.flatnonzero(ent > 1.6)}
    short = [stand_in(r["b"]) for r in texts if len(r["b"]) + 1 <= CUT][:40]
    add_masks(short)
    fit, r_d = F.cell_fit(short[:30], MASKS)
    for r in short[30:]:
        lay = layouts(r, fit, r_d)
        assert np.array_equal(lay["D.SIVPO620"], lay["D.SIVPO"])
    big = stand_in(joined)
    add_masks([big])
    lay = layouts(big, fit, r_d)
    after = [t for t in lay["D.SIVPO620"].tolist() if t >= CUT and big["SIVPO"][t]]
    assert not [t for t in lay["D.SIVPO"].tolist() if t >= 2 and big["SIVPO"][t]] and after
    res["layout"] = (f"short texts: D.SIVPO620 = D.SIVPO on {len(short) - 30}; a {len(joined) + 1}-token text: "
                     f"{len(after)} starts from token {CUT} on where SIV+PO's mask holds")
    # 5. the guard: none of these may pass here, the O6c rules file has no filing heading yet, and the scratch
    #    repository test
    assert filed_headings("x\n## O6c confirmatory rule, filed (2026-11-02)\ny")
    for text in ("## Confirmatory rule, filed (2026-11-02)\n", "```\n## O6c confirmatory rule, filed (2026-11-02)\n```\n",
                 "<!--\n## O6c confirmatory rule, filed (2026-11-02)\n-->\n", "## O6c confirmatory rule, filed (2026-10-30)\n"):
        assert not filed_headings(text), text
    assert not filed_headings(open(os.path.join(_seg.HERE, RULES)).read())
    refused = {x: filed_check(x)[1] for x in ("", "HEAD", "HEAD^{tree}", ":0", "no-such-commit")}
    assert all(refused.values()), refused
    res["guard refuses here"] = refused
    res["guard covers"] = code_paths()
    res["guard in a scratch repository"] = guard_selftest()
    res["model"] = F.model_record()
    print(json.dumps(res, indent=1))
    json.dump({"status": "self-test of the O6c code on dev problems (bytes only; no model)", **res},
              open(os.path.join(F.RES, "blt_copy_latch_cutoff_selftest.json"), "w"), indent=1)
    print("self-test passed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        cmd_selftest()
    elif cmd == "blocks":
        cmd_blocks()
    elif cmd == "score":
        cmd_score()
    elif cmd == "verdict":
        cmd_verdict()
    else:
        sys.exit(__doc__)
