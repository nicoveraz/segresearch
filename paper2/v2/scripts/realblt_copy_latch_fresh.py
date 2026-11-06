"""O6b confirmatory test of L8 on fresh GSM8K problems (followup/O6b_COPY_LATCH_RULES.md, section "Confirmatory
rule, proposed for filing"). PROPOSED, NOT APPROVED: every command that reads or scores the fresh pool (GSM8K train
idx >= 1800) refuses to run unless O6B_FILED names a filing commit (filed_check): an ancestor of HEAD that introduced
a heading line FILED_LINE below, outside code fences and HTML comments and dated 2026-10-30 or later, and from which
this script and every repository module it imports are unchanged. Each job then runs once per filing, after the jobs
before it, whose outputs must be committed and pushed, and with the code, the model snapshot and the libraries J1
recorded (gate). Every output records the filing commit, HEAD, each code file's hash, the model snapshot, the library
versions and the device. Nothing here goes into paper2v2.

The rule (onset, latch), its causality check, V_8, SIV, PO and the bootstrap are imported unchanged: from
realblt_copy_latch.py (83d48e3), realblt_copy_veto.py and analysis/2026-10-30_copyfix/bytes/final_rules.py. Added
here: the cell templates (F1-F6), SIV+PO (the bytes designer's design 2: siv OR po), the whole-number new-value test,
a scorer for copies of any length, and the verdict code.

    PY="../segresearch/.venv-blt/bin/python -B"
    $PY realblt_copy_latch_fresh.py selftest   builders and counters on dev problems (bytes only; no model)
    $PY realblt_copy_latch_fresh.py devcheck   exploratory, no model: dev recount, byte rules on dev rewrites, xdomain
    O6B_FILED=<commit> $PY realblt_copy_latch_fresh.py blocks    J0: block indices and byte-level counts (no model)
    O6B_FILED=<commit> $PY realblt_copy_latch_fresh.py j1        patcher on every fresh text; checks; R1, R2; C4, K2
    O6B_FILED=<commit> $PY realblt_copy_latch_fresh.py j2|j3|j4|j5|j6|j7|verdict
                     F1 | F6 | F2a+F2b | F3 | F4 | F5 (the conversion as released) | every bar and kill rule
    O6_SMOKE=1 $PY realblt_copy_latch_fresh.py blocks|j1..j7|verdict   10 fit + 10 test dev problems (train idx
                     0-299) per template -> *_smoke.*: a code check only, never used for a verdict or a prediction
"""
import sys

sys.dont_write_bytecode = True                  # importing the rule modules must not write bytecode into the repo

import collections
import copy
import datetime
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
import time
import types

os.environ["HF_HUB_OFFLINE"] = "1"              # the model and tokenizer come from the local snapshot, never the hub

import numpy as np

import _seg
import realblt_copy_latch as L                  # the rule and its checks, unchanged; it asserts the 512-byte windows
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_light
import blt_load
import realblt_budget
from realblt_budget import GSM, GSM_TRAIN, MAX_TOKENS, MODEL, lengths_from_starts
from realblt_copy_veto import boot_ratio, ends_of, vetoed
from realblt_quadrant import KINDS, TOK, build

SMOKE = os.environ.get("O6_SMOKE") == "1"
TAG = "_smoke" if SMOKE else ""
RES = L.RES
STATUS = ("O6b confirmatory code, PROPOSED (not approved); smoke runs use dev problems and are code checks only"
          if SMOKE else "O6b confirmatory test (fresh problems)")
FILED = "## Confirmatory rule, filed"           # heading the rules file must carry at O6B_FILED, as a whole line:
FILED_LINE = re.compile(r"^## Confirmatory rule, filed \((\d{4}-\d{2}-\d{2})\)$", re.M)    # '... filed (YYYY-MM-DD)'
FILED_SINCE = datetime.date(2026, 10, 30)       # a filing heading's date may not be earlier
RULES = "followup/O6b_COPY_LATCH_RULES.md"
SNAPSHOT = "91aa6b8e168046ad91e517d2191e1e974f50cc01"   # itazap/blt-1b-hf, the snapshot dev and the smoke used

FRESH0 = 1800                                   # the fresh pool: GSM8K train idx >= 1800, in file order
CELLS = ("F1", "F2a", "F2b", "F3", "F4")        # blocks are taken in this order; F6 reuses F1's problems
SIZES = {"F1": (300, 600), "F2a": (300, 300), "F2b": (300, 300), "F3": (300, 400), "F4": (300, 200)}
SMOKE_SIZES = (10, 10)                          # 20 dev problems per template
SEEDS = {"F1": 6101, "F2a": 6102, "F2b": 6103, "F3": 6104, "F4": 6105}      # one per cell, fixed before any text
F3_KINDS = ("digits6", "letters4", "mixed", "dose")                       # cycled over each cell list's texts
DOSES = ("5", "10", "20", "25", "40", "50", "100", "200", "250", "325", "500", "1000")   # F3's 12 doses, mg
F4_DIST = (550, 950)                            # bytes from the source ID's first byte to the copy's
C4_MAX_D = 4                                    # C4: at most 4 new-value removals of D's computed-value starts in F6
MARGIN = {"exact": 0.02, "bpb": 0.002, "bits": 0.5}     # paired verdict: smallest difference that counts
RULE_MASKS = ("L8", "V8", "SIV", "SIVPO")       # signals refitted per cell (entl8, entv8, entsiv, entsivpo)
VALUE = L.VALUE
NUMV = re.compile(rb"\$?\d[\d,]*(?:\.\d+)?")    # a value as O6's VALUE pattern reads it, '$' included


def out_path(name, ext="json"):
    return os.path.join(RES, f"blt_copy_latch_fresh_{name}{TAG}.{ext}")


# ---------------------------------------------------------------- templates
def parts(p):
    """question, solution, answer as realblt_quadrant.build and realblt_budget.problem strip them"""
    sol, ans = p["answer"].rsplit("#### ", 1)
    sol = re.sub(r"<<[^>]*>>", "", sol).strip()
    return p["question"], sol, ans.strip()


def doses_allowed(avoid):
    """The doses whose digits occur in no part of the problem (question, solution, answer) unflanked by digits. Each
    part is tested apart: joined, '$25' and '25' read '$2525' and hide the dose (train idx 71)."""
    return [d for d in DOSES if not any(re.search(rf"(?<!\d){d}(?!\d)", s) for s in avoid)]


def draw(rng, kind, avoid):
    """An ID of the given kind that does not occur in the problem, avoid = (question, solution, answer). Doses: one of
    the 12 that doses_allowed keeps, or None if it keeps none."""
    if kind == "dose":
        ok = doses_allowed(avoid)
        return ok[int(rng.integers(len(ok)))] if ok else None
    avoid = "".join(avoid)
    while True:
        if kind == "digits4":
            x = str(rng.integers(1000, 10000))
        elif kind == "digits6":
            x = str(rng.integers(100000, 1000000))
        elif kind == "letters4":
            x = "".join(chr(65 + int(c)) for c in rng.integers(0, 26, 4))
        else:                                                       # mixed, like QX7-4821
            a, b_ = (chr(65 + int(c)) for c in rng.integers(0, 26, 2))
            x = f"{a}{b_}{int(rng.integers(0, 10))}-{int(rng.integers(1000, 10000))}"
        if x not in avoid:
            return x


def label(b, s0, f0, tail, ids=()):
    """realblt_quadrant.build's unit labels (realblt_quadrant.py:66-79), with the two ID spans given as (start, end)
    bytes. TOK matches that overlap an ID get no other label, so an ID of letters or with a dash is one unit."""
    units = {k: [] for k in KINDS}
    if ids:
        units["id_new"].append(tuple(ids[0])); units["id_copy"].append(tuple(ids[1]))
    seen = set()
    for m in TOK.finditer(b):
        a, e = m.span(); w = m.group(0)
        if a >= s0 and not any(a < y and x < e for x, y in ids):
            if w[:1].isdigit():
                if a == f0:
                    units["final"].append((a, e))
                elif re.search(rb"= \$?$", b[max(0, a - 3):a]):
                    units["computed"].append((a, e))
                elif w in seen:
                    units["copied"].append((a, e))
            elif a > 0 and b[a - 1:a] == b" " and len(w) > 1 and a < tail:
                units["word_new" if w not in seen else "word_rep"].append((a, e))
        seen.add(w)
    return units


def make(p, rng, cell, kind="digits4", x=None):
    """(bytes, units, info) for one problem in a cell's template, or None if no ID can be drawn. F1 is
    realblt_quadrant.build itself; x forces the ID (dev rewrites and tests only)."""
    q, sol, ans = parts(p)
    if cell == "F1" and x is None:
        b, units = build(p, rng)
        (a0, _), (a, e) = units["id_new"][0], units["id_copy"][0]
        return b, units, {"id": b[a:e].decode(), "kind": "digits4", "distance": a - a0}
    x = x or draw(rng, kind, (q, sol, ans))
    if x is None:
        return None
    if cell == "F2a":
        src, cp = f"Ticket {x}. ", f"\nIt was logged as #{x} and closed."
    elif cell == "F2b":
        src, cp = f"Order reference {x} was opened. ", f"\nWe shipped order reference {x} today."
    elif kind == "dose":
        src, cp = f"Dose ordered: {x} mg. ", f"\nDose ordered: {x} mg given."
    else:
        src, cp = f"Ticket {x}. ", f"\nTicket {x} closed."
    pre = ""
    if cell == "F4":                                                # the source line goes before the question
        pre, src = f"Notes:\nTicket {x} opened.\n", ""
    head, xb = (pre + q + "\n").encode(), x.encode()
    B = [head, src.encode(), sol.encode(), cp.encode(), b"\nThe final answer is ", ans.encode()]
    b = b"".join(B)
    a_new = pre.encode().index(xb) if pre else len(head) + B[1].index(xb)
    tail = len(head) + len(B[1]) + len(B[2])                        # the copy line's '\n'
    a_cp = tail + B[3].index(xb)
    units = label(b, len(head), len(b) - len(B[5]), tail, ((a_new, a_new + len(xb)), (a_cp, a_cp + len(xb))))
    return b, units, {"id": x, "kind": kind, "distance": a_cp - a_new}


def make_f6(p):
    """F6: realblt_budget.problem's text (no ticket), with build's unit labels"""
    q, sol, ans = parts(p)
    head = (q + "\n").encode()
    b = head + sol.encode() + b"\nThe final answer is " + ans.encode()
    f0 = len(b) - len(ans.encode())
    return b, label(b, len(head), f0, f0 - 21)


def keeper(tok):
    """O6's filter: at most MAX_TOKENS tokens, one token per byte plus BOS"""
    return lambda b: len(b) + 1 <= MAX_TOKENS and tok(b.decode(), return_tensors="pt").input_ids.shape[1] == len(b) + 1


def git(*args, sub=""):
    """git in this repository, or in its submodule `sub`"""
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=os.path.join(_seg.HERE, sub))


def code_paths():
    """This script and every repository module the process has loaded, including modules that repository modules
    hold without registering them (realblt_copy_latch.final_rules): the code a result rests on"""
    here = os.path.realpath(_seg.HERE)
    inside = lambda f: (isinstance(f, str) and os.path.isabs(f) and f.endswith(".py")      # torch.ops' file is relative
                        and os.path.realpath(f).startswith(here + os.sep))
    mods = [m for m in list(sys.modules.values()) if inside(getattr(m, "__file__", None))]
    mods += [x for m in mods for x in vars(m).values() if isinstance(x, types.ModuleType)]
    files = [os.path.abspath(__file__)] + [getattr(m, "__file__", None) for m in mods]
    return sorted({os.path.relpath(os.path.realpath(f), here) for f in files if inside(f)})


def by_repo(rev, paths):
    """{submodule, or '' for this repository: (its commit as recorded at rev, the paths relative to it)}"""
    modes = {ln.split("\t", 1)[1]: ln.split(" ", 1)[0] for ln in git("ls-tree", rev).stdout.splitlines()}
    out = {}
    for p in paths:
        top = p.split("/", 1)[0]
        sub = top if modes.get(top) == "160000" else ""             # 160000: a submodule (segresearch)
        rec = out.setdefault(sub, (git("rev-parse", f"{rev}:{sub}").stdout.strip() if sub else rev, []))
        rec[1].append(p[len(sub) + 1:] if sub else p)
    return out


def filed_headings(text):
    """The filing heading lines of a rules file: FILED_LINE matches outside code fences and HTML comments, dated
    FILED_SINCE or later"""
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


PIN = re.compile(r"realblt_copy_latch_fresh\.py, at commit ([0-9a-f]{7,40})")   # the Code section's pin


def filed_check(filed):
    """(commit, None) if `filed` names a filing commit, else (None, why). A filing commit:
    - is an ancestor of HEAD and has one parent;
    - introduced a filing heading line, one its parent's rules file lacks;
    - changes no covered file and no submodule against its parent: it changes the text only (a fix goes in its own
      commit, which the re-filing commit then pins);
    - holds the covered files and the submodule commits unchanged from the code commit its Code section pins;
    and the working tree's covered files are unchanged since (files in a submodule against the submodule commit it
    records). After filing, the rules file changes only by dated notes appended at its end (the rule's Filing)."""
    c = git("rev-parse", "--verify", "--quiet", f"{filed}^{{commit}}").stdout.strip() if filed else ""
    if not c:
        return None, f"O6B_FILED={filed!r} does not name a commit"
    if git("merge-base", "--is-ancestor", c, "HEAD").returncode != 0:
        return None, f"{c} is not an ancestor of HEAD"
    parents = git("rev-list", "--parents", "-n", "1", c).stdout.split()[1:]
    if len(parents) != 1:
        return None, f"{c} has {len(parents)} parents; a filing commit has one"
    text = git("show", f"{c}:{RULES}").stdout
    new = collections.Counter(filed_headings(text))
    new -= collections.Counter(filed_headings(git("show", f"{parents[0]}:{RULES}").stdout))
    if not new:
        return None, (f"{c} introduced no heading line '{FILED} (YYYY-MM-DD)' (outside code fences and comments, dated "
                      f"{FILED_SINCE} or later): only a filing or re-filing commit qualifies")
    repos = by_repo(c, code_paths())
    top = repos.get("", (c, []))[1] + [sub for sub in repos if sub]        # covered files here, and the submodules
    if git("diff", "--quiet", parents[0], c, "--", *top).returncode != 0:
        return None, f"{c} changes code against its parent; a filing commit changes the text only"
    m = PIN.search(text, max(0, text.rfind("\n### Code\n")))
    pin = git("rev-parse", "--verify", "--quiet", f"{m.group(1)}^{{commit}}").stdout.strip() if m else ""
    if not pin:
        return None, f"the rules file at {c} pins no code commit ('realblt_copy_latch_fresh.py, at commit ...')"
    if git("diff", "--quiet", pin, c, "--", *top).returncode != 0:
        return None, f"the code at {c} differs from the code commit its rules file pins, {pin}"
    for sub, (rev, ps) in repos.items():
        missing = set(ps) - set(git("ls-tree", "-r", "--name-only", rev, "--", *ps, sub=sub).stdout.split())
        if missing or git("diff", "--quiet", rev, "--", *ps, sub=sub).returncode != 0:
            where = f"submodule {sub} at {rev}" if sub else c
            return None, f"the code differs from {where}: {sorted(missing) or 'git diff ' + rev}"
    return c, None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_record():
    """The model a job loads: MODEL at SNAPSHOT from the local cache, offline, with the weights' blob name, the library
    versions and the patched modeling_blt.py's hash. Refuses another snapshot, or a hub that is not offline."""
    from huggingface_hub import constants, try_to_load_from_cache
    import transformers
    import transformers.models.blt.modeling_blt as mb
    path = try_to_load_from_cache(MODEL, "model.safetensors")
    if not (isinstance(path, str) and f"/snapshots/{SNAPSHOT}/" in path and os.environ.get("HF_HUB_OFFLINE") == "1"
            and constants.HF_HUB_OFFLINE):
        sys.exit(f"refused: {MODEL} must load offline from snapshot {SNAPSHOT} (found {path})")
    blob = os.path.basename(os.readlink(path) if os.path.islink(path) else path)    # the file's LFS sha256
    return {"model": MODEL, "snapshot": SNAPSHOT, "weights_blob": blob,
            "transformers": transformers.__version__, "torch": torch.__version__, "numpy": np.__version__,
            "modeling_blt_sha256": sha256(mb.__file__), "device": "mps" if torch.backends.mps.is_available() else "cpu"}


PROVENANCE = {}


def provenance():
    """What every output records: HEAD, each code file's sha256, the submodules' commits, the rules file's sha256, the
    model record and the filing commit. Outside smoke runs the filing is checked (filed_check) once per process, and
    nothing is recorded, read or scored unless it passes. Every job but J7, which switches them off in its own process,
    runs with the 512-byte windows."""
    if not PROVENANCE:
        if sys.flags.optimize:
            sys.exit("refused: python -O (or PYTHONOPTIMIZE) drops the rule's asserts; run without it")
        if blt_load.WINDOW != 512:
            sys.exit("refused: the 512-byte windows are off (SEGR_BLT_WINDOW); only J7 switches them off, itself")
        if blt_layouts.MODE != "train":
            sys.exit(f"refused: thresholds must be fitted in train mode (blt_layouts.MODE is {blt_layouts.MODE!r}; "
                     "SEGR_BLT_THRESH)")
        paths = code_paths()
        rec = {"head": git("rev-parse", "HEAD").stdout.strip(), "filed_commit": None,
               "files_sha256": {f: sha256(os.path.join(_seg.HERE, f)) for f in paths},
               "submodules": {s: git("rev-parse", "HEAD", sub=s).stdout.strip() for s in by_repo("HEAD", paths) if s},
               "script_sha256": sha256(os.path.abspath(__file__)),
               "rules_sha256": sha256(os.path.join(_seg.HERE, RULES)), "window": blt_load.WINDOW,
               "threshold_mode": blt_layouts.MODE, **model_record()}
        if not SMOKE:
            c, why = filed_check(os.environ.get("O6B_FILED", ""))
            if why:
                sys.exit(f"refused: {why}. No fresh problem was read or scored.")
            rec["filed_commit"] = c
        PROVENANCE.update(rec)
    return PROVENANCE


SAME_AS_J1 = ("filed_commit", "script_sha256", "files_sha256", "snapshot", "weights_blob", "transformers", "torch",
              "numpy", "modeling_blt_sha256")
OUTPUTS = ("blocks", "j1", "j2", "j3", "j4", "j5", "j6", "j7", "verdict")


def failures(name):
    """How many committed versions of a job's output, in the history behind HEAD, record a failed implementation check"""
    rel = os.path.relpath(out_path(name), _seg.HERE)
    n = 0
    for c in git("log", "--format=%H", "HEAD", "--", rel).stdout.split():
        try:
            x = json.loads(git("show", f"{c}:{rel}").stdout)
        except ValueError:                                          # the file was deleted at that commit
            continue
        n += any("implementation check failed" in s for s in x.get("stop", []))
    return n


def gate(name):
    """Why command `name` (an OUTPUTS entry) may not run now, or None. Smoke runs are not gated.
    - It fetches first, so it sees the remote's history.
    - It runs once per filing: its output may not exist under the current filing commit, and no commit after the
      filing commit, on HEAD or on its upstream, may have touched its output file (a redo needs a re-filing commit,
      under which J0 and every job run again). J1 and J7 also refuse once their checks have failed twice.
    - Every job output that exists is committed and unmodified, none tracked is missing from disk, and HEAD is pushed
      and not behind its upstream.
    - Every command after J1 runs with the filing, code, model snapshot and libraries J1 recorded, and J2-J7 on the
      patcher outputs J1 wrote. J1's stops block J2-J7, and K5 blocks J3-J7 (blocked). `verdict` needs J2-J7 under
      this filing unless a stop applies."""
    if SMOKE:
        return None
    code = provenance()
    if git("fetch", "--quiet").returncode != 0:
        return "git fetch failed: the gate needs the remote's history"
    if read_out(name) is not None:
        return f"{name} already ran under the filing commit {code['filed_commit']}; a redo needs a re-filing commit"
    rel = os.path.relpath(out_path(name), _seg.HERE)
    for span in (f"{code['filed_commit']}..HEAD", f"{code['filed_commit']}..@{{upstream}}"):
        if git("log", "--format=%H", span, "--", rel).stdout.strip():
            return f"a commit in {span} touched {rel}: {name} already ran under this filing"
    if name in ("j1", "j7") and failures(name) >= 2:
        return f"{name}'s implementation checks failed twice: the test has stopped"
    for o in OUTPUTS:
        r = os.path.relpath(out_path(o), _seg.HERE)
        tracked = git("ls-files", "--error-unmatch", r).returncode == 0
        if tracked and not os.path.exists(out_path(o)):
            return f"{r} is tracked but missing from disk"
        if os.path.exists(out_path(o)) and (not tracked or git("diff", "--quiet", "HEAD", "--", r).returncode):
            return f"{r} must be committed and unmodified before the next job"
    if git("rev-parse", "--verify", "--quiet", "@{upstream}").returncode != 0:
        return "HEAD has no upstream branch: set one, and push every job's output before the next job"
    if git("rev-list", "--count", "@{upstream}..HEAD").stdout.strip() != "0":
        return "HEAD has commits that are not pushed: push every job's output before the next job"
    if git("rev-list", "--count", "HEAD..@{upstream}").stdout.strip() != "0":
        return "HEAD is behind its upstream: bring it up to date before the next job"
    if name == "blocks":
        return None
    if name == "j1":
        return None if os.path.exists(out_path("blocks")) else "J0's block file is missing: run J0 first"
    j1 = read_out("j1")
    if j1 is None:
        return "J1 has not run under this filing"
    differ = [k for k in SAME_AS_J1 if j1.get("code", {}).get(k) != code[k]]
    if differ:
        return f"this run differs from J1's record in {differ}"
    if name == "verdict":
        j2 = read_out("j2")
        missing = [k for k in ("j2", "j3", "j4", "j5", "j6", "j7") if read_out(k) is None]
        if missing and not (j1.get("stop") or (j2 is not None and not c0_of(j2)["holds"])):
            return f"`verdict` needs J2-J7 under this filing unless a stop applies (missing: {', '.join(missing)})"
        return None
    npz = out_path("patcher", "npz")
    if not os.path.exists(npz) or sha256(npz) != j1.get("patcher_sha256"):
        return ("the patcher outputs are missing or differ from those J1 wrote: recompute them with J1's code and accept "
                "them only if their sha256 equals J1's record; otherwise the test stops")
    return blocked(name, j1, read_out("j2"))


def pool_rows():
    """Smoke: dev problems, GSM8K train idx 0-299. Otherwise the fresh pool, idx >= FRESH0 in file order, and only
    once the confirmatory rule is filed (provenance)."""
    if SMOKE:
        with open(GSM_TRAIN) as f:
            for i, line in enumerate(itertools.islice(f, 300)):
                yield i, json.loads(line)
        return
    provenance()
    with open(GSM_TRAIN) as f:
        for i, line in enumerate(f):
            if i >= FRESH0:
                yield i, json.loads(line)


def take_blocks(rows, tok):
    """Each cell's block: the next fit + test problems, in file order, whose cell text passes O6's filter; F4 also
    needs a distance in F4_DIST. Rejected problems are skipped and still use their draws. F3 cycles F3_KINDS over the
    texts kept in its fit list and in its test list. F6 is F1's problems in realblt_budget's format.
    Returns (blocks: indices, IDs and text hashes; texts: group -> rows with bytes and units)."""
    keep = keeper(tok)
    it = iter(rows)
    blocks, texts = {}, {}
    for cell in CELLS:
        rng = np.random.default_rng(SEEDS[cell])
        nf, nt = SMOKE_SIZES if SMOKE else SIZES[cell]
        got, visited, first = [], 0, None
        while len(got) < nf + nt:
            try:
                idx, p = next(it)
            except StopIteration:
                sys.exit(f"the pool ran out in {cell}")
            visited += 1; first = idx if first is None else first
            k = len(got) - (nf if len(got) >= nf else 0)
            x = make(p, rng, cell, F3_KINDS[k % 4] if cell == "F3" else "digits4")
            if x is None or not keep(x[0]):
                continue
            b, units, info = x
            if cell == "F4" and not F4_DIST[0] <= info["distance"] <= F4_DIST[1]:
                continue
            got.append({"idx": idx, **info, "n_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "b": b,
                        "units": units, "p": p})
        blocks[cell] = {"seed": SEEDS[cell], "visited": visited, "first_idx": first, "last_idx": got[-1]["idx"]}
        for split, rs in (("fit", got[:nf]), ("test", got[nf:])):
            blocks[cell][split] = [{k: v for k, v in r.items() if k not in ("b", "units", "p")} for r in rs]
            texts[f"{cell}.{split}"] = rs
    for split in ("fit", "test"):                                   # F6: F1's problems, no ticket
        rs = []
        for r in texts[f"F1.{split}"]:
            b, units = make_f6(r["p"])
            if keep(b):
                rs.append({"idx": r["idx"], "n_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "b": b,
                           "units": units})
        texts[f"F6.{split}"] = rs
        blocks.setdefault("F6", {})[split] = [{k: v for k, v in r.items() if k not in ("b", "units")} for r in rs]
    for rs in texts.values():
        for r in rs:
            r.pop("p", None)
    return blocks, texts


# ---------------------------------------------------------------- masks, counts and checks (CPU)
def masks_bytes(b):
    """The byte-only rules over tokens (True = no start): V_8; SIV (design 1, kmin 5); PO (design 2's added path);
    SIVPO = SIV or PO (design 2); L8b = L8 without its patcher conditions 4-6 (an ablation, reported only)."""
    n = len(b) + 1
    m = {"V8": vetoed(b, n, 8), "SIV": L.final_rules.siv(b, n, kmin=5), "PO": L.final_rules.po(b, n)}
    m["SIVPO"] = m["SIV"] | m["PO"]
    m["L8b"] = L.latch(b, np.full(n, np.inf), np.zeros(n, np.int16), agree="up")[0]     # conditions 4-6 always hold
    assert not (m["L8b"] & ~m["V8"]).any(), "L8b is not within V_8"
    return m


def add_masks(rows):
    for r in rows:
        r["hb"] = r["ent"].astype(np.float64) / L.LN2
        r["L8"], r["L8c"] = L.latch(r["b"], r["hb"], r["arg"])
        r.update(masks_bytes(r["b"]))
        assert not (r["L8"] & ~r["V8"]).any(), "L8 is not within V_8"
        assert not (r["L8c"] & ~r["L8"]).any(), "L8c is not within L8"
        assert not (r["L8"] & ~r["L8b"]).any(), "L8 is not within L8b"


def checks(groups):
    """On every text, before any scoring: causality of L8 and L8c (realblt_copy_latch.causal_changes: for every cut,
    bytes and outputs from the cut on are replaced twice), and the alignment of the stored outputs (wherever the
    actual byte costs under 0.01 bits, the argmax at that output is that byte). The subset asserts run in add_masks."""
    out = {}
    t0 = time.time()
    for g, rows in groups.items():
        changed = compared = boundary = decisions = 0; sure = misaligned = 0
        for i, r in enumerate(rows):
            c, k = L.causal_changes(r["b"], r["hb"], r["arg"], seed=i)
            changed += c; compared += k
            boundary += 4 * (len(r["b"]) - 1); decisions += 2 * (len(r["b"]) + 1)
            s = r["sur"] < 0.01
            sure += int(s.sum())
            misaligned += int((r["arg"][:-1][s] != np.frombuffer(r["b"], np.uint8)[s].astype(int) + L.OFFSET).sum())
        out[g] = {"texts": len(rows), "changed": changed, "comparisons": compared, "boundary_comparisons": boundary,
                  "distinct_decisions": decisions, "confident_positions": sure, "misaligned": misaligned}
        print(f"  checks: {g} changed {changed}, misaligned {misaligned}, {time.time() - t0:.0f}s", flush=True)
    out["passed"] = all(v["changed"] == 0 and v["misaligned"] == 0 for v in out.values())
    return out


def same_value(b, t, vend):
    """Whole-number test: the start at token t is same-value when an earlier occurrence of the 8 bytes before t is
    followed by exactly the value b[t-1:vend], read as a whole number with the value's own pattern. So 720 after
    7200, and 1,200 after 1,200,000, are new values (O6's prefix test called them same-value). A trailing comma is
    punctuation, not part of the value: `150,` and `150` are the same value."""
    v = b[t - 1:vend].rstrip(b",")
    for e in ends_of(b, b[t - 9:t - 1], t - 2):
        m = NUMV.match(b, e + 1)
        if m and m.group(0).rstrip(b",") == v:
            return True
    return False


def same_value_prefix(b, t, vend):
    """O6's test (realblt_copy_veto.py:202), reported beside the whole-number test"""
    return any(b[e + 1:].startswith(b[t - 1:vend]) for e in ends_of(b, b[t - 9:t - 1], t - 2))


def value_counts(texts, lay_of, rules, keep_cases=("L8", "L8c", "SIVPO")):
    """C4's counts. For each computed-value token (the byte after '= ' and the value's first digit, O6's V4 tokens)
    that starts a patch in a base layout and is masked by a rule: 'removed' (O6's strict count), and of those,
    'new_value' (the whole-number test; C4 and K2 count these) and 'new_value_prefix_test' (O6's test)."""
    res = {f"{base}-{rule}": {"in_layout": 0, "removed": 0, "new_value": 0, "new_value_prefix_test": 0, "cases": []}
           for base in ("D", "E10") for rule in rules}
    pos = collections.Counter()
    for j, r in enumerate(texts):
        b = r["b"]
        lay = lay_of(r)
        for m in VALUE.finditer(b):
            for t in sorted({m.start() + 3, m.start(1) + 1}):
                pos["all"] += 1
                for rule in rules:
                    pos[rule] += int(r[rule][t])
                if not any(t in st for st in lay.values()):
                    continue
                new = t < 10 or not same_value(b, t, m.end(1))
                new_pre = t < 10 or not same_value_prefix(b, t, m.end(1))
                for base, st in lay.items():
                    if t not in st:
                        continue
                    for rule in rules:
                        x = res[f"{base}-{rule}"]
                        x["in_layout"] += 1
                        if not r[rule][t]:
                            continue
                        x["removed"] += 1; x["new_value"] += int(new); x["new_value_prefix_test"] += int(new_pre)
                        if rule in keep_cases:
                            x["cases"].append({"text": j, "idx": r.get("idx"), "token": t, "value": m.group(1).decode(),
                                               "new_value": bool(new), "new_value_prefix_test": bool(new_pre),
                                               "context": b[max(0, t - 41):t - 1].decode(errors="replace")})
    return {"computed_value_positions": dict(pos), **res}


def cell_fit(rows, masks=RULE_MASKS):
    """Thresholds on a cell's fit texts (blt_layouts.Fit, train mode) and D's pooled train rate there"""
    fit = blt_layouts.Fit((0.10,))
    for r in rows:
        fit.add(L.sig_of(r, masks))
    fit.done()
    return fit, sum(len(r["D"]) for r in rows) / sum(len(r["b"]) + 1 for r in rows)


def copy_tokens(r):
    a, e = r["units"]["id_copy"][0]
    return np.arange(a + 1, e + 1)                                  # c1..cL: token t = byte t-1


def recall_footprint(test, fit):
    """R1 and R2 on F1's test texts, from patcher outputs only: L8 at the copy's first byte (c1), over all copies and
    over those where E10 starts at c1; and D starts each rule removes outside c1..c4, per text."""
    rows = []
    for r in test:
        c = copy_tokens(r)
        E = set(blt_layouts.layout("entropy", 0.10, L.sig_of(r), fit).tolist())
        cs = set(c.tolist())
        rows.append({"idx": r["idx"], "L8_c1": bool(r["L8"][c[0]]), "L8_all": bool(r["L8"][c].all()),
                     "E10_c1": int(c[0]) in E, "D_c1": int(c[0]) in set(r["D"].tolist()),
                     "SIVPO_c1": bool(r["SIVPO"][c[0]]), "SIV_c1": bool(r["SIV"][c[0]]),
                     "outside": {rule: int(sum(1 for t in r["D"] if r[rule][t] and int(t) not in cs))
                                 for rule in ("L8", "L8c", "SIV", "SIVPO", "V8", "L8b")}})
    e10 = [x for x in rows if x["E10_c1"]]
    res = {"n_copies": len(rows), "n_E10_c1": len(e10),
           "L8_fires_c1": float(np.mean([x["L8_c1"] for x in rows])),
           "L8_fires_c1_at_E10_starts": float(np.mean([x["L8_c1"] for x in e10])) if e10 else None,
           "L8_latched_whole_copy": float(np.mean([x["L8_all"] for x in rows])),
           "SIVPO_fires_c1": float(np.mean([x["SIVPO_c1"] for x in rows])),
           "outside_per_text": {rule: float(np.mean([x["outside"][rule] for x in rows])) for rule in rows[0]["outside"]}}
    at_e10 = res["L8_fires_c1_at_E10_starts"]                   # None: no copy with an E10 start at c1 to test
    res["R1 at E10's c1 starts"] = "not testable (no E10 start at c1)" if at_e10 is None else bool(at_e10 >= 0.95)
    res["R1"] = bool(res["L8_fires_c1"] >= 0.95 and (at_e10 is None or at_e10 >= 0.95))
    res["R2"] = bool(res["outside_per_text"]["L8"] <= 1.0)
    return res, rows


# ---------------------------------------------------------------- stored patcher outputs
ARR = ("ent", "arg", "sur", "D")


def save(C, path):
    meta, cols = [], collections.defaultdict(list)
    for g, rows in C.items():
        for r in rows:
            meta.append({"group": g, "n": len(r["b"]) + 1, "nD": len(r["D"]),
                         **{k: v for k, v in r.items() if k not in ARR + ("b",) and not isinstance(v, np.ndarray)}})
            cols["b"].append(np.frombuffer(r["b"], np.uint8)); cols["D"].append(r["D"])
            for k in ("ent", "arg", "sur"):
                cols[k].append(r[k])
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **{k: np.concatenate(v) for k, v in cols.items()})


def load(path, groups=None):
    z = np.load(path)
    x = {k: z[k] for k in ("b",) + ARR}
    C = collections.defaultdict(list)
    ob = oe = osr = od = 0
    for m in json.loads(str(z["meta"])):
        n, nd = m["n"], m["nD"]
        if groups is None or m["group"] in groups:
            r = {k: v for k, v in m.items() if k not in ("group", "n", "nD")}
            r["units"] = {k: [tuple(u) for u in v] for k, v in r["units"].items()}
            r.update(b=x["b"][ob:ob + n - 1].tobytes(), ent=x["ent"][oe:oe + n], arg=x["arg"][oe:oe + n],
                     sur=x["sur"][osr:osr + n - 1], D=x["D"][od:od + nd])
            C[m["group"]].append(r)
        ob += n - 1; oe += n; osr += n - 1; od += nd
    return dict(C)


def rebuild(tok):
    """The block texts again, checked against the committed block file (indices, IDs and text hashes). Outside smoke
    runs the block file must be committed, unmodified, and written under the same filed commit."""
    stored = json.load(open(out_path("blocks")))
    if not SMOKE:
        rel = os.path.relpath(out_path("blocks"), _seg.HERE)
        if git("ls-files", "--error-unmatch", rel).returncode or git("diff", "--quiet", "HEAD", "--", rel).returncode:
            sys.exit(f"refused: {rel} must be committed, unmodified, before J1")
        if stored.get("code", {}).get("filed_commit") != provenance()["filed_commit"]:
            sys.exit("refused: the block file was written under another filed commit")
    blocks, texts = take_blocks(pool_rows(), tok)
    for cell, v in blocks.items():
        for split in ("fit", "test"):
            assert v[split] == stored["blocks"][cell][split], f"{cell}.{split} differs from the committed block file"
    return texts


# ---------------------------------------------------------------- BLT-1B scoring
def scorer(model, dev, ids, units):
    """O6's scorer (realblt_copy_latch.scorer) for copies of any length: bits_copy covers the whole copied string"""
    n = ids.shape[1]
    ac, ae = units["id_copy"][0] if units["id_copy"] else (0, 0)
    done = {}

    def score(st):
        key = tuple(int(x) for x in st)
        if key not in done:
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(key, n).to(dev),
                                             use_cache=False).logits[0].float(), -1)
            tgt = ids[0, 1:]
            bits = (-lp[:-1].gather(1, tgt[:, None])[:, 0] / L.LN2).cpu().numpy()     # index t-1 = token t = byte t-1
            ok = (lp[:-1].argmax(-1) == tgt).cpu().numpy()
            per = {kd: [len(v), int(sum(ok[a:e].all() for a, e in v)), float(sum(bits[a:e].sum() for a, e in v))]
                   for kd, v in units.items()}
            done[key] = {"rate": len(key) / n, "bits": float(bits.sum()), "bits_copy": float(bits[ac:ae].sum()),
                         "copy_len": ae - ac, "n_bytes": n - 1, "kinds": per}
        return done[key]
    return score


def summarize(rows, names, pairs):
    """O6 step 2's readouts (realblt_copy_latch.summarize), with the copy's own length"""
    Lr = {name: [r["layouts"][name] for r in rows] for name in names}
    res = {"n": len(rows), "by_layout": {}, "pairs": {}}
    for name in names:
        x = Lr[name]
        d = {"rate": float(np.mean([y["rate"] for y in x])),
             "bpb": sum(y["bits"] for y in x) / sum(y["n_bytes"] for y in x),
             "bpb_without_copy": sum(y["bits"] - y["bits_copy"] for y in x) / sum(y["n_bytes"] - y["copy_len"] for y in x)}
        for kd in KINDS:
            nu = sum(y["kinds"][kd][0] for y in x)
            if nu:
                d[kd] = {"n": nu, "exact": sum(y["kinds"][kd][1] for y in x) / nu,
                         "bits_per_unit": sum(y["kinds"][kd][2] for y in x) / nu}
        res["by_layout"][name] = d
    for a, base in pairs:
        p = {}
        for kd in KINDS:
            nu = [y["kinds"][kd][0] for y in Lr[base]]
            if sum(nu):
                p[kd] = {"exact": boot_ratio([y["kinds"][kd][1] for y in Lr[a]], [y["kinds"][kd][1] for y in Lr[base]], nu),
                         "bits_per_unit": boot_ratio([y["kinds"][kd][2] for y in Lr[a]],
                                                     [y["kinds"][kd][2] for y in Lr[base]], nu)}
        p["bpb"] = boot_ratio([y["bits"] for y in Lr[a]], [y["bits"] for y in Lr[base]], [y["n_bytes"] for y in Lr[base]])
        p["bpb_without_copy"] = boot_ratio([y["bits"] - y["bits_copy"] for y in Lr[a]],
                                           [y["bits"] - y["bits_copy"] for y in Lr[base]],
                                           [y["n_bytes"] - y["copy_len"] for y in Lr[base]])
        p["rate"] = float(np.mean([y["rate"] for y in Lr[a]]) - np.mean([y["rate"] for y in Lr[base]]))
        res["pairs"][f"{a} vs {base}"] = p
    return res


def layouts(r, names, fit, r_d):
    sig = L.sig_of(r, RULE_MASKS)
    D = r["D"]
    E = blt_layouts.layout("entropy", 0.10, sig, fit)
    c = copy_tokens(r) if r["units"]["id_copy"] else np.zeros(0, int)
    lay = {}
    for name in names:
        if name in ("D", "E10"):
            lay[name] = D if name == "D" else E
        elif name.endswith("-all"):                                 # no start at any byte of the copy, no refill
            base = D if name.startswith("D") else E
            lay[name] = base[~np.isin(base, c)]
        elif name == "E10-L8":                                      # E10 without the starts L8 masks, no refill
            lay[name] = E[~r["L8"][E]]
        else:                                                       # X.RULE: the rule's mask, refitted (matched)
            base, rule = name.split(".")
            lay[name] = blt_layouts.layout("ent" + rule.lower(), r_d if base == "D" else 0.10, sig, fit)
    return lay


def rerun_patcher(model, tok, dev, rows):
    """Rerun the patcher on rows and count texts whose outputs differ from the stored ones"""
    cfg = model.config
    mism = 0
    for r in rows:
        ids = tok(r["b"].decode(), return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            ent, plen, plog = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                  max_patch_length=cfg.max_patch_length)
        n = ids.shape[1]
        D = np.r_[0, np.cumsum(plen[0].cpu().numpy())[:-1]]
        mism += int(not (np.array_equal(ent[0, :n].float().cpu().numpy(), r["ent"])
                         and np.array_equal(plog[0].float().argmax(-1).cpu().numpy(), r["arg"])
                         and sorted(int(t) for t in set(D.tolist()) if t < n) == r["D"].tolist()))
    return mism


def score_cell(cell, names, pairs, model, tok, dev, C, rerun=True):
    fit_rows, test = C[f"{cell}.fit"], C[f"{cell}.test"]
    add_masks(fit_rows); add_masks(test)
    fit, r_d = cell_fit(fit_rows)
    kinds_used = sorted({"entropy"} | {"ent" + n.split(".")[1].lower() for n in names if "." in n})
    thr = {f"{kd}@{R:.4f}": fit._threshold(kd, R) for kd in kinds_used for R in (r_d, 0.10)}
    mism = rerun_patcher(model, tok, dev, test) if rerun else None
    print(f"{cell}: {len(fit_rows)} fit / {len(test)} test; D's train rate {r_d:.4f}; patcher rerun differs on {mism}",
          flush=True)
    rows = []
    t0 = time.time()
    for i, r in enumerate(test):
        ids = tok(r["b"].decode(), return_tensors="pt").input_ids.to(dev)
        lay = layouts(r, names, fit, r_d)
        score = scorer(model, dev, ids, r["units"])
        row = {"idx": r["idx"], "layouts": {name: score(lay[name]) for name in names}, "n_tokens": len(r["b"]) + 1}
        if r["units"]["id_copy"]:
            c = copy_tokens(r)
            row.update(kind=r.get("kind"), distance=r.get("distance"), copy_len=len(c),
                       fires_c1={m: bool(r[m][c[0]]) for m in ("L8", "L8c", "SIV", "PO", "SIVPO", "V8", "L8b")},
                       starts_c1={name: bool(int(c[0]) in set(lay[name].tolist())) for name in names})
        rows.append(row)
        if (i + 1) % 50 == 0 or SMOKE:
            print(f"  {i + 1} texts, {time.time() - t0:.0f}s", flush=True)
    out = {"cell": cell, "train_rate_default": r_d, "thresholds_nats": thr, "patcher_rerun_mismatched_texts": mism,
           **summarize(rows, names, pairs)}
    if rows and "fires_c1" in rows[0]:
        out["fires_c1"] = {m: float(np.mean([r["fires_c1"][m] for r in rows])) for m in rows[0]["fires_c1"]}
    if cell == "F3":
        out["by_kind"] = {}
        for kind in F3_KINDS:
            sub = [r for r in rows if r["kind"] == kind]
            if sub:
                out["by_kind"][kind] = {**summarize(sub, names, pairs),
                                        "fires_c1": {m: float(np.mean([r["fires_c1"][m] for r in sub])) for m in sub[0]["fires_c1"]}}
    return out, rows


def model_load(dev):
    L.alone()
    return blt_light.load(MODEL, dev).eval()


def blocked(name, j1, j2=None):
    """Why job `name` may not score, or None. J1's stops (a failed check, K10, K2) block J2-J7; K5 (C0 failing in J2)
    blocks J3-J7. Callers skip this in smoke runs, which use dev problems."""
    if j1 is None:
        return "J1 has not run: no scoring"
    if j1.get("stop"):
        return f"J1 stopped the test ({j1['stop']}): no scoring"
    if name != "j2":
        if j2 is None:
            return "J2 has not run, so K5 cannot be read: no scoring"
        if not c0_of(j2)["holds"]:
            return "K5: C0 failed in J2 (the hazard does not replicate): no further scoring"
    return None


def read_out(name):
    """A job's output, or None. Outside smoke runs only an output written under the current filing commit counts."""
    if not os.path.exists(out_path(name)):
        return None
    x = json.load(open(out_path(name)))
    return x if SMOKE or x.get("code", {}).get("filed_commit") == provenance()["filed_commit"] else None


def run_job(name, cells):
    """cells: [(cell, layout names, pairs)]; scores each cell's test texts with J1's stored patcher outputs"""
    why = gate(name)
    if why:
        sys.exit(f"refused: {why}")
    code = provenance()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = model_load(dev)
    groups = [f"{c}.{s}" for c, _, _ in cells for s in ("fit", "test")]
    C = load(out_path("patcher", "npz"), groups)
    res, allrows = {"status": STATUS, "code": code}, {}
    for cell, names, pairs in cells:
        res[cell], allrows[cell] = score_cell(cell, names, pairs, model, tok, dev, C)
    json.dump({**res, "rows": allrows}, open(out_path(name), "w"), indent=1)
    for cell, _, _ in cells:
        print(cell, json.dumps({k: (round(v[0] * 100, 1), [round(x * 100, 1) for x in v[1]])
                                for k, v in ((p, q["id_copy"]["exact"]) for p, q in res[cell]["pairs"].items() if "id_copy" in q)}))
    print("wrote", out_path(name))


PAIRS_SIVPO = [("D.L8", "D.SIVPO"), ("E10.L8", "E10.SIVPO")]
F1_NAMES = ["D", "D-all", "D.L8", "D.V8", "D.SIVPO", "E10", "E10-all", "E10.L8", "E10-L8", "E10.V8", "E10.SIVPO"]
F1_PAIRS = [("D-all", "D"), ("D.L8", "D"), ("D.V8", "D"), ("D.SIVPO", "D"), ("E10-all", "E10"), ("E10.L8", "E10"),
            ("E10-L8", "E10"), ("E10.V8", "E10"), ("E10.SIVPO", "E10")] + PAIRS_SIVPO
F6_NAMES = ["D", "D.L8", "D.SIVPO", "E10", "E10.L8", "E10.SIVPO"]
F6_PAIRS = [("D.L8", "D"), ("D.SIVPO", "D"), ("E10.L8", "E10"), ("E10.SIVPO", "E10")] + PAIRS_SIVPO
F2A_NAMES = ["D", "D-all", "D.L8", "D.SIVPO", "E10", "E10.L8", "E10.SIVPO"]
F2A_PAIRS = [("D-all", "D"), ("D.L8", "D"), ("D.SIVPO", "D"), ("E10.L8", "E10"), ("E10.SIVPO", "E10")] + PAIRS_SIVPO
F2B_NAMES = ["D", "D-all", "D.L8", "D.SIV", "D.SIVPO", "E10", "E10-all", "E10.L8", "E10.SIVPO"]
F2B_PAIRS = [("D-all", "D"), ("D.L8", "D"), ("D.SIV", "D"), ("D.SIVPO", "D"), ("E10-all", "E10"), ("E10.L8", "E10"),
             ("E10.SIVPO", "E10")] + PAIRS_SIVPO
F3_NAMES = ["D", "D-all", "D.L8", "D.SIVPO", "E10", "E10-all", "E10.L8", "E10.SIVPO"]
F3_PAIRS = [("D-all", "D"), ("D.L8", "D"), ("D.SIVPO", "D"), ("E10-all", "E10"), ("E10.L8", "E10"),
            ("E10.SIVPO", "E10")] + PAIRS_SIVPO
F5_NAMES = ["D", "D-all", "D.L8", "D.SIVPO"]
F5_PAIRS = [("D-all", "D"), ("D.L8", "D"), ("D.SIVPO", "D"), ("D.L8", "D.SIVPO")]


# ---------------------------------------------------------------- J0 and J1
def cmd_blocks():
    why = gate("blocks")
    if why:
        sys.exit(f"refused: {why}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    blocks, texts = take_blocks(pool_rows(), tok)
    counts = {}                                                     # byte-level counts, fixed before any model run
    for g in ("F6.test", "F6.fit"):
        rows = texts[g]
        for r in rows:
            r.update(masks_bytes(r["b"]))
        vc = value_counts(rows, lambda r: {}, ("V8", "SIV", "PO", "SIVPO", "L8b"))
        counts[g] = {"computed_value_positions": vc["computed_value_positions"], "texts": len(rows)}
    for g in [f"{c}.test" for c in CELLS]:
        rows = texts[g]
        fire = collections.Counter()
        for r in rows:
            m = masks_bytes(r["b"]); c = copy_tokens(r)
            for rule in m:
                fire[rule] += int(m[rule][c[0]])
        counts[g] = {"texts": len(rows), "fires_c1": dict(fire)}
    out = {"status": STATUS, "pool": "dev, GSM8K train idx 0-299 (smoke)" if SMOKE else f"GSM8K train idx >= {FRESH0}",
           "code": provenance(), "blocks": blocks, "byte_counts": counts}
    if not SMOKE:                                                   # after filing only: the pool file as read
        with open(GSM_TRAIN, "rb") as f:
            out["pool_file"] = {"path": GSM_TRAIN, "sha256": sha256(GSM_TRAIN), "lines": sum(1 for _ in f)}
    json.dump(out, open(out_path("blocks"), "w"), indent=1)
    print(json.dumps({c: {k: v for k, v in b.items() if k not in ("fit", "test")} | {"fit": len(b["fit"]), "test": len(b["test"])}
                      for c, b in blocks.items()}, indent=1))
    print(json.dumps(counts, indent=1))
    print("wrote", out_path("blocks"), "- commit it before J1")


def cmd_j1():
    L.alone()
    why = gate("j1")
    if why:
        sys.exit(f"refused: {why}")
    code = provenance()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    C = rebuild(tok)
    print("texts: " + ", ".join(f"{g} {len(v)}" for g, v in C.items()), flush=True)
    model = blt_light.load(MODEL, dev).eval()
    L.patcher_pass(C, tok, model, dev)
    del model
    save(C, out_path("patcher", "npz"))
    for rows in C.values():
        add_masks(rows)                                             # a failed subset assert ends J1 here
    out = {"status": STATUS, "code": code, "patcher_sha256": sha256(out_path("patcher", "npz")),
           "subset_check": "passed: L8 within V_8, L8c within L8, L8 within L8b on every text", "checks": checks(C)}
    f2a = [bool(r["L8"][copy_tokens(r)[0]]) for r in C["F2a.test"]]     # G2 (a): decided by construction, so a check
    out["checks"]["G2 (a)"] = {"F2a test texts": len(f2a), "L8_fires_c1": float(np.mean(f2a)),
                               "holds": bool(np.mean(f2a) <= 0.05)}
    out["checks"]["passed"] = bool(out["checks"]["passed"] and out["checks"]["G2 (a)"]["holds"])
    if not out["checks"]["passed"] and not SMOKE:     # a bug, not a result: stop before any decision-relevant number
        second = failures("j1") >= 1
        out["stop"] = ["implementation check failed" + (" again (second failure)" if second else "")]
        json.dump(out, open(out_path("j1"), "w"), indent=1)
        sys.exit("J1: an implementation check failed. R1, R2, the F6 counts, C4 and K2 were not computed. Commit and "
                 "push this output. " + ("This is the second failure: the test stops." if second else
                                        "The fix goes in its own commit, then a re-filing commit (the PI's approval "
                                        "quoted), under which J0 and every job run again, once. A second failure stops "
                                        "the test."))
    fit1, _ = cell_fit(C["F1.fit"])
    out["R"], rf_rows = recall_footprint(C["F1.test"], fit1)
    fit6, r6 = cell_fit(C["F6.fit"])
    lay6 = lambda r: {"D": set(r["D"].tolist()),
                      "E10": set(blt_layouts.layout("entropy", 0.10, L.sig_of(r), fit6).tolist())}
    out["F6_counts"] = value_counts(C["F6.test"], lay6, ("L8", "L8c", "V8", "SIV", "PO", "SIVPO", "L8b"))
    out["thresholds_nats"] = {}
    for cell in CELLS + ("F6",):
        f, rd = cell_fit(C[f"{cell}.fit"])
        out["thresholds_nats"][cell] = {"train_rate_default": rd, **{f"{kd}@{R:.4f}": f._threshold(kd, R)
                                        for kd in ("entropy", "entl8", "entv8", "entsiv", "entsivpo") for R in (rd, 0.10)}}
    x = out["F6_counts"]
    out["C4"] = {"D_new_value": x["D-L8"]["new_value"], "D_allowed": C4_MAX_D, "D_strict": x["D-L8"]["removed"],
                 "D_in_layout": x["D-L8"]["in_layout"], "E10_new_value": x["E10-L8"]["new_value"],
                 "E10_strict": x["E10-L8"]["removed"], "E10_in_layout": x["E10-L8"]["in_layout"]}
    out["C4"]["holds"] = bool(out["C4"]["D_new_value"] <= C4_MAX_D and out["C4"]["E10_new_value"] == 0)
    stop = []
    if not out["checks"]["passed"]:
        stop.append("implementation check failed")
    if not (out["R"]["R1"] and out["R"]["R2"]):
        stop.append("K10")
    if out["C4"]["E10_new_value"] > 0:
        stop.append("K2")
    out["stop"] = stop
    json.dump({**out, "R_rows": rf_rows}, open(out_path("j1"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("F6_counts", "thresholds_nats")}, indent=1))
    print("wrote", out_path("j1"))


def cmd_j7():
    """F5: F1's texts under the conversion as released (blt_load.WINDOW = 0, as SEGR_BLT_WINDOW=0 sets it; load()
    reads it at call time), with the patcher outputs, masks and thresholds recomputed from that model"""
    L.alone()
    why = gate("j7")
    if why:
        sys.exit(f"refused: {why}")
    code = provenance()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    blt_load.WINDOW = 0
    model = blt_light.load(MODEL, dev).eval()
    if not (model.model.patcher.config.sliding_window is None and model.config.local_attention_window_len is None):
        sys.exit("refused: F5's model still has windows")             # an explicit check, kept under python -O
    C = load(out_path("patcher", "npz"), ("F1.fit", "F1.test"))
    for rows in C.values():
        for r in rows:
            for k in ARR:
                r.pop(k)
    L.patcher_pass(C, tok, model, dev)
    for rows in C.values():
        add_masks(rows)
    chk = checks(C)
    if not chk["passed"] and not SMOKE:     # a bug, as in J1: F5 is not scored
        second = failures("j7") >= 1
        json.dump({"status": STATUS, "code": code, "window": 0, "checks": chk,
                   "stop": ["implementation check failed (J7)" + (" again (second failure)" if second else "")]},
                  open(out_path("j7"), "w"), indent=1)
        sys.exit("F5: an implementation check failed; nothing was scored. Commit and push this output. The other jobs "
                 "and `verdict` still run under this filing, whose outputs remain the confirmatory result: F5 stays "
                 "unscored there, and the released paired verdict is not reached. " +
                 ("This is the second failure of J7's checks." if second else "A corrected rerun needs a re-filing the "
                  "PI approves (the fix in its own commit, then a re-filing commit), under which J0 and every job run "
                  "again, once; it is reported beside the filed result as a deviation, never in its place.")
                 + f" {chk}")
    res, rows = score_cell("F1", F5_NAMES, F5_PAIRS, model, tok, dev, C, rerun=False)
    res["cell"] = "F5"
    res["checks"] = chk
    fit, _ = cell_fit(C["F1.fit"])
    res["R"], _ = recall_footprint(C["F1.test"], fit)
    by_len = {}
    for lo, hi in ((0, 600), (600, 700), (700, 2000)):
        sub = [r for r in rows if lo <= r["n_tokens"] < hi]
        if sub:
            by_len[f"tokens {lo}-{hi}"] = {"n": len(sub), "L8_fires_c1": float(np.mean([r["fires_c1"]["L8"] for r in sub])),
                                           **{nm: float(np.mean([r["layouts"][nm]["kinds"]["id_copy"][1] for r in sub]))
                                              for nm in F5_NAMES}}
    res["by_length"] = by_len
    json.dump({"status": STATUS, "code": code, "window": 0, "F5": res, "rows": {"F5": rows}}, open(out_path("j7"), "w"),
              indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairs", "by_layout")}, indent=1))
    print("wrote", out_path("j7"))


# ---------------------------------------------------------------- verdicts
def pt(pair, kd="id_copy", what="exact"):
    """(point, [lo, hi]) of a pair's readout"""
    return tuple(pair[kd][what]) if kd in pair else (tuple(pair[what]) if what in pair else None)


def classify(diff, margin, other="SIV+PO"):
    """Paired verdict row: diff = (point, [lo, hi]) oriented so that positive means L8 is better than `other`"""
    v, (lo, hi) = diff
    if v >= margin and lo > 0:
        return "L8 better"
    if v <= -margin and hi < 0:
        return f"{other} better"
    return "tie"


def paired_rows(cellres, rates=("D", "E10"), rival="SIVPO"):
    """L8 against SIV+PO (rival=None: against the unlatched layout) in one cell: id_copy exact, id_copy bits per copy
    and the cost rows, at each rate, positive = L8 better"""
    out = {}
    for base in rates:
        key, other = (f"{base}.L8 vs {base}", base) if rival is None else (f"{base}.L8 vs {base}.{rival}", "SIV+PO")
        if key not in cellres["pairs"]:
            continue
        p = cellres["pairs"][key]
        rows = {}
        for kd, row in (("id_copy", "id_copy exact"), ("computed", "C3 computed"), ("final", "C3f final"),
                        ("copied", "C3c copied")):
            if kd in p:
                rows[row] = {"diff": pt(p, kd), "class": classify(pt(p, kd), MARGIN["exact"], other)}
        v, (lo, hi) = p["bpb_without_copy"]
        rows["C2 bits per byte without the copy"] = {"diff": (-v, [-hi, -lo]),
                                                     "class": classify((-v, [-hi, -lo]), MARGIN["bpb"], other)}
        if "id_copy" in p:
            v, (lo, hi) = p["id_copy"]["bits_per_unit"]
            rows["id_copy bits per copy"] = {"diff": (-v, [-hi, -lo]),
                                             "class": classify((-v, [-hi, -lo]), MARGIN["bits"], other)}
        out[base] = rows
    return out


def c0_of(j2):
    """C0, K5's bar: F1's D-all - D on id_copy exact, point estimate >= +20 points and lower bound >= +10"""
    v = pt(j2["F1"]["pairs"]["D-all vs D"])
    return {"value": v, "holds": v[0] >= 0.20 and v[1][0] >= 0.10}


def not_reached(why):
    return {"holds": None, "scope": f"not reached: {why}"}


def paired_not_reached(why):
    return {"windowed": {"class": "not reached", "reading": f"not reached: {why}"},
            "released": {"class": "not reached", "reading": f"not reached: {why}"}, "note": NOT_NEEDED}


def stop_scope(stops):
    """The fix claim's scope after a stop: K2 and K10 fail L8; K5 and the implementation checks leave it untested"""
    if any(s.startswith("implementation") for s in stops):
        return "not reached: the implementation checks failed"
    failed = [s.split(":")[0] for s in stops if s.startswith(("K2", "K10"))]
    return f"failed: {', '.join(failed)}" if failed else f"not reached: {', '.join(stops)} (the hazard did not replicate)"


def verdict_from(J):
    """Every bar, stop and kill rule of the rule, from the job outputs J = {'j1': ..., 'j2': ...}. The fix claim's
    scope says 'failed' or 'not reached' when it does not hold. After a stop, or with an output missing, the
    generality claim and the paired verdicts are 'not reached'. K6-K9 are read only when there is no stop."""
    V = {"status": STATUS, "bars": {}, "stops": [], "kills": [], "reported": {}}
    B = V["bars"]
    if "j1" not in J:
        V["fix_claim"], V["generality_claim"] = {"holds": False, "scope": "not reached: J1 has not run"}, \
            not_reached("J1 has not run")
        V["paired"] = paired_not_reached("J1 has not run")
        return V
    j1 = J["j1"]
    B["checks"] = j1["checks"]["passed"]
    if not B["checks"]:
        V["stops"].append("implementation check failed (J1)")
    if "R" in j1:                                                   # J1 computes R and C4 only after its checks pass
        B["R1"], B["R2"] = j1["R"]["R1"], j1["R"]["R2"]
        if not (B["R1"] and B["R2"]):
            V["stops"].append("K10: R1 or R2 failed (J1); no scoring")
    if "C4" in j1:
        B["C4"] = {"D_new_value": j1["C4"]["D_new_value"], "allowed": C4_MAX_D,
                   "E10_new_value": j1["C4"]["E10_new_value"], "D_strict_O6": j1["C4"]["D_strict"],
                   "E10_strict_O6": j1["C4"]["E10_strict"],
                   "holds": j1["C4"]["D_new_value"] <= C4_MAX_D and j1["C4"]["E10_new_value"] == 0}
        if j1["C4"]["E10_new_value"] > 0:
            V["kills"].append("K2: a new-value removal from E10 in F6; drop L8")
            V["stops"].append("K2: a new-value removal from E10 in F6")
    if "j2" not in J:
        why = f"the test stopped ({', '.join(V['stops'])})" if V["stops"] else "J2 has not run"
        V["fix_claim"] = {"holds": False, "scope": stop_scope(V["stops"]) if V["stops"] else "not reached: J2 has not run"}
        V["generality_claim"], V["paired"] = not_reached(why), paired_not_reached(why)
        return V
    F1 = J["j2"]["F1"]["pairs"]
    B["C0"] = c0_of(J["j2"])
    c0 = B["C0"]["value"]
    if not B["C0"]["holds"]:
        V["kills"].append("K5: C0 fails; the hazard does not replicate; stop")
        V["stops"].append("K5")
    stopped = bool(V["stops"])                                      # K6-K9 are read only without a stop
    c1 = pt(F1["D.L8 vs D"])
    B["C1"] = {"value": c1, "holds": c1[0] >= 0.15 and c1[1][0] > 0}
    V["reported"]["C1 share of D-all - D"] = c1[0] / c0[0] if c0[0] > 0 else None
    bound, e10l8, e10m = pt(F1["E10-all vs E10"]), pt(F1["E10.L8 vs E10"]), pt(F1["E10-L8 vs E10"])
    testable = bound[1][0] > 0
    i_ = e10m[0] >= 0.9 * bound[0] - 1e-9          # points are k/n: an exact 90% share (10 k1 = 9 k2) passes
    ii = e10l8[1][0] > 0
    B["C1e"] = {"onset_removal_bound": bound, "E10-L8 - E10": e10m, "E10.L8 - E10": e10l8, "testable": testable,
                "i": i_, "ii": ii, "holds": bool(testable and i_ and ii)}
    V["reported"]["PI target +15 at E10"] = {"E10.L8 - E10": e10l8[0], "met": e10l8[0] >= 0.15}
    if e10l8[0] < 0.15 and not stopped:
        why = ("bound untestable (its lower bound is <= 0)" if not testable else
               "refill (E10-L8 - E10 reaches +15; the matched refill gives it back)" if e10m[0] >= 0.15 else
               "bound-limited" if i_ else "misses onset starts")
        V["reported"]["K8"] = (f"target not met; reading: {why} (L8 fires at "
                               f"{j1.get('R', {}).get('L8_fires_c1_at_E10_starts')} of E10's c1 starts)")
    for bar, kd in (("C3", "computed"), ("C3f", "final"), ("C3c", "copied")):
        vals = {b: pt(F1[f"{b}.L8 vs {b}"], kd)[0] for b in ("D", "E10")}
        B[bar] = {"value": vals, "holds": {b: v >= -0.02 for b, v in vals.items()}}
    vals = {b: F1[f"{b}.L8 vs {b}"]["bpb_without_copy"][0] for b in ("D", "E10")}
    B["C2"] = {"value": vals, "holds": {b: v <= 0.01 for b, v in vals.items()}}
    rate = {f"F1 {b}": F1[f"{b}.L8 vs {b}"]["rate"] for b in ("D", "E10")}
    if "j3" in J:
        F6 = J["j3"]["F6"]["pairs"]
        rate.update({f"F6 {b}": F6[f"{b}.L8 vs {b}"]["rate"] for b in ("D", "E10")})
        vals = {b: pt(F6[f"{b}.L8 vs {b}"], "computed")[0] for b in ("D", "E10")}
        B["C4s"] = {"value": vals, "holds": {b: v >= -0.02 for b, v in vals.items()}}
    B["rate"] = {"value": rate, "holds": {k: abs(v) <= 0.005 for k, v in rate.items()}}
    ctl = F1["D.V8 vs D"]
    cv, bv = pt(ctl, "computed")[0], ctl["bpb_without_copy"][0]
    B["control"] = {"computed": cv, "bpb_without_copy": bv, "holds": cv <= -0.02 or bv >= 0.01}
    # the fix claim: at both rates, or at the default rate alone when C1e cannot be tested (K7 reads the same rates)
    at = ("D", "E10") if testable else ("D",)
    failed = [k for k in ("C1", "C2", "C3", "C3f", "C3c", "C4", "C4s", "rate", "C1e")
              if k in B and not (B[k]["holds"] if isinstance(B[k]["holds"], bool) else
                                 all(v for kk, v in B[k]["holds"].items() if kk.split()[-1] in at))]
    if not testable:
        failed = [k for k in failed if k != "C1e"]
    k7, k6 = not stopped and bool(failed), not stopped and not B["control"]["holds"]
    if k7:
        V["kills"].append(f"K7: {', '.join(failed)} failed; no fix claim")
    if k6:
        V["kills"].append("K6: the control shows no cost (computed > -2 and bits per byte < +0.01); no fix claim")
    if stopped:
        V["fix_claim"] = {"holds": False, "scope": stop_scope(V["stops"])}
    elif k7:                                                        # K7's reading governs; K6's follows it
        V["fix_claim"] = {"holds": False, "scope": f"failed: K7 ({', '.join(failed)})" +
                          ("; and void: the control showed no cost (K6)" if k6 else "")}
    elif k6:
        V["fix_claim"] = {"holds": False, "scope": "void: the control showed no cost (K6)"}
    elif "j3" not in J:
        V["fix_claim"] = {"holds": False, "scope": "not reached: J3 has not run"}
    else:
        V["fix_claim"] = {"holds": True, "scope": "both rates" if testable else
                          "default rate only (C1e untestable: the onset-removal bound's lower bound is <= 0)"}
    # generality: one AND. G1 and each G3 kind are tested only where their cell shows the hazard (D-all - D >= +20).
    # K9 is read without a stop, as K6-K8 are; its sentence prints only while the fix claim holds (sentences)
    untested = lambda what: f"not testable: no hazard in {what} (D-all - D below +20)"
    if "j4" in J:
        F2a, F2b = J["j4"]["F2a"], J["j4"]["F2b"]
        p = F2b["pairs"]
        haz, g1 = pt(p["D-all vs D"]), pt(p["D.L8 vs D"])
        t = haz[0] >= 0.20
        B["G1"] = {"F2b D-all - D": haz, "D.L8 - D": g1, "testable": t,
                   "holds": bool(g1[0] >= 0.15 and g1[1][0] > 0) if t else None, **({} if t else {"status": untested("F2b")})}
        bnd, g1e = pt(p["E10-all vs E10"]), pt(p["E10.L8 vs E10"])
        te = bnd[1][0] > 0
        V["reported"]["G1e (reported only)"] = {"F2b E10-all - E10": bnd, "E10.L8 - E10": g1e, "testable": te,
                                                "lower bound > 0": bool(g1e[1][0] > 0) if te else None,
                                                **({} if te else {"status": "not testable: E10-all - E10 has no lower "
                                                                            "bound above 0 in F2b"})}
        V["reported"]["G1s (D.SIV - D, decided by construction)"] = pt(p["D.SIV vs D"])
        q = F2a["pairs"]

        def g2_at(b):                                               # G2 (b), the claim conditions at rate b
            r = q[f"{b}.L8 vs {b}"]
            return (abs(pt(r)[0]) <= 0.03 and r["bpb_without_copy"][0] <= 0.01
                    and all(pt(r, kd)[0] >= -0.02 for kd in ("computed", "final", "copied") if kd in r))
        g2a = j1["checks"].get("G2 (a)", {"holds": F2a["fires_c1"]["L8"] <= 0.05})["holds"]   # J1's check; J4 if absent
        B["G2"] = {"(a) L8 fires at F2a's c1, J1's check": g2a, "(a) in J4": F2a["fires_c1"]["L8"],
                   "D.L8 - D": pt(q["D.L8 vs D"]), "E10.L8 - E10": pt(q["E10.L8 vs E10"]),
                   "(b) holds": {b: g2_at(b) for b in ("D", "E10")}}
        if not stopped and t and not B["G1"]["holds"]:
            V["kills"].append("K9: G1 fails: the fix does not extend to the context reworded mid-line (F2b)")
    if "j5" in J:
        g3 = {}
        for kind in ("digits6", "letters4", "mixed"):                # doses are reported only
            bk = J["j5"]["F3"]["by_kind"].get(kind)
            if not bk:
                continue
            da, dl = pt(bk["pairs"]["D-all vs D"]), pt(bk["pairs"]["D.L8 vs D"])
            ap = da[0] >= 0.20
            g3[kind] = {"D-all - D": da, "D.L8 - D": dl, "applies": ap,
                        "holds": bool(dl[0] >= 0.15 and dl[1][0] > 0 and dl[0] >= 0.7 * da[0] - 1e-9) if ap else None,
                        **({} if ap else {"status": untested(f"{kind} IDs")})}
        B["G3"] = g3
        if not stopped:
            V["kills"] += [f"K9: G3 fails: the fix does not extend to {k} IDs" for k, v in g3.items()
                           if v["applies"] and not v["holds"]]
        dose = J["j5"]["F3"]["by_kind"].get("dose")
        if dose:
            V["reported"]["F3 doses"] = {"L8_fires": dose["fires_c1"]["L8"], "D.L8 - D": pt(dose["pairs"]["D.L8 vs D"]),
                                         "D-all - D": pt(dose["pairs"]["D-all vs D"])}
    if stopped:
        V["generality_claim"] = not_reached(f"the test stopped ({', '.join(V['stops'])})")
    elif "j3" not in J:
        V["generality_claim"] = not_reached("J3 has not run")
    elif "G1" not in B or "G3" not in B:
        V["generality_claim"] = not_reached("J4 or J5 has not run")
    else:
        tested = {**({"F2b's reworded context (G1)": B["G1"]["holds"]} if B["G1"]["testable"] else {}),
                  **{f"{k} IDs (G3)": v["holds"] for k, v in B["G3"].items() if v["applies"]}}
        if not tested:
            V["generality_claim"] = {"holds": None, "scope": "not testable: no hazard (D-all - D >= +20) in F2b or in "
                                                             "any F3 ID kind"}
        else:
            parts = {"fix claim": V["fix_claim"]["holds"], "G2 (a)": B["G2"]["(a) L8 fires at F2a's c1, J1's check"],
                     **{f"G2 (b) at {b}": B["G2"]["(b) holds"][b] for b in at}, **tested}
            V["generality_claim"] = {"holds": all(parts.values()), "parts": parts,
                                     "failing": [k for k, v in parts.items() if not v],
                                     "scope": "; ".join(tested) + f" (G1 and G3 at the default rate; G2 (b) at "
                                                                  f"{' and '.join(at)})"}
    V["reported"]["patcher rerun mismatches (a limitation, not a stop)"] = {
        c: J[j][c].get("patcher_rerun_mismatched_texts") for j, cs in (("j2", ("F1",)), ("j3", ("F6",)),
                                                                        ("j4", ("F2a", "F2b")), ("j5", ("F3",)),
                                                                        ("j6", ("F4",))) if j in J for c in cs}
    if "j6" in J:
        F4 = J["j6"]["F4"]
        V["reported"]["F4"] = {"D-all - D": pt(F4["pairs"]["D-all vs D"]), "D.L8 - D": pt(F4["pairs"]["D.L8 vs D"]),
                               "L8_fires": F4["fires_c1"]["L8"]}
    if "j7" in J:
        if "F5" in J["j7"]:
            F5 = J["j7"]["F5"]
            bl = F5["by_layout"]
            V["reported"]["F5"] = {"D.L8 - D": pt(F5["pairs"]["D.L8 vs D"]),
                                   "bits per copy D.L8, D": (bl["D.L8"]["id_copy"]["bits_per_unit"],
                                                             bl["D"]["id_copy"]["bits_per_unit"]),
                                   "by_length": F5["by_length"]}
        else:
            V["reported"]["F5"] = f"not scored: {J['j7'].get('stop')}"
    if stopped:
        V["paired"] = paired_not_reached(f"the test stopped ({', '.join(V['stops'])})")
    else:
        V["paired"] = paired_verdict(J)
        if any(k.startswith("K6") for k in V["kills"]):
            V["paired"]["K6"] = "the control showed no cost, so a tie on a cost row carries no information"
    return V


# the paired verdicts' readings, fixed before any data (the PI's choice "Split, label honestly (Recommended)",
# 2026-10-30; copy rows set each class since the third check; reworded after the fourth check)
SPLIT = ("in {a} of the {n} lost copies L8 did not fire at c1 (copies L8 missed); in {b} it fired, and the loss is a "
         "side effect of the refit")
READINGS = {
    "windowed": {
        "tie": ("On the cells and rates with the hazard, no copy row is classed as differing: none differs by 2 points "
                "or more with a 95% interval that excludes 0 (largest difference {x} points, {row}, {interval}). A "
                "100-text row can still sit a few points apart. This is not an equivalence test."),
        "SIV+PO better": "L8 lost copies the byte rule kept, in {where}: " + SPLIT + ".",
        "L8 better": ("L8 kept copies the byte rule lost, in {where}. This is not expected by construction: within the "
                      "window SIV+PO masks the whole copy."),
        "mixed": "Read cell by cell; it settles nothing.",
        "not testable": "No cell and rate showed the hazard, so no copy row could set a class."},
    "released": {
        "L8 better": ("Relative to SIV+PO, a rule that stops where the model cannot copy helps under the released "
                      "conversion; F5's exact copies by text length show where. A byte rule with a length cutoff might "
                      "match it; that is untested."),
        "tie": ("Under the released conversion the copy row is not classed as differing ({x} points, {interval}): a "
                "difference of 2 points or more would need a 95% interval that excludes 0. This is not an equivalence "
                "test."),
        "SIV+PO better": "Under the released conversion, L8 lost copies the byte rule kept: " + SPLIT + "."}}
BESIDE = {   # the sentence for each other row, when it does not tie
    "no hazard": "No hazard here ({gate}), so a difference here is not a fixed or missed copy; it is reported only.",
    "F2a": "Neither rule touches the copy; a difference is a side effect of the refit.",
    "bits": "L8's copies cost {} bits than the byte rule's on {}.",                   # more | fewer
    "cost": "L8 cost {} than the byte rule on {}.",                                   # more | less
    "dose copy lost": ("L8 lost dose copies the byte rule kept: in {a} of the {n} lost copies L8 did not fire at c1, "
                       "as its S0 condition skips short doses by design, and this is that design's cost; in {b} it "
                       "fired, and the loss is a side effect of the refit."),
    "dose copy kept": "L8 kept dose copies the byte rule lost.",
    "dose other": ("The byte rule also masks the recurring line prefix before the dose; a difference there is neither a "
                   "copy loss nor an L8 cost.")}
GATE_TEXT = {"D": "D-all - D below +20", "E10": "E10-all - E10 has no lower bound above 0"}
NOT_NEEDED = "No outcome of this test shows that reading the model is needed."
CLASS_CELLS = ("F1", "F2b", "F3 digits6", "F3 letters4", "F3 mixed")     # their id_copy exact rows set the windowed class
DISPLAY = {"digits6": "6-digit IDs", "letters4": "letter IDs", "mixed": "mixed IDs", "dose": "doses"}


def disp(cell):
    """A cell's name as printed: 'F3 6-digit IDs', not 'F3 digits6'"""
    return f"F3 {DISPLAY[cell[3:]]}" if cell.startswith("F3 ") else cell


def points(v):
    """'-3.0' and '[-7.0, +0.0]' for a (point, [lo, hi]) of fractions"""
    return f"{v[0] * 100:+.1f}", f"[{v[1][0] * 100:+.1f}, {v[1][1] * 100:+.1f}]"


def hazard(cellres, base):
    """Whether a cell shows the hazard at rate `base`: at D, D-all - D >= +20 points (point estimate; G1's and G3's
    gate); at E10, E10-all - E10 has a lower bound > 0 (C1e's and G1e's gate)"""
    p = cellres["pairs"].get(f"{base}-all vs {base}")
    return p is not None and (pt(p)[0] >= 0.20 if base == "D" else pt(p)[1][0] > 0)


def discordance(rows, base):
    """The texts where X.L8 and X.SIVPO differ on id_copy exact: for the copies L8 lost and those it kept, how many,
    and how many with L8 firing at c1 and with X starting a patch at c1; the same counts crossed (direction by L8's
    firing by X's start); and the texts' problem indices"""
    out = {"lost": collections.Counter(), "kept": collections.Counter()}
    joint, idx = collections.Counter(), {"lost": [], "kept": []}
    for r in rows:
        a, b = (r["layouts"][f"{base}.{m}"]["kinds"]["id_copy"][1] for m in ("L8", "SIVPO"))
        if a != b:
            d = "kept" if a > b else "lost"
            side = out[d]
            side["copies"] += 1
            side["L8 fired at c1" if r["fires_c1"]["L8"] else "L8 did not fire at c1"] += 1
            side[f"{base} starts at c1" if r["starts_c1"][base] else f"{base} does not start at c1"] += 1
            joint[f"L8 {d} the copy; L8 {'fires' if r['fires_c1']['L8'] else 'does not fire'} at c1; {base} "
                  f"{'starts' if r['starts_c1'][base] else 'does not start'} at c1"] += 1
            idx[d].append(r["idx"])
    return {**{k: dict(v) for k, v in out.items()}, "joint": dict(joint), "texts (problem idx)": idx}


def split(recs):
    """{n, a, b}: lost copies over the rows `recs`, those where L8 did not fire at c1, those where it fired"""
    n = sum(r.get("discordant texts", {}).get("lost", {}).get("copies", 0) for r in recs)
    a = sum(r.get("discordant texts", {}).get("lost", {}).get("L8 did not fire at c1", 0) for r in recs)
    return {"n": n, "a": a, "b": n - a}


def side_sentence(rec, cellres):
    """The pre-registered sentence for a row beside the classes that does not tie"""
    cell, base, row, better = rec["cell"], rec["rate"], rec["row"], rec["class"] == "L8 better"
    where = f"{disp(cell)}, {base}, {row}"
    if row == "id_copy exact" and cell != "F2a":
        if not hazard(cellres, base):
            return BESIDE["no hazard"].format(gate=GATE_TEXT[base])
        if cell == "F3 dose":
            return BESIDE["dose copy kept"] if better else BESIDE["dose copy lost"].format(**split([rec]))
    if cell == "F2a" and row.startswith("id_copy"):
        return BESIDE["F2a"]
    if cell == "F3 dose" and not row.startswith("id_copy"):
        return BESIDE["dose other"]
    if row == "id_copy bits per copy":
        return BESIDE["bits"].format("fewer" if better else "more", where)
    return BESIDE["cost"].format("less" if better else "more", where)


def set_class(recs):
    lb, sb = (any(r["class"] == c for r in recs) for c in ("L8 better", "SIV+PO better"))
    return "not testable" if not recs else "mixed" if lb and sb else "SIV+PO better" if sb else \
        "L8 better" if lb else "tie"


def paired_verdict(J):
    """L8 against SIV+PO as two verdicts, each read as fixed in advance (READINGS). Only copy rows with the hazard set a
    class:
    - windowed: the id_copy exact rows of F1, F2b and F3's 6-digit, letter and mixed IDs, each at D where its cell's
      D-all - D >= +20 and at E10 where its E10-all - E10 has a lower bound > 0; 'not testable' if none qualifies;
    - released: F5's id_copy exact row at D, with no gate; F5's L8 against D is reported beside it.
    Every other row (F2a, F6, bits per copy, the cost rows, copy rows without the hazard) is reported beside, with its
    sentence (BESIDE) when it does not tie; the doses and F4 are reported apart. Each id_copy row that does not tie
    carries its discordant texts, and the 'SIV+PO better' readings give their split. A verdict whose jobs are missing
    (windowed: J2-J5; released: a scored J7) is 'not reached'. No outcome shows that reading the model is needed."""
    res = {c: r for c, r in (("F1", J.get("j2", {}).get("F1")), ("F6", J.get("j3", {}).get("F6")),
                             ("F2a", J.get("j4", {}).get("F2a")), ("F2b", J.get("j4", {}).get("F2b")),
                             ("F4", J.get("j6", {}).get("F4")), ("F5", J.get("j7", {}).get("F5"))) if r}
    for kind, bk in J.get("j5", {}).get("F3", {}).get("by_kind", {}).items():
        res[f"F3 {kind}"] = bk
    src = {"F1": ("j2", "F1"), "F2a": ("j4", "F2a"), "F2b": ("j4", "F2b"), "F4": ("j6", "F4"), "F5": ("j7", "F5")}
    texts = lambda c: ([r for r in J["j5"].get("rows", {}).get("F3", []) if r.get("kind") == c[3:]] if c.startswith("F3")
                       else J[src[c][0]].get("rows", {}).get(src[c][1], []))
    out = {"class rows": [], "beside": [], "released row": None, "released, beside": [], "doses (apart)": [],
           "F4 (apart)": []}
    for c, cellres in res.items():
        for b, rows in paired_rows(cellres, rates=("D",) if c == "F5" else ("D", "E10")).items():
            for row, x in rows.items():
                rec = {"cell": c, "rate": b, "row": row, "class": x["class"], "diff": x["diff"]}
                if row == "id_copy exact" and x["class"] != "tie":
                    rec["discordant texts"] = discordance(texts(c), b)
                if c == "F5":
                    group = "released row" if row == "id_copy exact" else "released, beside"
                elif c == "F4":
                    group = "F4 (apart)"
                elif c == "F3 dose":
                    group = "doses (apart)"
                elif c in CLASS_CELLS and row == "id_copy exact" and hazard(cellres, b):
                    group = "class rows"
                else:
                    group = "beside"
                if x["class"] != "tie" and group not in ("class rows", "released row", "F4 (apart)"):
                    rec["sentence"] = side_sentence(rec, cellres)
                if group == "released row":
                    out[group] = rec
                else:
                    out[group].append(rec)
    miss = [k for k in ("j2", "j3", "j4", "j5") if k not in J]
    if miss:
        windowed = {"class": "not reached", "reading": f"not reached: {', '.join(miss)} missing",
                    "class rows so far": out["class rows"]}
    else:
        cls = set_class(out["class rows"])
        deciding = [r for r in out["class rows"] if r["class"] == cls]
        fill = {"where": "; ".join(f"{disp(r['cell'])} at {r['rate']}" for r in deciding), **split(deciding)}
        if cls == "tie":
            top = max(out["class rows"], key=lambda r: abs(r["diff"][0]))
            x, interval = points(top["diff"])
            fill.update(x=x, interval=interval, row=f"{disp(top['cell'])} at {top['rate']}")
        windowed = {"class": cls, "reading": READINGS["windowed"][cls].format(**fill), "class rows": out["class rows"]}
    if out["released row"] is None:
        released = {"class": "not reached", "reading": "not reached: no scored J7 (F5)"}
    else:
        r = out["released row"]
        x, interval = points(r["diff"])
        released = {"class": r["class"], "reading": READINGS["released"][r["class"]].format(x=x, interval=interval,
                                                                                             **split([r])),
                    "class row": r, "beside": out["released, beside"],
                    "L8 against D (beside)": paired_rows(res["F5"], rates=("D",), rival=None).get("D")}
    rate = {c: {b: r["pairs"][f"{b}.SIVPO vs {b}"]["rate"] for b in ("D", "E10") if f"{b}.SIVPO vs {b}" in r["pairs"]}
            for c, r in res.items() if not c.startswith("F3 ")}
    if "j5" in J:
        F3 = J["j5"]["F3"]["pairs"]
        rate["F3"] = {b: F3[f"{b}.SIVPO vs {b}"]["rate"] for b in ("D", "E10") if f"{b}.SIVPO vs {b}" in F3}
    return {"windowed": windowed, "released": released, "beside": out["beside"], "doses (apart)": out["doses (apart)"],
            "F4 (apart)": out["F4 (apart)"], "note": NOT_NEEDED,
            "SIV+PO matched rate minus the unlatched layout's (reported; L8's tolerance is 0.005)": rate}


TEMPLATE = "In O6's `\\nTicket` template, scored teacher-forced with 512-byte windows ({n} texts), "
F6_BARS = ("C4", "C4s")                                # K7's bars measured on F6; the rate bar is split by its cells


def k7_sentences(B, scope, n1):
    """K7's reading in two parts, each printed only if it has a bar: F1's bars under O6's template, then F6's (C4 with
    its count, C4s); the rate bar goes under the cell or cells whose rate failed, at the rates the fix claim reads"""
    bars = scope.split("(", 1)[1].split(")", 1)[0].split(", ")
    at = ("D", "E10") if B["C1e"]["testable"] else ("D",)
    rates = {}
    for k, ok in B["rate"]["holds"].items():                       # keys "F1 D", "F6 E10", ...
        cell, r = k.split()
        if r in at and not ok:
            rates.setdefault(cell, []).append(r)
    name = lambda b: (f"C4 ({B['C4']['D_new_value']} new-value removals by the patcher at the default rate; at most "
                      f"{C4_MAX_D} allowed)" if b == "C4" else b)
    f1 = [b for b in bars if b not in F6_BARS + ("rate",)] + [f"rate ({' and '.join(rates[c])})" for c in ("F1",)
                                                             if c in rates]
    f6 = [name(b) for b in bars if b in F6_BARS] + [f"rate ({' and '.join(rates[c])})" for c in ("F6",) if c in rates]
    out = []
    if f1:
        out.append(f"In O6's `\\nTicket` template (F1), scored teacher-forced with 512-byte windows ({n1} texts), L8 "
                   f"failed on fresh problems: {', '.join(f1)}.")
    if f6:
        out.append(f"On the same problems without the ticket (F6), L8 failed: {', '.join(f6)}.")
    return out


def sentences(V, J):
    """The licensed sentences of the rule's Claims, with their numbers filled in, in the order the rule reads them"""
    out, B, stops = [], V["bars"], V["stops"]
    n1 = J.get("j2", {}).get("F1", {}).get("n")
    pct = lambda v: f"{v[0] * 100:+.1f} [{v[1][0] * 100:+.1f}, {v[1][1] * 100:+.1f}]"
    if any(s.startswith("implementation") for s in stops):
        second = any("again (second failure)" in s for s in J.get("j1", {}).get("stop", []))
        out.append("The implementation failed; nothing was tested." if second else
                   "not reached: the implementation check failed; a redo needs the PI's approval")
    if any(s.startswith("K10") for s in stops):
        out.append("L8 failed its recall or footprint check before scoring.")
    if any(s.startswith("K2") for s in stops):
        out.append("L8 removed a new value at 10%; L8 is dropped.")
    if "K5" in stops:
        out.append(TEMPLATE.format(n=n1) + "the hazard did not replicate on fresh problems; L8's copy fix was not "
                   "scored.")
    fix = V["fix_claim"]
    if fix["holds"]:
        share = V["reported"].get("C1 share of D-all - D")
        both = fix["scope"] == "both rates"
        out.append("On fresh GSM8K problems in O6's `\\nTicket NNNN` template, in BLT-1B with its 512-byte local windows, "
                   "scored teacher-forced, L8 restores verbatim copies at BLT-1B's default patch rate"
                   + (" and at entropy@10%, each compared at a matched rate. " if both else ", compared at a matched rate. ")
                   + f"At the default rate it gains {pct(B['C1']['value'])} points of exact copies, {share:.2f} of what "
                   "removing every copy start gains" + (f"; at 10% it gains {pct(B['C1e']['E10.L8 - E10'])}." if both
                                                         else ".")
                   + " Its costs stay within the bars, read on point estimates: at most 2 points on computed values, "
                   "final answers and copied numbers; at most +0.01 bits per byte outside the copy; a patch rate within "
                   "0.005; at most 4 new-value removals of about 1,700 computed-value starts at the default rate, and "
                   "none at 10%." + ("" if both else " At 10% the onset-removal bound could not be told from zero, so "
                                     "the fix was not tested at that rate."))
    elif fix["scope"].startswith("failed: K7"):
        out += k7_sentences(B, fix["scope"], n1)
    if any(k.startswith("K6") for k in V["kills"]):
        out.append("The cost bars could not see V_8's known cost; no claim either way.")
    if fix["scope"].startswith("not reached") and not stops:
        out.append(f"The fix claim was {fix['scope']}.")
    g = V["generality_claim"]
    if g["holds"] is True:
        names = {"F2b's reworded context (G1)": "the context reworded mid-line"}
        ids = [DISPLAY[k] for k, v in B.get("G3", {}).items() if v["applies"]]
        parts = ([names["F2b's reworded context (G1)"]] if B.get("G1", {}).get("testable") else []) + \
            ([(", ".join(ids[:-1]) + " and " if len(ids) > 1 else "") + ids[-1] + " in the same `\\nTicket` template"]
             if ids else [])
        out.append(f"In BLT-1B with its 512-byte local windows, the fix also holds for {'; and '.join(parts)}, at the "
                   "default rate.")
    elif g["holds"] is None and g["scope"].startswith("not testable"):
        out.append("No other wording or ID kind showed the hazard, so generality was not tested.")
    elif g["holds"] is False:
        k9 = [k for k in V["kills"] if k.startswith("K9")]
        if not fix["holds"]:                                        # K9's sentence prints only while the fix claim holds
            out.append("No generality claim: the fix claim " + ("is void (K6)." if fix["scope"].startswith("void")
                                                                else "failed."))
        elif k9:
            for k in k9:
                m = re.search(r"extend to (\S+) IDs", k)
                cell = m.group(1) if m else "F2b"
                n = J["j4"]["F2b"]["n"] if cell == "F2b" else J["j5"]["F3"]["by_kind"][cell]["n"]
                what = "the context reworded mid-line (F2b's template)" if cell == "F2b" else DISPLAY[cell]
                lead = TEMPLATE.format(n=n) if cell != "F2b" else \
                    f"Scored teacher-forced with 512-byte windows ({n} texts), "
                out.append(lead + f"at the default rate, the fix does not extend to {what}.")
        else:
            out.append(f"No generality claim: {', '.join(g['failing'])}.")
    elif not stops:
        out.append(f"The generality claim was {g['scope']}.")
    for w in ("windowed", "released"):
        if V["paired"][w]["class"] != "not reached":
            out.append(f"Paired verdict, {w}: {V['paired'][w]['reading']}")
    out.append(NOT_NEEDED)
    return out


def earlier_filing(filing):
    """The job outputs the filing commit's own tree holds. A re-filing commit carries the earlier filing's outputs; only
    those written under the filing the earlier J1 records are kept, so two earlier filings are never combined."""
    J = {}
    for k in ("j1", "j2", "j3", "j4", "j5", "j6", "j7"):
        r = git("show", f"{filing}:{os.path.relpath(out_path(k), _seg.HERE)}")
        if r.returncode == 0:
            J[k] = json.loads(r.stdout)
    first = J.get("j1", {}).get("code", {}).get("filed_commit")
    return {k: v for k, v in J.items() if first and v.get("code", {}).get("filed_commit") == first}


def flat(x, path=""):
    if isinstance(x, dict):
        return {k: v for kk, vv in x.items() for k, v in flat(vv, f"{path}/{kk}" if path else str(kk)).items()}
    return {path: x}


def differences(V, W):
    """Every bar, claim, class, kill or stop in which verdict V (the re-filed outputs) differs from W (the earlier)"""
    keys = ("bars", "fix_claim", "generality_claim")
    fv, fw = flat({k: V.get(k) for k in keys}), flat({k: W.get(k) for k in keys})
    out = {p: {"earlier": fw.get(p), "re-filed": fv.get(p)} for p in sorted(set(fv) | set(fw)) if fv.get(p) != fw.get(p)}
    for k in ("kills", "stops"):
        if V[k] != W[k]:
            out[k] = {"earlier": W[k], "re-filed": V[k]}
    for w in ("windowed", "released"):
        a, b = W["paired"][w]["class"], V["paired"][w]["class"]
        if a != b:
            out[f"paired/{w}/class"] = {"earlier": a, "re-filed": b}
    return out


def result_of(V, J, earlier, filing):
    """Which result counts (the rule's 'Which result counts'), and the text `verdict` prints for it. Without an earlier
    filing: this filing's chain. After a J1 check failure under the earlier filing (the defined redo; J1 exits before
    any decision-relevant number): the re-filed chain, with the differences beside it. After any other re-filing (a
    J7 check failure, or a bug found once scoring had started): the earlier filing's outputs remain the confirmatory
    result, and this re-filed chain is a deviation, printed beside it under that label."""
    if not earlier:
        where = f"filing {filing}" if filing else "smoke run, dev problems"
        return {"counts": "this filing's chain", "filing": filing}, [f"Confirmatory result ({where})", *V["sentences"]]
    prior = earlier["j1"]["code"]["filed_commit"]
    if any("implementation check failed" in s for s in earlier["j1"].get("stop", [])):
        return ({"counts": "the re-filed chain (J1 redo)", "filing": filing, "earlier filing": prior},
                ["Result: re-filed chain (J1 redo)", *V["sentences"]])
    W = verdict_from(earlier)
    S = sentences(W, earlier)
    return ({"counts": "the earlier filing's chain", "filing": prior,
             "deviation": f"corrected rerun (re-filing {filing})",
             "confirmatory result": {"fix_claim": W["fix_claim"], "generality_claim": W["generality_claim"],
                                     "stops": W["stops"], "kills": W["kills"],
                                     "paired": {w: W["paired"][w]["class"] for w in ("windowed", "released")},
                                     "sentences": S}},
            [f"Confirmatory result (filing {prior})", *S, f"Deviation: corrected rerun (re-filing {filing})",
             *V["sentences"]])


def cmd_verdict():
    why = gate("verdict")
    if why:
        sys.exit(f"refused: {why}")
    J = {k: read_out(k) for k in ("j1", "j2", "j3", "j4", "j5", "j6", "j7") if read_out(k) is not None}
    code = provenance()
    V = {"code": code, **verdict_from(J)}
    V["sentences"] = sentences(V, J)
    earlier = {}
    if not SMOKE:
        earlier = earlier_filing(code["filed_commit"])
        if earlier:
            prior = earlier["j1"]["code"]["filed_commit"]
            V["earlier filing"] = {"filing commit": prior, "differences": differences(V, verdict_from(earlier))}
            V["rules file changes since the earlier filing"] = git("diff", "--stat", prior, "HEAD", "--", RULES).stdout
        V["rules file changes since filing"] = git("diff", "--stat", code["filed_commit"], "HEAD", "--", RULES).stdout
    V["result"], text = result_of(V, J, earlier, code["filed_commit"])
    json.dump(V, open(out_path("verdict"), "w"), indent=1, default=str)
    print(json.dumps(V, indent=1, default=str))
    print("\n".join(text))
    print("wrote", out_path("verdict"))


# ---------------------------------------------------------------- self-test (bytes only, dev problems)
def gate_selftest():
    """The job gate on a scratch repository with a remote, the smoke outputs standing in for job outputs: what it lets
    run and what it refuses, under a filing and after a two-commit re-filing"""
    smoke = {k: json.load(open(os.path.join(RES, f"blt_copy_latch_fresh_{k}_smoke.json")))
             for k in ("blocks", "j1", "j2", "j3", "j4", "j5", "j6", "j7")}
    g, keep = globals(), (_seg.HERE, code_paths, out_path, model_record, SMOKE, os.environ.get("O6B_FILED"))
    got = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "work")
        os.makedirs(os.path.join(work, "followup")); os.makedirs(os.path.join(work, "results"))
        run = lambda *a, at=work: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                                                 cwd=at, capture_output=True, text=True).stdout.strip()
        put = lambda path, text: open(os.path.join(work, path), "w").write(text)

        def commit(rules=None, push=True):
            if rules is not None:
                put(RULES, rules)
            run("add", "-A"); run("commit", "-qm", "c")
            if push:
                run("push", "-q")
            return run("rev-parse", "HEAD")
        run("init", "-q", "--bare", os.path.join(tmp, "origin.git"), at=tmp)
        run("init", "-q", "-b", "main"); run("remote", "add", "origin", os.path.join(tmp, "origin.git"))
        put("code.py", "x = 1\n"); put(".gitignore", "*.npz\n")
        code0 = commit("# rules\n", push=False); run("push", "-q", "-u", "origin", "main")
        pinned = f"# rules\n### Code\n- realblt_copy_latch_fresh.py, at commit {code0}\n"
        commit(pinned)
        filing = commit(pinned + "## Confirmatory rule, filed (2026-11-02)\nApproved: the PI's words.\n")
        try:
            _seg.HERE, g["SMOKE"] = work, False
            g["code_paths"] = lambda: ["code.py"]
            g["out_path"] = lambda name, ext="json": os.path.join(work, "results", f"blt_copy_latch_fresh_{name}.{ext}")
            g["model_record"] = lambda: {"model": MODEL, "snapshot": SNAPSHOT, "weights_blob": "b", "transformers": "t",
                                         "torch": "t", "numpy": "n", "modeling_blt_sha256": "m", "device": "cpu"}

            def file_under(c):
                os.environ["O6B_FILED"] = c
                PROVENANCE.clear()
                return provenance()

            def write(name, content, code):
                json.dump({**content, "code": code}, open(out_path(name), "w"))
                return commit()
            rel = lambda name: os.path.relpath(out_path(name), work)
            code = file_under(filing)
            open(out_path("patcher", "npz"), "wb").write(b"patcher outputs")
            write("blocks", smoke["blocks"], code)
            write("j1", {**smoke["j1"], "stop": [], "patcher_sha256": sha256(out_path("patcher", "npz"))}, code)
            before_j2 = run("rev-parse", "HEAD")
            assert gate("j2") is None, gate("j2")
            j2 = copy.deepcopy(smoke["j2"])
            j2["F1"]["pairs"]["D-all vs D"]["id_copy"]["exact"] = [0.76, [0.70, 0.81]]
            write("j2", j2, code)
            got["J2 again under the same filing"] = gate("j2")
            os.remove(out_path("j2"))
            got["J2 after its output was removed with rm"] = gate("j2")
            run("checkout", "--", rel("j2"))
            run("reset", "-q", "--hard", before_j2)
            got["J2 after a reset to before its output (pushed, then reset)"] = gate("j2")
            run("reset", "-q", "--hard", "@{upstream}")
            run("rm", "-q", rel("j2")); commit()
            got["J2 after git rm, committed and pushed"] = gate("j2")
            run("checkout", "HEAD~1", "--", rel("j2")); commit()
            for k in ("j3", "j4", "j5", "j6", "j7"):
                write(k, smoke[k], code)
            put("code.py", "x = 2\n"); fix = commit()
            refiling = commit(pinned.replace(code0, fix) + "## Confirmatory rule, filed (2026-11-02)\nApproved: the "
                              "PI's words.\n## Confirmatory rule, filed (2026-11-03)\nThe fix, disclosed. Approved: the "
                              "PI's words.\n")
            code_b = file_under(refiling)
            write("blocks", smoke["blocks"], code_b)
            write("j1", {**smoke["j1"], "stop": [], "patcher_sha256": sha256(out_path("patcher", "npz"))}, code_b)
            got["J3 after the re-filing, with J2 from the earlier filing"] = gate("j3")
            got["`verdict` after the re-filing, with J2-J7 from the earlier filing"] = gate("verdict")
            assert gate("j2") is None, gate("j2")
            earlier = earlier_filing(refiling)
            now = {k: (read_out(k) if k == "j1" else copy.deepcopy(earlier[k])) for k in earlier}
            now["j2"]["F1"]["pairs"]["D.L8 vs D"]["id_copy"]["exact"] = [0.10, [0.05, 0.15]]
            diff = differences(verdict_from(now), verdict_from(earlier))
            Vn = verdict_from(now)
            Vn["sentences"] = sentences(Vn, now)
            redo = {"j1": {**copy.deepcopy(earlier["j1"]), "checks": {**earlier["j1"]["checks"], "passed": False},
                           "stop": ["implementation check failed"]}}
            j7fail = {**copy.deepcopy(earlier), "j7": {"status": STATUS, "code": earlier["j7"]["code"], "window": 0,
                                                       "checks": {"passed": False},
                                                       "stop": ["implementation check failed (J7)"]}}
            which = {b: result_of(Vn, now, e, refiling) for b, e in (("a J1 redo", redo), ("a J7 check failure", j7fail),
                                                                     ("another bug", earlier))}
        finally:
            _seg.HERE, g["code_paths"], g["out_path"], g["model_record"], g["SMOKE"], filed_env = keep
            PROVENANCE.clear()
            os.environ.pop("O6B_FILED", None)
            if filed_env is not None:
                os.environ["O6B_FILED"] = filed_env
    assert all(got.values()), got
    assert sorted(earlier) == ["j1", "j2", "j3", "j4", "j5", "j6", "j7"] and "bars/C1/value" in diff, diff
    r, text = which["a J1 redo"]
    assert r["counts"] == "the re-filed chain (J1 redo)" and text[0] == "Result: re-filed chain (J1 redo)", r
    for b in ("a J7 check failure", "another bug"):
        r, text = which[b]
        assert r["counts"] == "the earlier filing's chain" and r["filing"] == filing, r
        assert text[0] == f"Confirmatory result (filing {filing})", text
        assert f"Deviation: corrected rerun (re-filing {refiling})" in text, text
    assert which["a J7 check failure"][0]["confirmatory result"]["paired"]["released"] == "not reached"
    return {"lets J2 run under the filing": True, "refuses": got,
            "differences listed against the earlier filing (one changed bar)": sorted(diff),
            "which result counts after a re-filing": {b: v[0]["counts"] for b, v in which.items()}}


def dev_problems(path, n=300):
    with open(path) as f:
        return [json.loads(line) for line in itertools.islice(f, n)]


def cmd_selftest():
    tok = AutoTokenizer.from_pretrained(MODEL)
    keep = keeper(tok)
    res = {}
    # 1. the generic labeller reproduces realblt_quadrant.build on O6's 588 dev texts (same draws)
    rng = np.random.default_rng(0)
    same = total = 0
    for path in (GSM_TRAIN, GSM):
        for p in dev_problems(path):
            b, units = build(p, rng)
            if not keep(b):
                continue
            tid = b[units["id_copy"][0][0]:units["id_copy"][0][1]].decode()
            b2, u2, _ = make(p, None, "F1", x=tid)
            total += 1
            same += int(b2 == b and u2 == units)
    assert same == total, f"generic F1 builder differs from build on {total - same} of {total}"
    res["F1 generic = build"] = f"{same} of {total}"
    # 2. F6 equals realblt_budget.problem's bytes
    ok = n6 = 0
    for p in dev_problems(GSM):
        pr = realblt_budget.problem(p, tok, "cpu")
        if pr is None:
            continue
        n6 += 1; ok += int(make_f6(p)[0] == pr[0])
    assert ok == n6
    res["F6 = realblt_budget.problem"] = f"{ok} of {n6}"
    # 3. every template on dev problems: the ID twice, units, context recurrence (V_8 at c1) as designed
    probs = dev_problems(GSM_TRAIN)
    for cell, kinds in (("F1", ("digits4",)), ("F2a", ("digits4",)), ("F2b", ("digits4",)), ("F3", F3_KINDS),
                        ("F4", ("digits4",))):
        for kind in kinds:
            rng = np.random.default_rng(1)
            n = v8 = 0
            for p in probs[:60]:
                x = make(p, rng, cell, kind) if cell != "F1" else make(p, rng, cell)
                if x is None:
                    continue
                b, units, info = x
                (a0, e0), (a1, e1) = units["id_new"][0], units["id_copy"][0]
                xb = info["id"].encode()
                assert b[a0:e0] == xb and b[a1:e1] == xb and e1 - a1 == len(xb)
                if kind == "dose":
                    assert len(re.findall(rb"(?<!\d)" + re.escape(xb) + rb"(?!\d)", b)) == 2
                else:
                    assert b.count(xb) == 2
                assert info["distance"] == a1 - a0
                for kd, v in units.items():
                    if kd in ("id_new", "id_copy"):
                        continue
                    for a, e in v:
                        assert e <= a0 or a >= e0, (cell, kind, kd)
                        assert e <= a1 or a >= e1, (cell, kind, kd)
                n += 1; v8 += int(vetoed(b, len(b) + 1, 8)[a1 + 1])
            res[f"{cell} {kind}"] = f"{n} texts; V_8 at c1 in {v8}"
            assert (v8 == 0) if cell == "F2a" else (v8 == n), (cell, kind, v8, n)
    # 4. the whole-number test
    cases = [(b"Rent = $7200/year. Rent = $720/year", b"720", False, True),
             (b"Total sold = 1,200,000 kg. Total sold = 1,200 kg", b"1,200", False, True),
             (b"each box = 150 eggs; each box = 150 eggs", b"150", True, True),
             (b"Price = $15.50 now. Price = $15 now", b"15", False, True),
             (b"Each box = 150, so far. Each box = 150 eggs", b"150", True, True),      # a trailing comma is not
             (b"Each box = 150 eggs. Each box = 150, so far", b"150,", True, False)]    # part of the value
    for b, v, want_whole, want_prefix in cases:
        t = b.rindex(v) + 1
        assert same_value(b, t, t - 1 + len(v)) == want_whole, b
        assert same_value_prefix(b, t, t - 1 + len(v)) == want_prefix, b
    res["whole-number test"] = f"{len(cases)} cases"
    # 5. C4's integer and the control's complement
    assert C4_MAX_D == 4 and isinstance(C4_MAX_D, int)
    for cv in np.linspace(-0.05, 0.02, 71):
        for bv in np.linspace(-0.005, 0.02, 101):
            holds = cv <= -0.02 or bv >= 0.01
            k6 = cv > -0.02 and bv < 0.01
            assert holds != k6
    res["control and K6 partition"] = "checked on a 71 x 101 grid"
    # 6. blocks on dev rows are deterministic and sized as the smoke run takes them
    global SMOKE
    was, SMOKE = SMOKE, True
    try:
        b1, t1 = take_blocks(pool_rows(), tok)
        b2, _ = take_blocks(pool_rows(), tok)
    finally:
        SMOKE = was
    assert b1 == b2
    res["dev blocks (smoke sizes)"] = {c: [len(v["fit"]), len(v["test"]), v["first_idx"], v["last_idx"]]
                                       for c, v in b1.items() if "first_idx" in v}
    res["F3 kinds in the smoke block"] = dict(collections.Counter(r["kind"] for r in b1["F3"]["test"]))
    # 7. the paired-verdict classifier
    assert classify((0.10, [0.05, 0.15]), 0.02) == "L8 better"
    assert classify((-0.10, [-0.15, -0.05]), 0.02) == "SIV+PO better"
    assert classify((0.01, [0.005, 0.02]), 0.02) == "tie" and classify((0.05, [-0.01, 0.1]), 0.02) == "tie"
    res["classifier"] = "4 cases"
    # 8. C1e (i) on exact counts. boot_ratio's point is (k_a - k_b) / n, a correctly rounded k / n; with the tolerance
    #    the test agrees with 10 k1 >= 9 k2 at every pair of counts at n = 600
    n = 600
    assert boot_ratio([108], [0], [n])[0] == 108 / n
    pairs = [(k1, k2) for k2 in range(1, n + 1) for k1 in range(0, n + 1)]
    bad = [(k1, k2) for k1, k2 in pairs if (k1 / n >= 0.9 * (k2 / n) - 1e-9) != (10 * k1 >= 9 * k2)]
    assert not bad, bad[:5]
    ties = sum(1 for k2 in range(10, n + 1, 10) if not 9 * k2 // 10 / n >= 0.9 * (k2 / n))
    res["C1e (i) exact counts"] = (f"{len(pairs)} pairs at n = {n} agree with 10 k1 >= 9 k2; without the tolerance "
                                   f"{ties} of the {n // 10} exact ties failed")
    # 9. the filed-rule guard: none of these may pass, and no line of the rules file is a filing heading yet
    assert filed_headings("x\n## Confirmatory rule, filed (2026-11-02)\ny") == ["## Confirmatory rule, filed (2026-11-02)"]
    for text in ("the heading `## Confirmatory rule, filed (2026-11-02)` once approved",
                 "## Confirmatory rule, filed\n", "```\n## Confirmatory rule, filed (2026-11-02)\n```\n",
                 "<!--\n## Confirmatory rule, filed (2026-11-02)\n-->\n", "## Confirmatory rule, filed (2026-10-29)\n",
                 "## Confirmatory rule, filed (2026-13-45)\n"):
        assert not filed_headings(text), text
    assert not filed_headings(open(os.path.join(_seg.HERE, RULES)).read())
    refused = {x: filed_check(x)[1] for x in ("", "HEAD", "HEAD^{tree}", ":0", "4b76f51", "no-such-commit")}
    assert all(refused.values()), refused
    res["guard refuses here"] = refused
    res["guard covers"] = {sub or ".": [rev[:12], ps] for sub, (rev, ps) in by_repo("HEAD", code_paths()).items()}
    assert any(p.endswith("final_rules.py") for _, ps in by_repo("HEAD", code_paths()).values() for p in ps)
    res["model"] = model_record()
    # 10. the guard in a scratch repository with a submodule: it accepts a filing or re-filing commit whose code is
    #     the pinned code, unchanged since, and nothing else
    here, paths_of = _seg.HERE, code_paths
    with tempfile.TemporaryDirectory() as tmp:
        sh = lambda *a, at="": subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                                              cwd=os.path.join(tmp, at), capture_output=True, text=True).stdout.strip()
        put = lambda path, text: open(os.path.join(tmp, path), "w").write(text)

        def commit(rules=None):
            if rules is not None:
                put(RULES, rules)
            sh("add", "-A"); sh("commit", "-qm", "c")
            return sh("rev-parse", "HEAD")
        sh("init", "-q", "-b", "main"); os.makedirs(os.path.join(tmp, "followup")); put("code.py", "x = 1\n")
        os.makedirs(os.path.join(tmp, "sub")); sh("init", "-q", at="sub"); put("sub/lib.py", "z = 1\n")
        sh("add", "-A", at="sub"); sh("commit", "-qm", "s", at="sub")                 # 'sub' enters as a gitlink
        code0 = commit("# rules\n")
        pinned = f"# rules\n### Code\n- realblt_copy_latch_fresh.py, at commit {code0}\n"
        heading = "## Confirmatory rule, filed (2026-11-02)\n"
        early = {"no heading": commit(pinned),
                 "a heading quoted in a sentence": commit(pinned + f"It becomes `{heading.strip()}`.\n"),
                 "a heading inside a code fence": commit(pinned + f"```\n{heading}```\n"),
                 "a heading inside an HTML comment": commit(pinned + f"<!--\n{heading}-->\n"),
                 "a heading dated before 2026-10-30": commit(pinned + "## Confirmatory rule, filed (2026-10-29)\n")}
        base = sh("rev-parse", "HEAD")
        sh("checkout", "-q", "-b", "side"); early["a side branch"] = commit(pinned + heading)   # never merged
        sh("checkout", "-q", "main")
        put("code.py", "x = 9\n"); early["a filing commit that also edits code"] = commit(pinned + heading)
        put("code.py", "x = 1\n"); commit(pinned)
        sh("checkout", "-q", "-b", "other"); commit(pinned + heading)
        sh("checkout", "-q", "main"); put("other.txt", "o\n"); commit()
        sh("merge", "-q", "--no-ff", "-m", "m", "other"); early["a merge"] = sh("rev-parse", "HEAD")
        commit(pinned)
        filed_text = pinned + heading + "Approved: the PI's words.\n"
        filed = commit(filed_text)
        mine = ["code.py", "sub/lib.py"]
        try:
            _seg.HERE, globals()["code_paths"] = tmp, lambda: mine
            got = {k: filed_check(c)[1] for k, c in early.items()}
            got["the index (:0)"] = filed_check(":0")[1]
            ok = [filed_check(filed)]
            put("code.py", "x = 2\n"); got["code changed after filing, not committed"] = filed_check(filed)[1]
            put("code.py", "x = 1\n"); put("sub/lib.py", "z = 2\n")
            got["submodule code changed after filing"] = filed_check(filed)[1]
            put("sub/lib.py", "z = 1\n")
            commit(filed_text + "A later note.\n")
            got["O6B_FILED=HEAD after a later commit"] = filed_check("HEAD")[1]
            ok.append(filed_check(filed))                           # a later commit that leaves the code alone
            put("code.py", "x = 2\n"); fix = commit(filed_text + "A later note.\n")
            got["the filing commit, with code changed in a later commit"] = filed_check(filed)[1]
            got["a later commit with changed code"] = filed_check(fix)[1]
            refile = filed_text.replace(code0, fix) + "A later note.\n## Confirmatory rule, filed (2026-11-03)\n" \
                "The fix, disclosed. Approved: the PI's words.\n"
            got["a re-filing commit whose pin is not its code"] = filed_check(commit(refile.replace(fix, code0)))[1]
            commit(filed_text + "A later note.\n")
            refiled = commit(refile)                                # the two-commit re-filing: the fix, then this
            ok.append(filed_check(refiled))
            put("new.py", "y = 1\n"); mine.append("new.py")
            got["an imported file the commit lacks"] = filed_check(refiled)[1]
        finally:
            _seg.HERE, globals()["code_paths"] = here, paths_of
    assert ok == [(filed, None), (filed, None), (refiled, None)] and all(got.values()), (ok, got)
    res["guard in a scratch repository"] = {"accepts": ["the filing commit", "the filing commit after a later commit "
                                                        "that leaves the code alone", "a two-commit re-filing"],
                                            "refuses": sorted(got)}
    # 11. the job gate in a scratch repository with a remote: each job once per filing (an output committed and pushed
    #     still counts after rm, a reset or git rm), and a re-filing reruns from J0 without reading the earlier
    #     filing's J2-J7, while `verdict` lists what differs from them and says which result counts
    res["job gate in a scratch repository"] = gate_selftest()
    # 12. a dose hidden by joining the problem's parts (train idx 71: its solution ends '$25' and its answer is '25'):
    #     draw() never returns it, while the joined test lets it through
    q, sol, ans = parts(dev_problems(GSM_TRAIN, 72)[71])
    joined = [d for d in DOSES if not re.search(rf"(?<!\d){d}(?!\d)", q + sol + ans)]
    drawn = {draw(np.random.default_rng(i), "dose", (q, sol, ans)) for i in range(400)}
    assert "25" in joined and "25" not in drawn and drawn == set(doses_allowed((q, sol, ans))), (sol[-30:], ans, drawn)
    res["doses tested by part"] = (f"train idx 71: draw() returned {len(drawn)} doses over 400 seeds, never 25; joined, "
                                   "'$25' and '25' read '$2525' and let it in")
    print(json.dumps(res, indent=1))
    json.dump({"status": "self-test of the confirmatory code on dev problems (bytes only; no model)", **res},
              open(os.path.join(RES, "blt_copy_latch_fresh_selftest.json"), "w"), indent=1)
    print("self-test passed")


# ---------------------------------------------------------------- devcheck (exploratory; stored outputs, no model)
def jeffreys_pass(k, n, N, allowed):
    """P(at most `allowed` events in N trials) with the rate's Jeffreys posterior Beta(k + 1/2, n - k + 1/2), and with
    the plug-in rate k/n (Poisson)"""
    grid = (np.arange(200000) + 0.5) / 200000 * min(1.0, 50.0 / max(N, 1))
    a, b = k + 0.5, n - k + 0.5
    logpdf = (a - 1) * np.log(grid) + (b - 1) * np.log1p(-grid)
    w = np.exp(logpdf - logpdf.max()); w /= w.sum()
    lam = grid * N
    cdf = sum(np.exp(-lam) * lam ** j / math.factorial(j) for j in range(allowed + 1))
    plug = k / n * N
    return {"jeffreys": float((w * cdf).sum()),
            "plug_in": float(sum(math.exp(-plug) * plug ** j / math.factorial(j) for j in range(allowed + 1)))}


def cmd_devcheck():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    out = {"status": "exploratory: stored dev outputs and dev problems (GSM8K idx 0-299) only; no model run"}
    C = L.load_cache(os.path.join(RES, "blt_copy_latch_patcher.npz"))
    for g, rows in C.items():
        add_masks(rows)
    print("masks", f"{time.time() - t0:.0f}s", flush=True)
    s1 = json.load(open(os.path.join(RES, "blt_copy_latch_step1.json")))

    # A. the dev recount of computed-value removals under the whole-number test
    eq_fit = blt_layouts.Fit((0.10,))
    for r in C["eq_train"]:
        eq_fit.add(L.sig_of(r))
    eq_fit.done()
    lay = lambda r: {"D": set(r["D"].tolist()),
                     "E10": set(blt_layouts.layout("entropy", 0.10, L.sig_of(r), eq_fit).tolist())}
    rules = ("L8", "L8c", "V8", "SIV", "PO", "SIVPO", "L8b")
    rc = {g: value_counts(C[g], lay, rules, keep_cases=("L8", "L8c", "SIVPO", "V8")) for g in ("eq_test", "eq_train")}
    out["A_recount"] = rc
    for key in ("D-L8", "E10-L8", "D-L8c", "D-V8"):
        assert rc["eq_test"][key]["removed"] == s1["b"][key]["removed"], key
        assert rc["eq_test"][key]["new_value_prefix_test"] == s1["b"][key]["new_value"], key
    x = rc["eq_test"]
    out["A_decisions"] = {
        "go (b) at D: L8 new-value removals (whole-number test), allowed 2 of 799": x["D-L8"]["new_value"],
        "go (b) at E10: L8 new-value removals (must be 0)": x["E10-L8"]["new_value"],
        "L8c trigger (more than 2 of 799)": x["D-L8"]["new_value"] > 2,
        "V8 at D, new / same (whole-number)": [x["D-V8"]["new_value"], x["D-V8"]["removed"] - x["D-V8"]["new_value"]],
        "V8 at D, new / same (prefix test, as reported)": [x["D-V8"]["new_value_prefix_test"],
                                                           x["D-V8"]["removed"] - x["D-V8"]["new_value_prefix_test"]],
        "removals where the two tests differ": {g: [{"rule": k, **c} for k in rc[g] if k.startswith(("D-", "E10-"))
                                                    for c in rc[g][k]["cases"] if c["new_value"] != c["new_value_prefix_test"]]
                                                for g in rc},
        "eq_train (E10 fitted in-sample on eq_train)": {key: {f: rc["eq_train"][key][f] for f in
                                                              ("in_layout", "removed", "new_value", "new_value_prefix_test")}
                                                        for key in ("D-L8", "E10-L8", "D-L8c", "D-V8", "D-SIVPO", "E10-SIVPO")}}

    # B. choice 2's numbers: pooled dev rate and F6's expected count of D's computed-value starts
    k = rc["eq_test"]["D-L8"]["new_value"] + rc["eq_train"]["D-L8"]["new_value"]
    n = rc["eq_test"]["D-L8"]["in_layout"] + rc["eq_train"]["D-L8"]["in_layout"]
    per = {g: rc[g]["D-L8"]["in_layout"] / len(C[g]) for g in ("eq_test", "eq_train")}
    per["pooled"] = n / (len(C["eq_test"]) + len(C["eq_train"]))
    Ns = {g: int(round(600 * v)) for g, v in per.items()}
    ke = rc["eq_test"]["E10-L8"]["new_value"] + rc["eq_train"]["E10-L8"]["new_value"]
    ne = rc["eq_test"]["E10-L8"]["in_layout"] + rc["eq_train"]["E10-L8"]["in_layout"]
    Ne = int(round(600 * ne / (len(C["eq_test"]) + len(C["eq_train"]))))
    out["B_choice2"] = {
        "pooled D": {"new_value": k, "in_layout": n, "rate": k / n},
        "D computed-value starts per text": per, "F6 expected D starts": Ns,
        "P(C4 at D holds: at most 4)": {g: jeffreys_pass(k, n, N, C4_MAX_D) for g, N in Ns.items()},
        "P(zero at D), O6's former bar, for reference": {g: jeffreys_pass(k, n, N, 0) for g, N in Ns.items()},
        "eq_test alone": {g: jeffreys_pass(rc["eq_test"]["D-L8"]["new_value"], rc["eq_test"]["D-L8"]["in_layout"], N, C4_MAX_D)
                          for g, N in Ns.items()},
        "E10 (K2)": {"new_value": ke, "in_layout": ne, "F6 expected E10 starts": Ne,
                     "P(K2) if E10's rate were D's pooled rate": 1 - math.exp(-k / n * Ne),
                     "P(K2), Jeffreys on E10's own 0 of n": 1 - jeffreys_pass(ke, ne, Ne, 0)["jeffreys"]}}
    onsets = []
    for g in ("eq_test", "eq_train"):
        for j, r in enumerate(C[g]):
            b = r["b"]
            for m in VALUE.finditer(b):
                for t in sorted({m.start() + 3, m.start(1) + 1}):
                    if r["L8"][t]:
                        onsets.append({"group": g, "text": j, "token": t, "value": m.group(1).decode(),
                                       "D_start": bool(t in set(r["D"].tolist())), "h_t-1_bits": float(r["hb"][t - 1]),
                                       "new_value": not same_value(b, t, m.end(1)),
                                       "context": b[max(0, t - 41):t - 1].decode(errors="replace")})
    out["B_choice2"]["L8 masks at computed-value positions"] = onsets

    # C. byte rules on dev rewrites (quad_test problems, their dev tickets; F3 and F4 with seeded IDs)
    probs = dev_problems(GSM)
    rng0 = np.random.default_rng(0)
    for p in dev_problems(GSM_TRAIN):                              # O6's draws: train tickets first
        build(p, rng0)
    tickets = []
    keep = keeper(tok)
    for p in probs:
        b, units = build(p, rng0)
        if keep(b):
            tickets.append((p, b[units["id_copy"][0][0]:units["id_copy"][0][1]].decode()))
    assert len(tickets) == 293
    rw = {}
    specs = [("F1", "digits4"), ("F2a", "digits4"), ("F2b", "digits4"), ("F4", "digits4")] + [("F3", k) for k in F3_KINDS]
    for cell, kind in specs:
        rng = np.random.default_rng(7)
        fire = collections.Counter(); whole = collections.Counter(); outside = collections.Counter(); nt = 0; dist = []
        for p, tid in tickets:
            x = make(p, rng, cell, kind, x=None if cell == "F3" else tid)
            if x is None:
                continue
            b, units, info = x
            m = masks_bytes(b)
            a, e = units["id_copy"][0]
            c = np.arange(a + 1, e + 1)
            nt += 1; dist.append(info["distance"])
            for rule, v in m.items():
                fire[rule] += int(v[c[0]]); whole[rule] += int(v[c].all())
                outside[rule] += int(v[2:].sum() - v[c].sum())
        rw[f"{cell} {kind}"] = {"texts": nt, "fires_c1": dict(fire), "masks_whole_copy": dict(whole),
                                "masked_tokens_outside_copy_per_text": {r: v / nt for r, v in outside.items()},
                                "distance_median": float(np.median(dist)),
                                "in_F4_window": int(sum(F4_DIST[0] <= d <= F4_DIST[1] for d in dist))}
        print("rewrites", cell, kind, rw[f"{cell} {kind}"]["fires_c1"], f"{time.time() - t0:.0f}s", flush=True)
    out["C_rewrites"] = rw

    # D. natural text: footprint and E10 cost of SIV+PO against L8 on the stored xdomain outputs
    stored = {dom: np.load(os.path.join(RES, f"xdomain_gains_{dom}.npz"), allow_pickle=True) for dom in L.DOMAINS}
    xd = {}
    for dom in L.DOMAINS:
        docs = [r for r in C["xd"] if r["dom"] == dom]
        for subset in ("all", "test"):
            ds = docs if subset == "all" else [r for r in docs if r["key"].startswith("test")]
            for rule in ("L8", "SIVPO", "SIV", "PO"):
                xd[f"{dom}/{subset}/{rule}"] = xd_rule(ds, stored[dom], rule)
    for key, v in s1["xdomain_E10"].items():                       # L8 reproduces step 1 (D-proxy aside)
        if key.endswith("/real"):
            mine = xd[key.replace("/real", "/L8")]
            for f in ("footprint", "E10_masked_per_kB", "net_bpb_E10"):
                assert abs(mine[f] - v[f]) < 1e-12, (key, f)
    out["D_xdomain"] = xd

    # E. far dev copies: the patcher's argmax at c1 beyond 512 bytes (blt_copy_gate.json)
    gate = json.load(open(os.path.join(RES, "blt_copy_gate.json")))["rows"]
    far = [r for r in gate if r["distance"] > 512]
    out["E_far_copies"] = {"n": len(far), "argmax_ok_c1": int(sum(r["patcher_ok"][0] for r in far)),
                           "distances": [r["distance"] for r in far], "D_exact": int(sum(r["D"]["exact"] for r in far)),
                           "D-all_exact": int(sum(r["D-all"]["exact"] for r in far))}

    # F. step 3's reproduction of O6's rows (low finding L2), computed and stored
    s3 = json.load(open(os.path.join(RES, "blt_copy_latch_step3.json")))["rows"]
    o6 = json.load(open(os.path.join(RES, "blt_copy_veto.json")))["rows"]
    out["F_step3_G0"] = {name: {"exact_equal": int(sum(a["layouts"][name]["kinds"]["id_copy"][1] == b_["layouts"][name]["kinds"]["id_copy"][1]
                                                       for a, b_ in zip(s3, o6))),
                                "bits_within_0.05": float(np.mean([abs(a["layouts"][name]["kinds"]["id_copy"][2]
                                                                       - b_["layouts"][name]["kinds"]["id_copy"][2]) < 0.05
                                                                   for a, b_ in zip(s3, o6)])),
                                "of": len(s3)} for name in ("E10", "E10.V8")}

    # G. the causality check's power: a planted read of h(t) instead of h(t-1) must be caught
    real = L.onset

    def peek(b, hb, arg, t, agree="real"):
        code, T, strict, e = real(b, hb, arg, t, agree)
        if code == 0 and t < len(hb) and hb[t] < 0.5:              # a rule that also looks at the output at token t
            return 6, T, False, e
        return code, T, strict, e
    L.onset = peek
    try:
        caught = sum(L.causal_changes(r["b"], r["hb"], r["arg"], seed=i)[0] for i, r in enumerate(C["quad_test"][:20]))
    finally:
        L.onset = real
    out["G_planted_peek"] = {"texts": 20, "changed_decisions": int(caught), "caught": caught > 0}
    assert caught > 0, "the causality check missed a planted read of the output at token t"

    # I. dev step 2: SIV against L8, paired (the scored dev rule closest to SIV+PO on F1's template)
    s2 = json.load(open(os.path.join(RES, "blt_copy_latch_step2.json")))["rows"]
    for r in s2:
        for y in r["layouts"].values():
            y.setdefault("copy_len", 4)
    sm = summarize(s2, ["D.L8", "D.SIV", "E10.L8", "E10.SIV"], [("D.L8", "D.SIV"), ("E10.L8", "E10.SIV")])
    out["I_dev_L8_vs_SIV"] = {"pairs": sm["pairs"], "paired_rows": paired_rows(sm, rival="SIV")}

    # H. the dev causality evidence, restated
    ntok = sum(len(r["b"]) + 1 for rows in C.values() for r in rows)
    ntext = sum(len(rows) for rows in C.values())
    out["H_causality_restated"] = {"texts": ntext, "distinct_L8_L8c_decisions": 2 * ntok,
                                   "boundary_comparisons": 4 * (ntok - 2 * ntext),
                                   "all_comparisons": sum(s1["causality_check"]["decisions_checked"].values()),
                                   "changed": sum(s1["causality_check"]["changed_decisions"].values())}
    out["seconds"] = time.time() - t0
    path = os.path.join(RES, "blt_copy_latch_devcheck.json")
    json.dump(out, open(path, "w"), indent=1)
    brief = {k: v for k, v in out.items() if k not in ("A_recount", "D_xdomain")}
    print(json.dumps(brief, indent=1, default=str)[:20000])
    print("wrote", path, "(exploratory)")


def xd_rule(docs, z, rule):
    """[S2]'s E10 cost (realblt_copy_latch.xd_cost) for any mask; D starts masked counted with the stored D starts
    (the patcher's own layout), not a float threshold (low finding L3)"""
    nbytes = sum(len(r["b"]) for r in docs)
    fp = npos = vb = vD = 0; vb_gain = 0.0; pool = []
    for r in docs:
        v = r[rule].copy(); v[:2] = False
        base = z[f"{r['key']}_base"].copy(); base[:2] = False
        sn = z[f"{r['key']}_entropy"]
        gb = np.nan_to_num(z[f"{r['key']}_gain_bits"])
        fp += int(v.sum()); npos += len(r["b"]) - 1
        hit = v & base
        vb += int(hit.sum()); vb_gain += float(gb[hit].sum())
        vD += int(v[r["D"][r["D"] >= 2]].sum())
        free = np.flatnonzero(~base & ~v)
        free = free[free >= 2]
        pool.append(np.c_[sn[free], gb[free]])
    P = np.concatenate(pool)
    top = P[np.argsort(-P[:, 0])[:vb]] if vb else np.zeros((0, 2))
    return {"docs": len(docs), "footprint": fp / npos, "E10_masked_per_kB": vb / (nbytes / 1000),
            "D_masked_per_kB": vD / (nbytes / 1000), "net_bpb_E10": (vb_gain - float(top[:, 1].sum())) / nbytes}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = {"j2": [("F1", F1_NAMES, F1_PAIRS)], "j3": [("F6", F6_NAMES, F6_PAIRS)],
            "j4": [("F2a", F2A_NAMES, F2A_PAIRS), ("F2b", F2B_NAMES, F2B_PAIRS)], "j5": [("F3", F3_NAMES, F3_PAIRS)],
            "j6": [("F4", F3_NAMES, F3_PAIRS)]}
    if cmd == "selftest":
        cmd_selftest()
    elif cmd == "devcheck":
        cmd_devcheck()
    elif cmd == "blocks":
        cmd_blocks()
    elif cmd == "j1":
        cmd_j1()
    elif cmd in jobs:
        run_job(cmd, jobs[cmd])
    elif cmd == "j7":
        cmd_j7()
    elif cmd == "verdict":
        cmd_verdict()
    else:
        sys.exit(__doc__)
