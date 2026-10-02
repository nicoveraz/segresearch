"""Results registry: one JSON line per result, parsed from the RESULT lines the scripts print.

Every experiment script prints RESULT lines (mathexp.py, reasonexp.py, realtext.py, the BLT-1B scripts). This
module turns any of them into a flat record, so the same parser serves live runs and the backfill from old logs
(build_registry.py). Standard library only: it is also imported by the BLT-1B scripts' PyTorch environment.

    import registry
    registry.emit("mathexp", f"RESULT ...", corpus="math", window=32)     # prints the line and records it

Records go to results/registry/<script>.jsonl (live, appended) or results/registry/backfill.jsonl (rebuilt).
Common fields: script, origin (live | backfill), commit, time, source (log path, backfill only), raw (the line),
plus the parsed fields below and any metadata passed in.
"""
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(ROOT, "results", "registry")
NUM = r"(-?\d+(?:\.\d+)?|nan)"


def _f(x):
    return float("nan") if x == "nan" else float(x)


def _pct(x):
    return round(_f(x) / 100, 6)


def parse(line):
    """Parse one RESULT line into a flat dict. Accuracies and coverages are fractions in [0, 1]."""
    line = line.strip()
    if not line.startswith("RESULT "):
        raise ValueError(f"not a RESULT line: {line[:60]}")
    body = line[len("RESULT "):]
    out = {}
    if re.match(r"(math|code|gsm8k) ", body) and ("rule=" in body or "mask=" in body):
        head, *segs = body.split(" | ")
        kind, *toks = head.split(" ")
        out["format"] = "mathexp" if kind == "math" else "realtext"
        if kind != "math":
            out["data"] = kind
        for k, v in re.findall(r"(\w+)=([^\s(]+)", head):
            k = {"mask": "rule"}.get(k, k)
            out[k] = int(v) if re.fullmatch(r"\d+", v) else (_f(v) if re.fullmatch(NUM, v) else v)
        m = re.search(r"patches " + NUM + r", scratchpads " + NUM, head)
        if m:
            out["patch_rate"], out["scratch_rate"] = _f(m.group(1)), _f(m.group(2))
        for seg in segs:
            for name, bits, acc, n in re.findall(r"(\w+): " + NUM + r" bits, acc " + NUM + r" \(n=(\d+)\)", seg):
                out[f"{name}_bits"], out[f"{name}_acc"], out[f"{name}_n"] = _f(bits), _f(acc), int(n)
            m = re.fullmatch(r"ans " + NUM + r" \(computed " + NUM + r", final " + NUM + r"\)", seg.strip())
            if m:
                out["ans_bits"], out["computed_bits"], out["final_bits"] = map(_f, m.groups())
            m = re.fullmatch(r"(copy|text|bpb|rate) " + NUM, seg.strip())
            if m:
                out[{"copy": "copy_bits", "text": "text_bits"}.get(m.group(1), m.group(1))] = _f(m.group(2))
            m = re.fullmatch(r"(\d+)s", seg.strip())
            if m:
                out["seconds"] = int(m.group(1))
        return out
    m = re.match(r"layout=(\S+)\s+bits/answer byte " + NUM + r" \| exact match " + NUM + "%", body)
    if m:                                                                  # realblt.py
        return {"format": "blt", "layout": m.group(1), "answer_bits": _f(m.group(2)), "answer_exact": _pct(m.group(3))}
    layout, rest = re.match(r"(\S+)\s+(.*)", body).groups()                # realblt_budget / _code / _reason
    out = {"format": "blt", "layout": layout}
    m = re.fullmatch(r"(\w+)@(\d+)", layout)
    if m:
        out["kind"], out["budget"] = m.group(1), int(m.group(2)) / 100
    else:
        out["kind"] = layout
    for seg in rest.split(" | "):
        seg = seg.strip()
        if m := re.fullmatch(r"patch rate " + NUM, seg):
            out["rate"] = _f(m.group(1))
        elif m := re.fullmatch(r"(?:(results) )?covered " + NUM + "%", seg):
            out[f"{m.group(1) or 'target'}_covered"] = _pct(m.group(2))
        elif m := re.fullmatch(r"([\w\- ]+): (.*)", seg):
            name = m.group(1).strip().replace("in-line results", "results").replace("repeated identifiers", "ident").replace(" ", "_")
            for part in m.group(2).split(", "):
                if p := re.fullmatch(r"covered " + NUM + "%", part):
                    out[f"{name}_covered"] = _pct(p.group(1))
                elif p := re.fullmatch(NUM + " bits", part):
                    out[f"{name}_bits"] = _f(p.group(1))
                elif p := re.fullmatch(r"(exact|choice)\s+" + NUM + "%", part):
                    out[f"{name}_{p.group(1)}"] = _pct(p.group(2))
    if "ident_covered" not in out and "target_covered" in out and "ident_bits" in out:
        out["ident_covered"] = out.pop("target_covered")
    return out


def commit():
    try:
        h = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except OSError:
        return None


def record(script, line, **meta):
    """Append the parsed RESULT line to results/registry/<script>.jsonl. Never raises: a registry problem must not
    lose a finished run (the line is already printed)."""
    try:
        rec = {"script": script, "origin": "live", "commit": commit(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **meta, **parse(line), "raw": line.strip()}
        os.makedirs(DIR, exist_ok=True)
        with open(os.path.join(DIR, f"{script}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:                                                  # noqa: BLE001
        print(f"registry: could not record ({e})", flush=True)


def emit(script, line, **meta):
    """Print a RESULT line and record it: the hook the experiment scripts call."""
    print(line, flush=True)
    record(script, line, **meta)


def save_items(name, data, **meta):
    """Save per-target results (lists of correct/incorrect per layout, with the problem each target came from) to
    results/items/<name>.json, for paired tests across layouts (stats.py). Never raises."""
    try:
        d = os.path.join(ROOT, "results", "items"); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.json"), "w") as f:
            json.dump({"commit": commit(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta, "data": data}, f)
    except Exception as e:                                                  # noqa: BLE001
        print(f"registry: could not save items ({e})", flush=True)


def load(pattern=None):
    """All records from results/registry/*.jsonl (optionally only files whose name contains `pattern`)."""
    rows = []
    if os.path.isdir(DIR):
        for fn in sorted(os.listdir(DIR)):
            if fn.endswith(".jsonl") and (pattern is None or pattern in fn):
                rows += [json.loads(l) for l in open(os.path.join(DIR, fn)) if l.strip()]
    return rows
