"""Progress bar, cost and ETA for the scaling pilot (#27), from the work directory.  python3 progress.py [WORK]

Each planned run counts as work proportional to its measured training time (bench.json), plus ~25% for evaluation;
a running run counts by its fraction of updates done. The ETA extrapolates the progress made since the 50M stage
started. Cost uses the pod's hourly rate and the session start in session.log.
"""
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

W = sys.argv[1] if len(sys.argv) > 1 else "/workspace/work"
RATE = 0.497                                   # $/h
PLAN = [(s, a, sd) for s in ("1m", "12m") for a in ("entropy10", "dep10", "syntax+entropy10") for sd in (0, 1, 2)] + \
       [("50m", a, sd) for a in ("entropy10", "dep10") for sd in (0, 1, 2)] + [("50m", "syntax+entropy10", sd) for sd in (0, 1)]
NAMES = {"entropy10": "entropy", "dep10": "dependence", "syntax+entropy10": "hand-written"}
EVAL = 0.25

bench = json.load(open(f"{W}/bench.json"))
hours = {s: bench[s]["hours_per_run"] * (1 + EVAL) for s in bench}
log = open(f"{W}/session.log").read()


def when(tag):
    m = re.search(rf"== {tag} (\w+ \w+ +\d+ [\d:]+ UTC \d+)", log)
    return datetime.strptime(" ".join(m.group(1).split()), "%a %b %d %H:%M:%S UTC %Y").replace(tzinfo=timezone.utc) if m else None


def frac(size, arm, seed):
    if os.path.exists(f"{W}/runs/{size}_{arm}_s{seed}/result.json"):
        return 1.0
    f = f"{W}/logs/{size}_{arm}_s{seed}.log"
    if not os.path.exists(f):
        return 0.0
    ups = re.findall(r"update +(\d+)/(\d+)", open(f).read())
    return (int(ups[-1][0]) / int(ups[-1][1])) / (1 + EVAL) if ups else 0.0


now = datetime.now(timezone.utc) if len(sys.argv) < 3 else datetime.fromisoformat(sys.argv[2])
total = sum(hours[s] for s, _, _ in PLAN)
done = sum(hours[s] * frac(s, a, sd) for s, a, sd in PLAN)
done50 = sum(hours["50m"] * frac(s, a, sd) for s, a, sd in PLAN if s == "50m")
total50 = sum(hours["50m"] for s, _, _ in PLAN if s == "50m")
pct = done / total
finished = sum(frac(*r) == 1.0 for r in PLAN)
bar = "█" * int(40 * pct) + "░" * (40 - int(40 * pct))
print(f"\n  [{bar}] {100 * pct:5.1f}%   runs finished {finished}/{len(PLAN)}\n")
for s, a, sd in PLAN:
    if s == "50m":
        f = frac(s, a, sd)
        state = "done" if f == 1 else ("queued" if f == 0 else f"{100 * f * (1 + EVAL):4.0f}% trained")
        print(f"   50M {NAMES[a]:12s} seed {sd}   {state}")
start, s50 = when("setup"), when("stage 50m")
el = (now - start).total_seconds() / 3600
print(f"\n  running for {el:.1f} h, spent ~${el * RATE:.2f}")
if s50 and done50 > 0:
    rate = done50 / ((now - s50).total_seconds() / 3600)            # 50M work per hour since the stage started
    left = (total50 - done50) / rate
    eta = datetime.fromtimestamp(now.timestamp() + left * 3600, timezone.utc)
    print(f"  about {left:.1f} h left -> finishes ~{eta:%a %H:%M} UTC, total cost ~${(el + left) * RATE:.2f}")
print()
