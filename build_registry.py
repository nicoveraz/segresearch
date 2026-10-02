"""Backfill the results registry from the raw logs in results/logs/ (rewrites results/registry/backfill.jsonl).

Each RESULT line becomes one record (registry.parse) plus metadata the old lines did not print: the experiment,
the corpus (math | reason), the local window, and for BLT-1B the threshold mode (problem | train, issue #26).
Also backfilled: the agent loop's results.tsv snapshot, the original loop's run/learned-chunking logs and the
held-out test tables.

    uv run build_registry.py
"""
import csv
import glob
import json
import os
import re

import registry

LOGS = os.path.join(registry.ROOT, "results", "logs")
OUT = os.path.join(registry.DIR, "backfill.jsonl")

MLX = {                                      # folder -> experiment (+ metadata the RESULT lines lack)
    "mlx_math_small": dict(experiment="math_small"),
    "mlx_math_new_strategies": dict(experiment="math_new_strategies"),
    "mlx_math_5seed_confirm": dict(experiment="math_new_strategies"),
    "mlx_math_larger_model": dict(experiment="math_larger"),
    "mlx_scratchpad8": dict(experiment="scratchpad8", scratch=True),
    "mlx_scratchpad16": dict(experiment="scratchpad16", scratch=True),
    "mlx_realtext_code": dict(experiment="realtext"),
    "mlx_realtext_more": dict(experiment="realtext"),
    "mlx_realtext_wordstart_fix": dict(experiment="realtext_wordstart_fix"),
    "mlx_math_tight10": dict(experiment="math_tight_budget"),
    "mlx_math_dep_trigger": dict(experiment="math_tight_budget"),
    "mlx_math_budget_sweep": dict(experiment="math_tight_budget"),
    "mlx_reason_trained": dict(experiment="reason_trained", corpus="reason", window=32),
    "mlx_reason_trained_window8": dict(experiment="reason_trained", corpus="reason", window=8),
}
BLT = {                                      # BLT-1B log file -> experiment, threshold mode
    "realblt.log": dict(experiment="blt_answer_boundary"),
    "realblt_budget.log": dict(experiment="blt_budget", thresh="problem", superseded=True),   # rerun in budget2
    "realblt_budget2.log": dict(experiment="blt_budget", thresh="problem"),
    "realblt26_budget_problem.log": dict(experiment="blt_budget", thresh="problem", rerun=26),
    "realblt26_budget_train.log": dict(experiment="blt_budget", thresh="train", rerun=26),
    "realblt_code.log": dict(experiment="blt_code", thresh="problem"),
    "realblt26_code_problem.log": dict(experiment="blt_code", thresh="problem", rerun=26),
    "realblt26_code_train.log": dict(experiment="blt_code", thresh="train", rerun=26),
    "realblt_reason_trace.log": dict(experiment="blt_trace", thresh="problem"),
    "realblt26_trace_problem.log": dict(experiment="blt_trace", thresh="problem", rerun=26),
    "realblt26_trace_train.log": dict(experiment="blt_trace", thresh="train", rerun=26),
    "realblt_reason_logic.log": dict(experiment="blt_logic", thresh="problem"),
    "realblt_reason_direct.log": dict(experiment="blt_direct", thresh="problem"),
    "realblt_reason_smoke.log": None,
}


def rec(script, source, line=None, **fields):
    r = {"script": script, "origin": "backfill", "commit": None, "source": os.path.relpath(source, registry.ROOT)}
    if line is not None:
        r.update(registry.parse(line)); r["raw"] = line.strip()
    r.update(fields)
    return r


def main():
    rows = []
    for folder, meta in MLX.items():
        for f in sorted(glob.glob(os.path.join(LOGS, folder, "*.log"))):
            for line in open(f):
                if line.startswith("RESULT"):
                    r = rec("mathexp", f, line, **meta)
                    if r["format"] == "realtext":
                        r["script"] = "realtext"
                    r.setdefault("corpus", "math" if r["format"] == "mathexp" else r.get("data"))
                    r.setdefault("window", 32)
                    rows.append(r)
    for folder in ("blt1b", "blt1b_reason_truefalse_superseded"):
        for f in sorted(glob.glob(os.path.join(LOGS, folder, "*.log"))):
            meta = BLT.get(os.path.basename(f), {}) if folder == "blt1b" else \
                dict(experiment="blt_" + os.path.basename(f)[len("realblt_reason_"):-4], thresh="problem", superseded=True,
                     note="True/False answers (unusable format)")
            if meta is None:
                continue
            script = re.sub(r"26_(budget|code|trace)_\w+", lambda m: {"budget": "_budget", "code": "_code", "trace": "_reason"}[m.group(1)],
                            os.path.basename(f)[:-4]).replace("_budget2", "_budget").replace("_reason_trace", "_reason") \
                .replace("_reason_logic", "_reason").replace("_reason_direct", "_reason")
            for line in open(f):
                if line.startswith("RESULT"):
                    rows.append(rec(script, f, line, **meta))
    # the original agent loop (synthetic corpus): results.tsv, run/learned logs, held-out test tables
    tsv = os.path.join(registry.ROOT, "results", "results_tsv_snapshot.tsv")
    for row in csv.DictReader(open(tsv), delimiter="\t"):
        rows.append(rec("run", tsv, experiment="loop", format="loop", commit=row["commit"], status=row["status"],
                        description=row["description"], **{k: float(row[k]) for k in ("val_ans_bits", "val_ans_long_bits", "boundary_rate")}))
    for f in sorted(glob.glob(os.path.join(LOGS, "misc", "*.log"))):
        name = os.path.basename(f)[:-4]
        if name.startswith(("run", "learned")):
            kv = dict(re.findall(r"^(\w+):\s+(\S+)$", open(f).read(), re.M))
            if kv:
                rows.append(rec("run" if name.startswith("run") else "learned_chunking", f, experiment="loop" if name.startswith("run") else name,
                                format="kv", **{k: float(v) for k, v in kv.items()}))
        elif name.startswith("test_final"):
            for line in open(f):
                m = re.match(r"(.+?)\s{2,}" + r"\s+".join([registry.NUM] * 9) + r"\s*$", line)
                if m:
                    v = [float(x) for x in m.groups()[1:]]
                    for i, fmt in enumerate("ABC"):
                        rows.append(rec("test_final", f, experiment=name, format="test_final", rule=m.group(1).strip(), split=fmt,
                                        ans_bits=v[3 * i], long_bits=v[3 * i + 1], rate=v[3 * i + 2]))
    os.makedirs(registry.DIR, exist_ok=True)
    with open(OUT, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by[r.get("experiment")] = by.get(r.get("experiment"), 0) + 1
    print(f"{len(rows)} records -> {os.path.relpath(OUT, registry.ROOT)}")
    for k, n in sorted(by.items()):
        print(f"  {k:28s} {n}")


if __name__ == "__main__":
    main()
