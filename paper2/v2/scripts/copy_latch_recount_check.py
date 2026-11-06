"""Dev recount (realblt_copy_latch_fresh devcheck, part A) under the amended whole-number test, where a trailing comma
is not part of a value (2026-10-30, after the second check), compared with the stored results/blt_copy_latch_devcheck.json.
Stored dev patcher outputs only; no model; writes nothing.

    ../segresearch/.venv-blt/bin/python -B copy_latch_recount_check.py | tee results/blt_copy_latch_fresh_recount_check.log
"""
import json
import os
import sys

sys.dont_write_bytecode = True
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
os.environ["O6_SMOKE"] = "1"
import realblt_copy_latch_fresh as F

C = F.L.load_cache(os.path.join(F.RES, "blt_copy_latch_patcher.npz"))
for rows in C.values():
    F.add_masks(rows)
eq_fit = F.blt_layouts.Fit((0.10,))
for r in C["eq_train"]:
    eq_fit.add(F.L.sig_of(r))
eq_fit.done()
lay = lambda r: {"D": set(r["D"].tolist()),
                 "E10": set(F.blt_layouts.layout("entropy", 0.10, F.L.sig_of(r), eq_fit).tolist())}
rules = ("L8", "L8c", "V8", "SIV", "PO", "SIVPO", "L8b")
rc = {g: F.value_counts(C[g], lay, rules, keep_cases=("L8", "L8c", "SIVPO", "V8")) for g in ("eq_test", "eq_train")}
old = json.load(open(os.path.join(F.RES, "blt_copy_latch_devcheck.json")))["A_recount"]
diff = []                                                           # (group, rule-layout, count, stored, now)
for g in rc:
    assert rc[g]["computed_value_positions"] == old[g]["computed_value_positions"]
    for key, v in rc[g].items():
        if key == "computed_value_positions":
            continue
        for f in ("in_layout", "removed", "new_value", "new_value_prefix_test"):
            if v[f] != old[g][key][f]:
                diff.append((g, key, f, old[g][key][f], v[f]))
print("keys compared:", sum(len(v) - 1 for v in rc.values()), "x 4 fields")
print("changed counts:", diff if diff else "none")
assert not diff
for g in rc:
    for key in ("D-L8", "E10-L8", "D-V8", "E10-V8"):
        print(g, key, {f: rc[g][key][f] for f in ("in_layout", "removed", "new_value", "new_value_prefix_test")})
