"""Branch test of realblt_copy_latch_fresh.verdict_from and paired_verdict (O6b confirmatory rule, proposed): with the
smoke outputs' structure and values set by hand, every bar, stop, kill rule, claim label and paired class gives the
outcome the rule names, and every row beside the classes gets its sentence. Also the stop part of the job gate
(blocked: J1's stops and K5). It prints how many hand-set cases it evaluated.

    O6_SMOKE=1 ../segresearch/.venv-blt/bin/python -B copy_latch_verdict_test.py    (reads *_smoke.json; writes nothing)
"""
import copy
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
os.environ["O6_SMOKE"] = "1"
import realblt_copy_latch_fresh as F

J0 = {k: json.load(open(F.out_path(k))) for k in ("j1", "j2", "j3", "j4", "j5", "j6", "j7")}


def setp(pair, kd, v, lo, hi, what="exact"):
    if kd is None:
        pair[what] = [v, [lo, hi]]
    else:
        pair[kd][what] = [v, [lo, hi]]


def base():
    """Every bar passing, F2b and each F3 ID kind showing the hazard, the paired rows tied except F5's copy rows (L8
    better)."""
    J = copy.deepcopy(J0)
    j1 = J["j1"]
    j1["checks"]["passed"] = True; j1["R"]["R1"] = True; j1["R"]["R2"] = True; j1["stop"] = []
    j1["C4"].update(D_new_value=1, E10_new_value=0)
    P = J["j2"]["F1"]["pairs"]
    setp(P["D-all vs D"], "id_copy", 0.76, 0.70, 0.81)
    setp(P["D.L8 vs D"], "id_copy", 0.75, 0.70, 0.80)
    setp(P["E10-all vs E10"], "id_copy", 0.157, 0.11, 0.20)
    setp(P["E10-L8 vs E10"], "id_copy", 0.157, 0.11, 0.20)
    setp(P["E10.L8 vs E10"], "id_copy", 0.15, 0.11, 0.19)
    for b in ("D", "E10"):
        for kd in ("computed", "final", "copied"):
            if kd in P[f"{b}.L8 vs {b}"]:
                setp(P[f"{b}.L8 vs {b}"], kd, 0.0, -0.01, 0.01)
        P[f"{b}.L8 vs {b}"]["bpb_without_copy"] = [0.0005, [0.0, 0.001]]
        P[f"{b}.L8 vs {b}"]["rate"] = 0.0005
    setp(P["D.V8 vs D"], "computed", -0.05, -0.08, -0.03)
    P["D.V8 vs D"]["bpb_without_copy"] = [0.02, [0.017, 0.022]]
    P6 = J["j3"]["F6"]["pairs"]
    for b in ("D", "E10"):
        setp(P6[f"{b}.L8 vs {b}"], "computed", 0.0, -0.01, 0.01)
        P6[f"{b}.L8 vs {b}"]["rate"] = 0.0
    for cellres in [J["j2"]["F1"], J["j3"]["F6"], J["j4"]["F2a"], J["j4"]["F2b"], J["j6"]["F4"], J["j7"]["F5"]] + \
            list(J["j5"]["F3"]["by_kind"].values()):
        for key, p in cellres["pairs"].items():
            if "SIVPO" in key and "L8" in key:
                for kd in ("id_copy", "computed", "final", "copied"):
                    if kd in p:
                        setp(p, kd, 0.0, -0.01, 0.01)
                        setp(p, kd, 0.0, -0.1, 0.1, "bits_per_unit")
                p["bpb_without_copy"] = [0.0, [-0.0005, 0.0005]]
    p5 = J["j7"]["F5"]["pairs"]["D.L8 vs D.SIVPO"]
    setp(p5, "id_copy", 0.11, 0.06, 0.16)
    setp(p5, "id_copy", -2.6, -3.3, -1.9, "bits_per_unit")
    F2b = J["j4"]["F2b"]["pairs"]
    setp(F2b["D-all vs D"], "id_copy", 0.72, 0.62, 0.80)
    setp(F2b["D.L8 vs D"], "id_copy", 0.70, 0.60, 0.78); setp(F2b["E10.L8 vs E10"], "id_copy", 0.12, 0.05, 0.18)
    F2a = J["j4"]["F2a"]; F2a["fires_c1"]["L8"] = 0.0
    for b in ("D", "E10"):
        p = F2a["pairs"][f"{b}.L8 vs {b}"]
        setp(p, "id_copy", 0.0, -0.01, 0.01)
        for kd in ("computed", "final", "copied"):
            if kd in p:
                setp(p, kd, 0.0, -0.01, 0.01)
        p["bpb_without_copy"] = [0.0, [-0.001, 0.001]]
    for kind, bk in J["j5"]["F3"]["by_kind"].items():
        setp(bk["pairs"]["D-all vs D"], "id_copy", 0.7, 0.6, 0.8); setp(bk["pairs"]["D.L8 vs D"], "id_copy", 0.68, 0.58, 0.78)
    return J


N = 0                                               # the number of hand-set cases evaluated


def verdict(J):
    global N
    N += 1
    return F.verdict_from(J)


def show(name, V):
    g = V.get("generality_claim", {})
    print(f"{name}: fix {V.get('fix_claim')} | kills {V['kills']} | stops {V['stops']} | paired: windowed "
          f"{V['paired']['windowed']['class']}, released {V['paired']['released']['class']} | generality "
          f"{g.get('holds')}: {g.get('scope')}")
    return V


def stopped(V, label):
    """after a stop: the fix claim's scope carries the stop's label, and no other claim or paired verdict is reached"""
    return (V["fix_claim"]["holds"] is False and V["fix_claim"]["scope"].startswith(label)
            and V["generality_claim"]["scope"].startswith("not reached")
            and V["paired"]["windowed"]["class"] == V["paired"]["released"]["class"] == "not reached"
            and not any(k.startswith(("K6", "K7", "K9")) for k in V["kills"]) and "K8" not in V["reported"])


def W(J):
    return verdict(J)["paired"]


def beside(P, cell, rate, row, group="beside"):
    return next(r for r in P[group] if (r["cell"], r["rate"], r["row"]) == (cell, rate, row))


# ---------------------------------------------------------------- the fix and generality claims
V = show("all pass", verdict(base()))
assert V["fix_claim"] == {"holds": True, "scope": "both rates"} and not V["kills"]
Wd, R = V["paired"]["windowed"], V["paired"]["released"]
assert Wd["class"] == "tie" and Wd["reading"].startswith("On the cells and rates with the hazard, no copy row is classed "
                                                         "as differing: none differs by 2 points or more with a 95% "
                                                         "interval that excludes 0 (largest difference ")
assert Wd["reading"].endswith("A 100-text row can still sit a few points apart. This is not an equivalence test.")
assert {(r["cell"], r["rate"]) for r in Wd["class rows"]} == {("F1", "D"), ("F1", "E10"), ("F2b", "D"),
                                                             ("F3 digits6", "D"), ("F3 letters4", "D"), ("F3 mixed", "D")}
assert R["class"] == "L8 better" and R["reading"].startswith("Relative to SIV+PO") and "length cutoff" in R["reading"]
assert {x["class"] for x in R["L8 against D (beside)"].values()} <= {"L8 better", "D better", "tie"}
assert V["paired"]["note"] == "No outcome of this test shows that reading the model is needed."
G = V["generality_claim"]
assert G["holds"] is True and "F2b" in G["scope"] and sum(k.endswith("IDs (G3)") for k in G["parts"]) == 3
assert V["bars"]["G1"]["holds"] is True and "status" not in V["bars"]["G1"]
S = F.sentences(V, base())
assert S[0].startswith("On fresh GSM8K problems in O6's `\\nTicket NNNN` template, in BLT-1B with its 512-byte local "
                       "windows, scored teacher-forced, L8 restores verbatim copies at BLT-1B's default patch rate and "
                       "at entropy@10%") and "it gains +75.0 [+70.0, +80.0] points of exact copies, 0.99 of" in S[0]
assert S[1] == ("In BLT-1B with its 512-byte local windows, the fix also holds for the context reworded mid-line; and "
                "6-digit IDs, letter IDs and mixed IDs in the same `\\nTicket` template, at the default rate.")
assert S[-1] == F.NOT_NEEDED and any(x.startswith("Paired verdict, windowed: On the cells") for x in S)

J = base(); setp(J["j2"]["F1"]["pairs"]["E10-all vs E10"], "id_copy", 120 / 600, 0.17, 0.23)
setp(J["j2"]["F1"]["pairs"]["E10-L8 vs E10"], "id_copy", 108 / 600, 0.15, 0.21)
V = show("C1e (i) at an exact 90% share", verdict(J))
assert not 108 / 600 >= 0.9 * (120 / 600), "the float comparison without the tolerance fails this tie"
assert V["bars"]["C1e"]["i"] and V["fix_claim"]["holds"] and not V["kills"]

J = base(); setp(J["j2"]["F1"]["pairs"]["E10-all vs E10"], "id_copy", 0.02, -0.01, 0.05)
V = show("C1e untestable", verdict(J))
assert V["fix_claim"]["holds"] and V["fix_claim"]["scope"].startswith("default rate only")
assert V["generality_claim"]["holds"] and V["generality_claim"]["scope"].endswith("G2 (b) at D)")
assert ("F1", "E10") not in {(r["cell"], r["rate"]) for r in V["paired"]["windowed"]["class rows"]}
assert F.sentences(V, J)[0].endswith("At 10% the onset-removal bound could not be told from zero, so the fix was not "
                                     "tested at that rate.")

J = base(); setp(J["j2"]["F1"]["pairs"]["E10-all vs E10"], "id_copy", 0.02, -0.01, 0.05)
setp(J["j2"]["F1"]["pairs"]["E10.L8 vs E10"], "computed", -0.05, -0.08, -0.02)
V = show("untestable, E10 cost fails", verdict(J))
assert V["fix_claim"]["holds"], "an E10 bar should not block the default-rate claim (K7 reads D only)"

J = base(); setp(J["j2"]["F1"]["pairs"]["E10-all vs E10"], "id_copy", 0.02, -0.01, 0.05)
setp(J["j2"]["F1"]["pairs"]["E10.L8 vs E10"], "id_copy", 0.02, -0.01, 0.05)
V = show("untestable, +15 missed", verdict(J))
assert "bound untestable" in V["reported"]["K8"]

J = base(); setp(J["j2"]["F1"]["pairs"]["E10-L8 vs E10"], "id_copy", 0.12, 0.08, 0.16)
V = show("C1e (i) fails", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "failed: K7 (C1e)"} and any(k.startswith("K7") for k in V["kills"])
assert V["generality_claim"]["holds"] is False and V["generality_claim"]["failing"] == ["fix claim"]
S = F.sentences(V, J)
assert S[0] == ("In O6's `\\nTicket` template (F1), scored teacher-forced with 512-byte windows (10 texts), L8 failed "
                "on fresh problems: C1e.") and "No generality claim: the fix claim failed." in S

J = base(); setp(J["j2"]["F1"]["pairs"]["D.L8 vs D"], "id_copy", 0.75, -0.002, 0.80)
V = show("C1 fails on its lower bound only", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "failed: K7 (C1)"}
assert F.sentences(V, J)[0] == ("In O6's `\\nTicket` template (F1), scored teacher-forced with 512-byte windows (10 "
                                "texts), L8 failed on fresh problems: C1.")

J = base(); setp(J["j2"]["F1"]["pairs"]["E10.L8 vs E10"], "id_copy", 0.13, 0.09, 0.17)
V = show("+15 missed through the refill", verdict(J))
assert V["fix_claim"]["holds"] and "refill" in V["reported"]["K8"]

J = base()
for key, v in (("E10-all vs E10", 0.14), ("E10-L8 vs E10", 0.14), ("E10.L8 vs E10", 0.13)):
    setp(J["j2"]["F1"]["pairs"][key], "id_copy", v, v - 0.04, v + 0.04)
V = show("+15 missed, C1e holds", verdict(J))
assert V["fix_claim"]["holds"] and "bound-limited" in V["reported"]["K8"]

J = base()
for key, v in (("E10-L8 vs E10", 0.12), ("E10.L8 vs E10", 0.12)):
    setp(J["j2"]["F1"]["pairs"][key], "id_copy", v, v - 0.04, v + 0.04)
V = show("+15 missed, C1e (i) fails", verdict(J))
assert not V["fix_claim"]["holds"] and "misses onset starts" in V["reported"]["K8"]

J = base(); setp(J["j2"]["F1"]["pairs"]["D.V8 vs D"], "computed", -0.015, -0.04, 0.01)
J["j2"]["F1"]["pairs"]["D.V8 vs D"]["bpb_without_copy"] = [0.007, [0.005, 0.009]]
V = show("control fails (K6)", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "void: the control showed no cost (K6)"}
assert V["generality_claim"]["holds"] is False, "the generality claim needs the fix claim"
assert "carries no information" in V["paired"]["K6"]
S = F.sentences(V, J)
assert "The cost bars could not see V_8's known cost; no claim either way." in S
assert "No generality claim: the fix claim is void (K6)." in S
setp(J["j2"]["F1"]["pairs"]["D.L8 vs D"], "id_copy", 0.10, 0.05, 0.15)
V = show("K6 with K7", verdict(J))
assert V["fix_claim"]["scope"] == "failed: K7 (C1); and void: the control showed no cost (K6)"
S = F.sentences(V, J)
assert S.index("The cost bars could not see V_8's known cost; no claim either way.") == 1, "K7's reading comes first"
assert "No generality claim: the fix claim failed." in S

J = base(); J["j1"]["C4"]["D_new_value"] = 5
V = show("C4: 5 at D", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "failed: K7 (C4)"}
S = F.sentences(V, J)
assert S[0] == ("On the same problems without the ticket (F6), L8 failed: C4 (5 new-value removals by the patcher at "
                "the default rate; at most 4 allowed).") and not any(x.startswith("In O6's") for x in S)

J = base(); setp(J["j3"]["F6"]["pairs"]["D.L8 vs D"], "computed", -0.03, -0.05, -0.01)
V = show("C4s fails", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "failed: K7 (C4s)"}
assert F.sentences(V, J)[0] == "On the same problems without the ticket (F6), L8 failed: C4s."

J = base(); J["j3"]["F6"]["pairs"]["E10.L8 vs E10"]["rate"] = 0.006
V = show("the rate bar fails in F6 only", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "failed: K7 (rate)"}
S = F.sentences(V, J)
assert S[0] == "On the same problems without the ticket (F6), L8 failed: rate (E10)." and not any(
    x.startswith("In O6's") for x in S)
J = base(); J["j1"]["C4"]["D_new_value"] = 4
assert verdict(J)["fix_claim"]["holds"], "4 at D must pass"

J = base(); J["j1"]["C4"]["E10_new_value"] = 1
V = show("K2", verdict(J))
assert stopped(V, "failed: K2") and any(k.startswith("K2") for k in V["kills"])
assert F.sentences(V, J)[0] == "L8 removed a new value at 10%; L8 is dropped."

J = base(); J["j2"]["F1"]["pairs"]["D.L8 vs D"]["rate"] = 0.006
V = show("rate bar fails", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "failed: K7 (rate)"}
assert F.sentences(V, J)[0] == ("In O6's `\\nTicket` template (F1), scored teacher-forced with 512-byte windows (10 "
                                "texts), L8 failed on fresh problems: rate (D).")

J = base(); J["j1"]["R"]["R1"] = False; J["j1"]["R"]["L8_fires_c1"] = 0.9   # as recall_footprint would write them
J["j1"]["R_rows"][0]["L8_c1"] = False
V = show("R1 fails (K10)", verdict(J))
assert stopped(V, "failed: K10") and F.sentences(V, J)[0] == "L8 failed its recall or footprint check before scoring."

J = {"j1": {"checks": {"passed": False}, "stop": ["implementation check failed"]}}
V = show("J1 check fails (J1 stops before R and C4)", verdict(J))
assert V["stops"] == ["implementation check failed (J1)"] and stopped(V, "not reached: the implementation checks failed")
assert F.sentences(V, J)[0] == "not reached: the implementation check failed; a redo needs the PI's approval"
J = {"j1": {"checks": {"passed": False}, "stop": ["implementation check failed again (second failure)"]}}
V = show("J1 check fails a second time", verdict(J))
assert stopped(V, "not reached: the implementation checks failed")
assert F.sentences(V, J)[0] == "The implementation failed; nothing was tested."

J = base(); setp(J["j2"]["F1"]["pairs"]["D-all vs D"], "id_copy", 0.15, 0.08, 0.22)
setp(J["j2"]["F1"]["pairs"]["D.L8 vs D"], "id_copy", 0.10, 0.05, 0.15)       # C1 fails too: K7 is not read under K5
V = show("C0 fails (K5)", verdict(J))
assert "K5" in V["stops"] and stopped(V, "not reached: K5")
assert F.sentences(V, J)[0] == ("In O6's `\\nTicket` template, scored teacher-forced with 512-byte windows (10 texts), "
                                "the hazard did not replicate on fresh problems; L8's copy fix was not scored.")

j1ok, j1stop = {"stop": []}, {"stop": ["K10"]}
j2ok, j2k5 = base()["j2"], J["j2"]
assert F.blocked("j2", j1ok) is None and F.blocked("j3", j1ok, j2ok) is None and F.blocked("j7", j1ok, j2ok) is None
assert F.blocked("j2", None).startswith("J1 has not run") and F.blocked("j2", j1stop).startswith("J1 stopped")
assert F.blocked("j3", j1ok, None).startswith("J2 has not run")
assert all(F.blocked(j, j1ok, j2k5).startswith("K5") for j in ("j3", "j4", "j5", "j6", "j7"))
assert F.blocked("j2", j1ok, j2k5) is None
print("job gate (stops): J1's stops block J2-J7, K5 blocks J3-J7, and J3-J7 need J2")

V = show("J1 missing", verdict({}))
assert V["fix_claim"]["scope"] == "not reached: J1 has not run" and V["paired"]["windowed"]["class"] == "not reached"

J = base(); del J["j3"]
V = show("J3 missing", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "not reached: J3 has not run"} and not V["kills"]
assert V["generality_claim"]["scope"] == "not reached: J3 has not run"
setp(J["j2"]["F1"]["pairs"]["D.L8 vs D"], "id_copy", 0.10, 0.05, 0.15)
V = show("J3 missing, C1 failing", verdict(J))
assert V["fix_claim"] == {"holds": False, "scope": "failed: K7 (C1)"}, "a failed bar governs a missing J3"

J = base(); J["j7"] = {"status": "x", "window": 0, "checks": {"passed": False}, "stop": ["implementation check failed (J7)"]}
V = show("J7 check fails", verdict(J))
assert V["fix_claim"]["holds"] and V["paired"]["released"]["class"] == "not reached"
assert V["paired"]["windowed"]["class"] == "tie" and V["reported"]["F5"].startswith("not scored")

J = base(); del J["j5"]
V = show("J5 missing", verdict(J))
assert V["generality_claim"]["scope"].startswith("not reached") and V["paired"]["windowed"]["class"] == "not reached"
assert V["paired"]["released"]["class"] == "L8 better", "the released verdict does not need J5"
assert {(r["cell"], r["rate"]) for r in V["paired"]["windowed"]["class rows so far"]} == {("F1", "D"), ("F1", "E10"),
                                                                                          ("F2b", "D")}

J = base(); setp(J["j4"]["F2b"]["pairs"]["D-all vs D"], "id_copy", 0.0, 0.0, 0.0)
setp(J["j4"]["F2b"]["pairs"]["D.L8 vs D"], "id_copy", 0.0, 0.0, 0.0)
V = show("no hazard in F2b (the smoke's case)", verdict(J))
assert V["bars"]["G1"]["holds"] is None and V["bars"]["G1"]["status"] == "not testable: no hazard in F2b (D-all - D below +20)"
G1e = V["reported"]["G1e (reported only)"]
assert G1e["lower bound > 0"] is None and G1e["status"].startswith("not testable")
assert not any(k.startswith("K9") for k in V["kills"])
assert V["generality_claim"]["holds"] and "F2b" not in V["generality_claim"]["scope"]

J = base(); setp(J["j4"]["F2b"]["pairs"]["D.L8 vs D"], "id_copy", 0.10, 0.05, 0.15)
V = show("G1 fails", verdict(J))
assert [k for k in V["kills"] if k.startswith("K9")] == [
    "K9: G1 fails: the fix does not extend to the context reworded mid-line (F2b)"]
assert V["generality_claim"]["holds"] is False and V["fix_claim"]["holds"], "K9 does not touch the fix claim"
assert ("Scored teacher-forced with 512-byte windows (10 texts), at the default rate, the fix does not "
        "extend to the context reworded mid-line (F2b's template).") in F.sentences(V, J)

J = base(); setp(J["j5"]["F3"]["by_kind"]["letters4"]["pairs"]["D.L8 vs D"], "id_copy", 0.40, 0.30, 0.50)
V = show("G3 fails for letters", verdict(J))
assert [k for k in V["kills"] if k.startswith("K9")] == ["K9: G3 fails: the fix does not extend to letters4 IDs"]
assert V["generality_claim"]["holds"] is False and V["generality_claim"]["failing"] == ["letters4 IDs (G3)"]
assert ("In O6's `\\nTicket` template, scored teacher-forced with 512-byte windows (3 texts), at the "
        "default rate, the fix does not extend to letter IDs.") in F.sentences(V, J)
setp(J["j2"]["F1"]["pairs"]["D.L8 vs D"], "id_copy", 0.10, 0.05, 0.15)
V = show("G3 fails for letters, and C1 fails", verdict(J))
assert [k for k in V["kills"] if k.startswith("K9")] == ["K9: G3 fails: the fix does not extend to letters4 IDs"]
S = F.sentences(V, J)
assert "No generality claim: the fix claim failed." in S and not any("does not extend" in x for x in S), \
    "K9 is still read, but its sentence prints only while the fix claim holds"

J = base(); setp(J["j4"]["F2b"]["pairs"]["D-all vs D"], "id_copy", 0.0, 0.0, 0.0)
for bk in J["j5"]["F3"]["by_kind"].values():
    setp(bk["pairs"]["D-all vs D"], "id_copy", 0.1, 0.02, 0.18)
V = show("no hazard anywhere", verdict(J))
assert V["generality_claim"]["holds"] is None and V["generality_claim"]["scope"].startswith("not testable")
assert all(v["holds"] is None and v["status"].startswith("not testable: no hazard in") for v in V["bars"]["G3"].values())

J = base(); setp(J["j4"]["F2a"]["pairs"]["E10.L8 vs E10"], "id_copy", -0.04, -0.07, -0.01)
V = show("G2 (b) fails at E10", verdict(J))
assert V["generality_claim"]["holds"] is False and V["generality_claim"]["failing"] == ["G2 (b) at E10"]
assert "No generality claim: G2 (b) at E10." in F.sentences(V, J)

J = base(); bk = J["j5"]["F3"]["by_kind"]["digits6"]["pairs"]
setp(bk["D.L8 vs D"], "id_copy", 70 / 136, 0.4, 0.6); setp(bk["D-all vs D"], "id_copy", 100 / 136, 0.6, 0.85)
V = show("G3 share at exactly 0.7 (70 of 100 copies, n = 136)", verdict(J))
assert not 70 / 136 >= 0.7 * (100 / 136), "the float comparison without the tolerance fails this tie"
assert V["bars"]["G3"]["digits6"]["holds"], "0.7 of D-all - D, exactly, passes"

# ---------------------------------------------------------------- the paired verdicts: only copy rows with the hazard
P1 = lambda J: J["j2"]["F1"]["pairs"]
J = base(); setp(P1(J)["D.L8 vs D.SIVPO"], "id_copy", -0.05, -0.08, -0.02)
P = W(J)["windowed"]
assert P["class"] == "SIV+PO better" and P["reading"].startswith("L8 lost copies the byte rule kept, in F1 at D: in ")
assert "lost copies L8 did not fire at c1 (copies L8 missed); in " in P["reading"]
assert P["reading"].endswith("it fired, and the loss is a side effect of the refit.")
assert "discordant texts" in next(r for r in P["class rows"] if (r["cell"], r["rate"]) == ("F1", "D"))
J = base(); setp(P1(J)["E10.L8 vs E10.SIVPO"], "id_copy", 0.05, 0.02, 0.08)
P = W(J)["windowed"]
assert P["class"] == "L8 better" and "not expected by construction" in P["reading"] and "F1 at E10" in P["reading"]
J = base(); setp(P1(J)["D.L8 vs D.SIVPO"], "id_copy", -0.05, -0.08, -0.02)
setp(J["j5"]["F3"]["by_kind"]["mixed"]["pairs"]["D.L8 vs D.SIVPO"], "id_copy", 0.06, 0.02, 0.1)
assert W(J)["windowed"]["class"] == "mixed"
J = base(); setp(J["j5"]["F3"]["by_kind"]["digits6"]["pairs"]["E10-all vs E10"], "id_copy", 0.1, 0.03, 0.17)
setp(J["j5"]["F3"]["by_kind"]["digits6"]["pairs"]["E10.L8 vs E10.SIVPO"], "id_copy", -0.06, -0.1, -0.02)
assert W(J)["windowed"]["reading"].startswith("L8 lost copies the byte rule kept, in F3 6-digit IDs at E10: "), \
    "E10 rows count where E10 shows the hazard"
J = base(); setp(J["j5"]["F3"]["by_kind"]["digits6"]["pairs"]["E10.L8 vs E10.SIVPO"], "id_copy", -0.06, -0.1, -0.02)
P = W(J)
assert P["windowed"]["class"] == "tie" and beside(P, "F3 digits6", "E10", "id_copy exact")["sentence"] == \
    "No hazard here (E10-all - E10 has no lower bound above 0), so a difference here is not a fixed or missed copy; it " \
    "is reported only."
J = base(); setp(J["j5"]["F3"]["by_kind"]["digits6"]["pairs"]["D.L8 vs D.SIVPO"], "id_copy", -0.03, -0.07, 0.0)
P = W(J)["windowed"]
assert P["class"] == "tie" and P["reading"] == ("On the cells and rates with the hazard, no copy row is classed as "
                                                "differing: none differs by 2 points or more with a 95% interval that "
                                                "excludes 0 (largest difference -3.0 points, F3 6-digit IDs at D, "
                                                "[-7.0, +0.0]). A 100-text row can still sit a few points apart. This "
                                                "is not an equivalence test."), P["reading"]

# each of these leaves the class the gated copy rows set (tie here, SIV+PO better in the last)
J = base(); F3 = J["j5"]["F3"]["by_kind"]
F3["dose"]["pairs"]["D.L8 vs D.SIVPO"]["bpb_without_copy"] = [0.006, [0.002, 0.009]]
P = W(J)
assert P["windowed"]["class"] == "tie" and beside(P, "F3 dose", "D", "C2 bits per byte without the copy",
                                                  "doses (apart)")["sentence"] == F.BESIDE["dose other"]
J = base(); setp(J["j5"]["F3"]["by_kind"]["dose"]["pairs"]["D.L8 vs D.SIVPO"], "id_copy", 1.0, 0.6, 1.4, "bits_per_unit")
P = W(J)
assert P["windowed"]["class"] == "tie" and beside(P, "F3 dose", "D", "id_copy bits per copy", "doses (apart)")[
    "sentence"] == "L8's copies cost more bits than the byte rule's on F3 doses, D, id_copy bits per copy."
J = base(); setp(J["j4"]["F2a"]["pairs"]["D.L8 vs D.SIVPO"], "id_copy", -0.10, -0.3, -0.01)
P = W(J); r = beside(P, "F2a", "D", "id_copy exact")
assert P["windowed"]["class"] == "tie" and r["sentence"] == F.BESIDE["F2a"] and "discordant texts" in r
J = base(); setp(J["j4"]["F2a"]["pairs"]["D.L8 vs D.SIVPO"], "computed", -0.05, -0.08, -0.03)
assert beside(W(J), "F2a", "D", "C3 computed")["sentence"] == "L8 cost more than the byte rule on F2a, D, C3 computed."
J = base(); setp(P1(J)["D.L8 vs D.SIVPO"], "id_copy", 1.0, 0.6, 1.4, "bits_per_unit")
P = W(J)
assert P["windowed"]["class"] == "tie" and beside(P, "F1", "D", "id_copy bits per copy")["sentence"] == \
    "L8's copies cost more bits than the byte rule's on F1, D, id_copy bits per copy."
J = base(); setp(J["j4"]["F2b"]["pairs"]["D-all vs D"], "id_copy", 0.0, 0.0, 0.0)
setp(J["j4"]["F2b"]["pairs"]["D.L8 vs D.SIVPO"], "id_copy", -0.05, -0.08, -0.02)
P = W(J)
assert P["windowed"]["class"] == "tie" and beside(P, "F2b", "D", "id_copy exact")["sentence"] == \
    F.BESIDE["no hazard"].format(gate=F.GATE_TEXT["D"])
J = base(); setp(P1(J)["D.L8 vs D.SIVPO"], "id_copy", -0.05, -0.08, -0.02)
J["j3"]["F6"]["pairs"]["D.L8 vs D.SIVPO"]["bpb_without_copy"] = [-0.003, [-0.004, -0.002]]
P = W(J)
assert P["windowed"]["class"] == "SIV+PO better", "an L8-better cost row does not make a copy loss 'mixed'"
assert beside(P, "F6", "D", "C2 bits per byte without the copy")["sentence"] == \
    "L8 cost less than the byte rule on F6, D, C2 bits per byte without the copy."

J = base(); setp(J["j2"]["F1"]["pairs"]["D-all vs D"], "id_copy", 0.15, 0.08, 0.22)        # no windowed row passes its gate
for cell in ("F2b",):
    setp(J["j4"][cell]["pairs"]["D-all vs D"], "id_copy", 0.0, 0.0, 0.0)
setp(J["j2"]["F1"]["pairs"]["E10-all vs E10"], "id_copy", 0.02, -0.01, 0.05)
for bk in J["j5"]["F3"]["by_kind"].values():
    setp(bk["pairs"]["D-all vs D"], "id_copy", 0.1, 0.02, 0.18)
N += 1
P = F.paired_verdict(J)                              # read directly: through verdict_from, F1's D gate failing is K5
assert P["windowed"]["class"] == "not testable" and P["windowed"]["reading"] == F.READINGS["windowed"]["not testable"]

J = base(); setp(J["j5"]["F3"]["by_kind"]["dose"]["pairs"]["D.L8 vs D.SIVPO"], "id_copy", -0.12, -0.2, -0.05)
P = W(J)
assert P["windowed"]["class"] == "tie" and beside(P, "F3 dose", "D", "id_copy exact", "doses (apart)")["sentence"].startswith(
    "L8 lost dose copies the byte rule kept: in ")
setp(J["j5"]["F3"]["by_kind"]["dose"]["pairs"]["D-all vs D"], "id_copy", 0.0, 0.0, 0.0)
assert beside(W(J), "F3 dose", "D", "id_copy exact", "doses (apart)")["sentence"] == \
    F.BESIDE["no hazard"].format(gate=F.GATE_TEXT["D"])
J = base(); setp(J["j6"]["F4"]["pairs"]["D.L8 vs D.SIVPO"], "id_copy", -0.4, -0.5, -0.3)
P = W(J)
assert P["windowed"]["class"] == "tie" and P["released"]["class"] == "L8 better", "F4 is apart"
assert beside(P, "F4", "D", "id_copy exact", "F4 (apart)")["class"] == "SIV+PO better"

p5 = lambda J: J["j7"]["F5"]["pairs"]["D.L8 vs D.SIVPO"]
J = base(); setp(p5(J), "id_copy", 0.0, -0.01, 0.01)
R = W(J)["released"]
assert R["class"] == "tie" and R["reading"] == ("Under the released conversion the copy row is not classed as differing "
                                                "(+0.0 points, [-1.0, +1.0]): a difference of 2 points or more would "
                                                "need a 95% interval that excludes 0. This is not an equivalence test.")
setp(p5(J), "id_copy", -0.05, -0.08, -0.02)
assert W(J)["released"]["reading"].startswith("Under the released conversion, L8 lost copies the byte rule kept: in ")
J = base(); setp(p5(J), "computed", -0.05, -0.08, -0.03)
R = W(J)["released"]
assert R["class"] == "L8 better" and next(r for r in R["beside"] if r["row"] == "C3 computed")["sentence"] == \
    "L8 cost more than the byte rule on F5, D, C3 computed."
print("paired verdicts: windowed tie (with its largest difference), SIV+PO better (with the discordant split), L8 "
      "better, mixed, not testable; an E10 row with and without the hazard; a dose cost row, a dose bits row, an F2a "
      "flip, an F2a cost row, an F1 bits row, an F2b copy row without the hazard and an L8-better cost row each leave "
      "the class; the dose copy row with and without the hazard; F4 apart; released L8 better, tie, SIV+PO better, and "
      "a cost row beside it")
print(f"verdict branches: all as the rule says ({N} hand-set cases evaluated)")
