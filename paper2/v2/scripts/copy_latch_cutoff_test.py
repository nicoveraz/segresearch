"""Branch test of realblt_copy_latch_cutoff.verdict_from (O6c, proposed): with hand-set score outputs, each class of the
primary row (L8 against SIV+PO<620 on exact copies), the underpowered case, a failed check and a missing score job give
the outcome and the reading the rule names. It prints how many hand-set cases it evaluated.

    O6C_SMOKE=1 ../segresearch/.venv-blt/bin/python -B copy_latch_cutoff_test.py    (reads nothing; writes nothing)
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
os.environ["O6C_SMOKE"] = "1"
import realblt_copy_latch_cutoff as C

N = 0


def S(prim, n=600, vs_sivpo=(0.067, [0.04, 0.095]), lost=13, missed=5):
    """A score output with the primary row `prim` = (point, [lo, hi]) on n texts, and the rows beside it"""
    pair = lambda v: {"id_copy": {"exact": [v[0], list(v[1])]}}
    return {"n": n, "stop": [],
            "pairs": {C.PRIMARY: pair(prim), "D.L8 vs D.SIVPO": pair(vs_sivpo), "D-all vs D": pair((0.5, [0.45, 0.55])),
                      "D.SIVPO620 vs D": pair((0.48, [0.43, 0.53]))},
            "discordance": {C.PRIMARY: {"lost": {"copies": lost, "L8 did not fire at c1": missed}, "kept": {}},
                            "D.L8 vs D.SIVPO": {"lost": {"copies": 2}, "kept": {"copies": 9}}},
            "by_length": {}}


def verdict(s):
    global N
    N += 1
    V = C.verdict_from(s)
    print(f"{V['class']:18s} {V['reading'][:90]}")
    return V


V = verdict(S((0.018, [0.005, 0.03])))
assert V["class"] == "tie" and V["reading"] == (
    "Under the released conversion, a byte rule that stops at token 620, a position fixed from dev texts, is not "
    "classed as differing from L8 (a class needs a difference of at least 2 points with its 95% interval on that side "
    "of 0; this is not an equivalence test). L8's model-reading gate shows no advantage over a fixed position cutoff "
    "on this template. With O6b's windowed result, no setting tested shows that reading the model is needed."), \
    "below the margin: a tie, whatever the interval"
assert V["sentences"][1].startswith("Primary: D.L8 - D.SIVPO<620 on exact copies, +1.80 [+0.50, +3.00] points on 600 "
                                    "texts")
assert verdict(S((0.02, [-0.001, 0.04])))["class"] == "tie", "at the margin, with 0 inside the interval: a tie"
V = verdict(S((0.02, [0.008, 0.033])))
assert V["class"] == "L8 better" and V["reading"] == (
    "L8's gate helps beyond a byte rule that stops at token 620, a position fixed from dev texts, under the released "
    "conversion, on this template; reading the model earns something there."), "exactly at the margin, interval above 0"
V = verdict(S((-0.03, [-0.05, -0.01])))
assert V["class"] == "the cutoff better" and V["reading"] == (
    "A byte rule that stops at token 620, a position fixed from dev texts, beats L8 under the released conversion, on "
    "this template.")
assert verdict(S((-0.019, [-0.035, -0.004])))["class"] == "tie"
V = verdict(S((0.05, [0.02, 0.08]), n=399))
assert V["class"] == "no class" and V["reading"] == "Fewer than 400 test texts (399): the numbers are reported with no class."
assert V["beside"]["D.L8 - D.SIVPO (O6b's released row, replicated)"]["class"] == "no class", \
    "fewer than 400 test texts: no row has a class, D.L8 - D.SIVPO beside it included"
assert verdict(S((0.05, [0.02, 0.08]), n=400))["class"] == "L8 better", "400 test texts are enough for a class"
V = verdict(S((0.018, [0.005, 0.03]), lost=16, missed=13))
assert V["beside"]["discordant split (primary)"] == {"n": 16, "a": 13, "b": 3}
assert "of the 16 copies L8 lost to the cutoff, L8 did not fire at c1 on 13 and fired on 3." in V["sentences"][2]
assert V["beside"]["D.L8 - D.SIVPO (O6b's released row, replicated)"]["class"] == "L8 better"
assert V["beside"]["D.SIVPO620 - D"] == (0.48, [0.43, 0.53]), "D.SIVPO<620 - D is reported under its own name"
assert "D.SIVPO620 - D.SIVPO" not in V["beside"]
assert V["beside"]["discordant texts (D.L8 vs D.SIVPO)"] == {"lost": {"copies": 2}, "kept": {"copies": 9}}
assert ("Beside it: D.L8 - D.SIVPO +6.70 [+4.00, +9.50] points (class under O6b's rule: L8 better); D-all - D +50.00 "
        "[+45.00, +55.00] points; D.SIVPO<620 - D +48.00 [+43.00, +53.00] points;") in V["sentences"][2]
assert verdict(S((-0.02, [-0.033, -0.008])))["class"] == "the cutoff better", "exactly at -2 points, interval below 0"
assert verdict(S((-0.02, [-0.04, 0.001])))["class"] == "tie", "at -2 points, 0 inside the interval"
V = verdict(S((0.0216667, [0.0, 0.045])))
assert V["class"] == "tie", "an interval end of exactly 0 is a tie"
assert "+2.17 [0.00, +4.50] points on 600 texts" in V["sentences"][1], "two decimals; an end of exactly 0 prints 0.00"
assert verdict(S((-0.0216667, [-0.045, 0.0])))["class"] == "tie", "an interval end of exactly 0 is a tie"
V = verdict({"stop": ["implementation check failed (window 0)"]})
assert V["class"] == "not reached" and V["reading"] == "The implementation failed under window 0; nothing was tested."
V = verdict(None)
assert V["class"] == "not reached" and V["reading"] == "not reached: the score job has not run"
print(f"O6c verdict branches: all as the rule says ({N} hand-set cases evaluated)")
