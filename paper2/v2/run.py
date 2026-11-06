"""Run a script of scripts/ unchanged, from anywhere:  python paper2/v2/run.py SCRIPT [ARGS...]

The scripts ran from the root of the author's private working repository (four from its paper2v2/ folder). Most find
results/ and data/ through _seg.HERE (scripts/_seg.py points it at this folder); the ones in OWN_PATH find them from
their own location instead. For those the runner sets __file__ to where the script sat relative to results/ (this
folder, or a paper2v2/ subfolder for the four), and points xdomain_analysis.HERE here for its two importers. Every
other script runs with its real __file__ (the O6b and O6c code hashes itself). No bytecode is written.
"""
import os
import sys

sys.dont_write_bytecode = True
V2 = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(V2, "scripts")
SEG = os.path.dirname(os.path.dirname(V2))                 # the segresearch root (public modules)
OWN_PATH = {"r9_cue_split", "budget_by_kind", "quadrant_analysis", "plot_profiles", "xdomain_analysis", "xdomain_data",
            "reducible2", "copy_latch_paired_odds", "copy_latch_cutoff_predict", "copy_latch_recount_check"}
IN_PAPER2V2 = {"intervals", "fig_tradeoff", "fig_detector_budget", "prose_rewrite"}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.split("\n\n")[0] + "\nscripts: " + " ".join(sorted(
            f[:-3] for f in os.listdir(SCRIPTS) if f.endswith(".py") and not f.startswith("_"))))
    name = os.path.basename(sys.argv[1]).removesuffix(".py")
    path = os.path.join(SCRIPTS, name + ".py")
    if not os.path.exists(path):
        sys.exit(f"no script {name}.py in {SCRIPTS}")
    if name in OWN_PATH:
        file = os.path.join(V2, name + ".py")
    elif name in IN_PAPER2V2:
        file = os.path.join(V2, "paper2v2", name + ".py")
    else:
        file = path
    sys.path[0:0] = [SCRIPTS, SEG]
    if name in ("xdomain_probe", "xdomain_newtok"):
        import xdomain_analysis
        xdomain_analysis.HERE = V2
    sys.argv = [file] + sys.argv[2:]
    code = compile(open(path).read(), path, "exec")
    exec(code, {"__name__": "__main__", "__file__": file, "__builtins__": __builtins__})


if __name__ == "__main__":
    main()
