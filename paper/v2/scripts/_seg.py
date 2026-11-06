"""Path shim for the scripts copied into paper/v2/scripts/ from the author's private working repository.

There, a module of this name put a pinned checkout of segresearch (commit 1985984) on sys.path and gave scripts a
results root in that repository. Here the scripts live inside segresearch, so this replacement puts the segresearch root
on sys.path and sets HERE to paper/v2, so the scripts read paper/v2/data/ and write paper/v2/results/. The copied
scripts are unchanged apart from their docstrings (and one line of r9_cue_split.py); see paper/v2/README.md.

    import _seg            # before importing blt_load, blt_layouts, realblt_budget, reuse_screen, scale.*
    _seg.HERE              # paper/v2
    _seg.emit(name, line)  # print a RESULT line
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # paper/v2
SEG = os.path.dirname(os.path.dirname(HERE))                                     # the segresearch root
if SEG not in sys.path:
    sys.path.insert(1, SEG)


def emit(script, line, **meta):
    print(line, flush=True)
