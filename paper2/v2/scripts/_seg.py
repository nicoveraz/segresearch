"""Public stand-in for the _seg.py of the author's private working repository (original sha256 563fdcaa..., quoted in
../README.md). The scripts in this folder ran from the root of that repository, with segresearch as a submodule pinned
to 1985984. Here the package folder (paper2/v2) stands in for that root, and the segresearch checkout that contains it
stands in for the submodule. Only HERE and SEG differ from the original.

    import _seg            # before importing blt_load, blt_layouts, realblt_budget, typed_screen, scale.*
    _seg.HERE              # paper2/v2: results/, data/, rules/ and figures/ live here
    _seg.emit(name, line)  # print a RESULT line (replaces segresearch's registry.emit here)
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # paper2/v2
SEG = os.path.dirname(os.path.dirname(HERE))                             # the segresearch root
if SEG not in sys.path:
    sys.path.insert(1, SEG)


def emit(script, line, **meta):
    print(line, flush=True)
