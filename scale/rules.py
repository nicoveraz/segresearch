"""Patch rules for the scaling pilot (#27), in pure numpy (no MLX), so masks can be built on any machine.

The same rules and thresholds as mathexp.Rule for the arms used here (scale/check_rules.py checks they give
identical masks):

    entropy{R}         patch where the small model's next-byte entropy is in the top R% of train positions
    dep{R}             patch where boundary dependence (a 2-byte-context lookup table) is in the top R%
    jump{R}            patch where the entropy RISES most from the previous byte (BLT's approximate monotonic rule):
                       top R% of H[t] - H[t-1] on train
    nov{R}             patch where the self-supervised novelty table (expected share of upcoming bytes that do NOT repeat
                       the last 128; reuse_screen.py, #31) is in the top R%: the dep rule's form with another table
    syntax+entropy{R}  patch right after every math-syntax marker ('=', '\\boxed{', '#### ', '>>'), the rest of
                       the budget filled by entropy (the hand-written results rule)

Entry t means "byte t starts a patch". Every rule depends only on bytes before t and entropy up to t.
"""
import re

import numpy as np

SYNTAX = [b"=", b"\\boxed{", b"#### ", b">>"]


def syntax_starts(b):
    """True at the byte right after any math-syntax marker (mathexp._syntax_starts)."""
    m = np.zeros(len(b), bool)
    s = b.tobytes()
    for tok in SYNTAX:
        i = s.find(tok)
        while i >= 0:
            if i + len(tok) < len(b):
                m[i + len(tok)] = True
            i = s.find(tok, i + 1)
    return m


def table_lookup(T2, T1, glob, b):
    """Dependence for every position t of b, from context (b[t-2], b[t-1]) (deptrigger.table_lookup)."""
    b = b.astype(np.int64)
    p1 = np.r_[0, b[:-1]]
    p2 = np.r_[0, 0, b[:-2]]
    d = T2[p2 * 256 + p1]
    d = np.where(np.isnan(d), T1[p1], d)
    return np.where(np.isnan(d), glob, d).astype(np.float32)


class Rule:
    """Fit thresholds on train bytes / entropies, then mask any byte array: Rule(name, train_b, train_H, table)
    (table: the dependence table for dep, the novelty table for nov)."""

    def __init__(self, name, train_b, train_H, table=None):
        m = re.fullmatch(r"(entropy|jump|dep|nov|syntax\+entropy)(\d+)", name)
        if not m:
            raise ValueError(f"unknown rule {name}")
        self.name, self.kind, R = name, m.group(1), int(m.group(2)) / 100
        if self.kind in ("dep", "nov"):
            self.table = (np.asarray(table["T2"]), np.asarray(table["T1"]), float(table["glob"]))
            self.thr = np.quantile(self.dep(train_b), 1 - R)
        elif self.kind == "entropy":
            self.thr = np.quantile(train_H, 1 - R)
        elif self.kind == "jump":
            self.thr = np.quantile(np.diff(train_H, prepend=train_H[0]), 1 - R)
        else:
            syn = syntax_starts(train_b)
            rest = ~syn
            self.thr = np.quantile(train_H[rest], 1 - (R - syn.mean()) / rest.mean())

    def dep(self, b):
        return table_lookup(*self.table, b) if len(b) else np.zeros(0, np.float32)

    def mask(self, b, H):
        if self.kind in ("dep", "nov"):
            return self.dep(b) > self.thr
        if self.kind == "entropy":
            return H > self.thr
        if self.kind == "jump":
            return (np.diff(H, prepend=H[0]) if len(H) else H) > self.thr
        return syntax_starts(b) | (H > self.thr)

    def state(self):
        """Everything needed to rebuild the rule elsewhere (thresholds fitted on the Mac's train split)."""
        return {"name": self.name, "thr": float(self.thr)}

    @classmethod
    def from_state(cls, st, table=None):
        r = cls.__new__(cls)
        r.name = st["name"]
        r.kind = re.fullmatch(r"(entropy|jump|dep|nov|syntax\+entropy)(\d+)", r.name).group(1)
        r.thr = np.float64(st["thr"])         # compare in float64, as the fitted threshold does: a plain Python float
                                              # against a float32 array compares in float32 under NumPy 2 and can flip ties
        if r.kind in ("dep", "nov"):
            r.table = (np.asarray(table["T2"]), np.asarray(table["T1"]), float(table["glob"]))
        return r
