"""Patch rules for the scaling pilot (#27), in pure numpy (no MLX), so masks can be built on any machine.

The same rules and thresholds as mathexp.Rule for the arms used here (scale/check_rules.py checks they give
identical masks):

    entropy{R}         patch where the small model's next-byte entropy is in the top R% of train positions
    dep{R}             patch where boundary dependence (a 2-byte-context lookup table) is in the top R%
    jump{R}            patch where the entropy RISES most from the previous byte (BLT's approximate monotonic rule):
                       top R% of H[t] - H[t-1] on train
    nov{R}             patch where the self-supervised novelty table (expected share of upcoming bytes that do NOT repeat
                       the last 128; reuse_screen.py, #31) is in the top R%: the dep rule's form with another table
    typed{R}           patch where the small model's typed uncertainty (sure of the kind of the next byte, not of
                       which one; scale.patchers.typed_from_logprobs) is in the top R%. Masks need the typed values T
                       (mask(b, H, T)); without them the rule scores the window itself with its entropy model .ent
    novc{R}            as nov{R}, with the novelty table fitted on the training corpus itself (novelty_corpus.npz:
                       4M bytes sampled from the 1.9 GB corpus, mostly OpenWebMath), not on GSM8K/MATH solutions
    final+typed{R}     a patch forced after the final-answer markers '#### ' and '\\boxed{' (format cue), the rest of
                       the budget by typed uncertainty (mathexp final+typedR)
    syntax+entropy{R}  patch right after every math-syntax marker ('=', '\\boxed{', '#### ', '>>'), the rest of
                       the budget filled by entropy (the hand-written results rule)
    novtop{Q}+entropy{R}  (added 2026-10-24, private working repository paper2v2/pod_patch; issue #29 R8c) patch where the
                       novelty table (as nov) is in the top Q% of train positions, the rest of the budget R filled by
                       entropy: the syntax+entropy construction with the table's strongest positions as the markers
    probetop{Q}+entropy{R}  (added 2026-10-27, paper2v2/pod_patch; R13) patch where a label-free probe on the entropy
                       model's own state (logistic, trained on "a number not seen earlier in the problem starts here",
                       auto-labels from math_train) is in the top Q% of train positions, at positions after a
                       non-alphanumeric ASCII byte; the rest of the budget R filled by entropy. Masks take the probe
                       scores P as the third argument (mask(b, H, P)); without them the rule scores the window itself
                       with its entropy model .ent

Entry t means "byte t starts a patch". Every rule depends only on bytes before t and entropy up to t.
"""
import re

import numpy as np

SYNTAX = [b"=", b"\\boxed{", b"#### ", b">>"]
FINAL = [b"#### ", b"\\boxed{"]                     # final-answer markers (mathexp.FINAL)


def syntax_starts(b, markers=None):
    """True at the byte right after any math-syntax marker (mathexp._syntax_starts)."""
    m = np.zeros(len(b), bool)
    s = b.tobytes()
    for tok in markers or SYNTAX:
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


def state_stream(ent, b, batch=512):
    """The entropy model's final (normed) state predicting each byte of b, (len(b), d) float32: windows as
    patchers.score_stream for arrays longer than CTX (the first CTX/2 + 1 positions stay 0, as H), one window for
    short ones (decoding windows, as patchers.typed_window; position 0 stays 0)."""
    import torch
    from scale.model import ln
    from scale.train import CTX

    def states(X):
        T = X.shape[1]
        h = ent.emb[X] + ent.pos[:T]
        ar = torch.arange(T, device=X.device)
        m = (ar[:, None] >= ar[None])[None].expand(X.shape[0], T, T)
        for L in ent.layers:
            h = L(h, m, ent.heads)
        return ln(h, ent.lnf)
    dev = next(ent.parameters()).device
    d = ent.emb.shape[1]
    S = np.zeros((len(b), d), np.float32)
    with torch.no_grad():
        if len(b) <= CTX:
            if len(b) > 1:
                S[1:] = states(torch.from_numpy(np.asarray(b)[None, :-1].astype(np.int64)).to(dev))[0].float().cpu().numpy()
            return S
        half = CTX // 2
        starts = np.arange(0, len(b) - CTX, half)
        for k in range(0, len(starts), batch):
            st = starts[k:k + batch]
            X = torch.from_numpy(np.stack([b[i:i + CTX] for i in st]).astype(np.int64)).to(dev)
            lo, hi = st[0] + half + 1, st[-1] + CTX + 1
            S[lo:hi] = states(X)[:, half:].float().cpu().numpy().reshape(-1, d)     # state at j predicts byte j+1
    return S


def probe_cand(b):
    """Positions where the probe may start a patch: the previous byte is ASCII and not alphanumeric."""
    p = np.r_[0, np.asarray(b[:-1])].astype(np.int64)
    c = ~(((p >= 48) & (p <= 57)) | ((p >= 65) & (p <= 90)) | ((p >= 97) & (p <= 122))) & (p < 128)
    c[:1] = False
    return c


def probe_scores(ent, b, H, probe):
    """Logistic probe score at every position of b, from the entropy model's state and the entropy H."""
    mu, sd, w = (np.asarray(probe[k], np.float32) for k in ("mu", "sd", "w"))
    d = len(mu)
    S = state_stream(ent, b)
    return (((S - mu) / sd) @ w[:d] + np.asarray(H, np.float32) * w[d] + w[d + 1]).astype(np.float32)


class Rule:
    """Fit thresholds on train bytes / entropies, then mask any byte array: Rule(name, train_b, train_H, table)
    (table: the dependence table for dep, the novelty table for nov)."""

    def __init__(self, name, train_b, train_H, table=None, train_T=None):
        if name.startswith("novtop"):
            Q, R = (int(x) / 100 for x in re.fullmatch(r"novtop(\d+)\+entropy(\d+)", name).groups())
            self.name, self.kind = name, "novtop"
            self.table = (np.asarray(table["T2"]), np.asarray(table["T1"]), float(table["glob"]))
            v = self.dep(train_b)
            self.thr_tab = np.float64(np.quantile(v, 1 - Q))
            top = v > self.thr_tab; rest = ~top
            self.thr = np.quantile(train_H[rest], 1 - (R - top.mean()) / rest.mean())
            return
        m = re.fullmatch(r"(entropy|jump|dep|novc|nov|typed|final\+typed|syntax\+entropy)(\d+)", name)
        if not m:
            raise ValueError(f"unknown rule {name}")
        self.name, self.kind, R = name, m.group(1), int(m.group(2)) / 100
        if self.kind in ("dep", "nov", "novc"):
            self.table = (np.asarray(table["T2"]), np.asarray(table["T1"]), float(table["glob"]))
            self.thr = np.quantile(self.dep(train_b), 1 - R)
        elif self.kind == "entropy":
            self.thr = np.quantile(train_H, 1 - R)
        elif self.kind == "typed":
            self.thr = np.quantile(train_T, 1 - R)
        elif self.kind == "final+typed":
            fin = syntax_starts(train_b, FINAL); rest = ~fin
            self.thr = np.quantile(train_T[rest], 1 - (R - fin.mean()) / rest.mean())
        elif self.kind == "jump":
            self.thr = np.quantile(np.diff(train_H, prepend=train_H[0]), 1 - R)
        else:
            syn = syntax_starts(train_b)
            rest = ~syn
            self.thr = np.quantile(train_H[rest], 1 - (R - syn.mean()) / rest.mean())

    def dep(self, b):
        return table_lookup(*self.table, b) if len(b) else np.zeros(0, np.float32)

    def mask(self, b, H, T=None):
        if self.kind == "probetop":
            P = T if T is not None else probe_scores(self.ent, b, H, self.probe)
            return ((P > self.thr_p) & probe_cand(b)) | (H > self.thr)
        if self.kind in ("typed", "final+typed"):
            if T is None:
                from scale.patchers import typed_window
                T = typed_window(self.ent, b)
            return (T > self.thr) | (syntax_starts(b, FINAL) if self.kind == "final+typed" else False)
        if self.kind in ("dep", "nov", "novc"):
            return self.dep(b) > self.thr
        if self.kind == "novtop":
            return (self.dep(b) > self.thr_tab) | (H > self.thr)
        if self.kind == "entropy":
            return H > self.thr
        if self.kind == "jump":
            return (np.diff(H, prepend=H[0]) if len(H) else H) > self.thr
        return syntax_starts(b) | (H > self.thr)

    def state(self):
        """Everything needed to rebuild the rule elsewhere (thresholds fitted on the Mac's train split)."""
        st = {"name": self.name, "thr": float(self.thr)}
        if self.kind == "novtop":
            st["thr_tab"] = float(self.thr_tab)
        if self.kind == "probetop":
            st["thr_p"] = float(self.thr_p)
            st["probe"] = {k: [float(x) for x in self.probe[k]] for k in ("mu", "sd", "w")}
        return st

    @classmethod
    def from_state(cls, st, table=None):
        r = cls.__new__(cls)
        r.name = st["name"]
        if r.name.startswith("novtop"):
            r.kind = "novtop"
            r.thr, r.thr_tab = np.float64(st["thr"]), np.float64(st["thr_tab"])
            r.table = (np.asarray(table["T2"]), np.asarray(table["T1"]), float(table["glob"]))
            return r
        if r.name.startswith("probetop"):
            r.kind = "probetop"
            r.thr, r.thr_p = np.float64(st["thr"]), np.float64(st["thr_p"])
            r.probe = st["probe"]
            return r
        r.kind = re.fullmatch(r"(entropy|jump|dep|novc|nov|typed|final\+typed|syntax\+entropy)(\d+)", r.name).group(1)
        r.thr = np.float64(st["thr"])         # compare in float64, as the fitted threshold does: a plain Python float
                                              # against a float32 array compares in float32 under NumPy 2 and can flip ties
        if r.kind in ("dep", "nov", "novc"):
            r.table = (np.asarray(table["T2"]), np.asarray(table["T1"]), float(table["glob"]))
        return r
