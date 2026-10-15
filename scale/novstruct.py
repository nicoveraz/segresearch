import sys, glob, os, re, numpy as np
sys.path.insert(0, os.path.expanduser("~/Documents/GitHub/segresearch"))
from scale.rules import table_lookup, syntax_starts, SYNTAX
from scale import generic as G
B = os.path.expanduser("~/.cache/segresearch-scale/bundle")

def ctx_report(tab, name, k=15):
    T2 = np.asarray(tab["T2"]); n = np.where(np.isnan(T2), -1, T2)
    top = np.argsort(-n)[:k]
    print(f"  top {name} contexts (prev 2 bytes -> novelty):", ", ".join(f"{bytes([i//256, i%256])!r}:{n[i]:.2f}" for i in top))

def stats(b, m, marks, label):
    for nm, s in marks.items():
        cov = (m & s).sum() / max(s.sum(), 1); prec = (m & s).sum() / max(m.sum(), 1)
        print(f"  {label}: rate {m.mean():.1%} | {nm}: {s.mean():.1%} of bytes; covered {cov:.0%}; share of boundaries {prec:.0%} (chance {s.mean():.0%})")

# math
val = np.fromfile(f"{B}/val.u8", np.uint8)[:3_000_000]
tr = np.fromfile(f"{B}/math_train.u8", np.uint8)[:3_000_000]
for f, R in (("novelty.npz", .10), ("novelty_corpus.npz", .13)):
    t = np.load(f"{B}/{f}"); tab = (t["T2"], t["T1"], float(t["glob"]))
    thr = np.quantile(table_lookup(*tab, tr), 1 - R); m = table_lookup(*tab, val) > thr
    s = val.tobytes()
    after_eq = syntax_starts(val, [b"=", b"= "])
    nl = np.r_[False, val[:-1] == 10]
    word = np.r_[True, ~np.isin(val[:-1], np.frombuffer(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", np.uint8))] & np.isin(val, np.frombuffer(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", np.uint8))
    print(f"MATH {f} at {R:.0%}")
    ctx_report(t, "math")
    stats(val, m, {"after '='": after_eq, "math syntax (all)": syntax_starts(val), "line start": nl, "token start": word}, "novelty")

# code: Python files on this machine (as generic.py's code domain)
files = sorted(glob.glob("/opt/homebrew/lib/python3*/**/*.py", recursive=True) + glob.glob(os.path.expanduser("~/.local/share/uv/python/*/lib/python3*/**/*.py"), recursive=True))
rng = np.random.default_rng(0); rng.shuffle(files)
buf = b"".join(open(f, "rb").read()[:10000] for f in files[:400])
b = np.frombuffer(buf, np.uint8); n = len(b) // 2; trc, vac = b[:n], b[n:]
fm = G.forward_match(trc); tab = G.tab_fit(trc, -fm)
thr = np.quantile(table_lookup(*tab, trc), 0.9); m = table_lookup(*tab, vac) > thr
alnum = np.frombuffer(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", np.uint8)
tok = np.isin(vac, alnum) & np.r_[True, ~np.isin(vac[:-1], alnum)]
nl = np.r_[False, vac[:-1] == 10]
stmt = tok & np.r_[False, (vac[:-1] == 32) | (vac[:-1] == 10)] & np.array([False] + [False]*0 + [True]*(len(vac)-1))
# statement start: first non-space byte of a line
first = np.zeros(len(vac), bool); s = vac.tobytes(); i = 0
for mm in re.finditer(rb"\n[ \t]*(\S)", s): first[mm.start(1)] = True
rhs = syntax_starts(vac, [b"= ", b"return ", b"("])
print(f"CODE ({len(b)/1e6:.1f} MB of Python, half to fit), novelty at 10% train")
ctx_report({"T2": tab[0]}, "code")
stats(vac, m, {"statement start": first, "after '= '/'return '/'('": rhs, "token start": tok, "mid-identifier": np.isin(vac, alnum) & ~tok}, "novelty")
