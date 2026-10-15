"""Every table and figure of paper 2, built from the result files (local; no plotting dependencies).

    python3 paper2/make_tables.py            # -> paper2/tables/*.md, paper2/figures/*.svg

Sources:
    results/scale/runs/<size>_<rule>_s<seed>[_clip]/result.json     math, same corpus and evaluation at every size
    results/generic/{code,prose}/<rule>_s<seed>.json                  outside math (bits per byte)
    results/generic/code_acc/<rule>_s<seed>.json                      code exact-match accuracy
    results/learned_boundary/{code,math}.json, hnet_code.json         where boundaries fall (structure)
Means over seeds with the per-seed values; "n" = seeds. Missing cells print "–".
"""
import glob
import json
import math
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_T = os.path.join(ROOT, "paper2", "tables")
OUT_F = os.path.join(ROOT, "paper2", "figures")
SIZES = ["1m", "12m", "50m"]


def runs():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results/scale/runs/*/result.json")):
        m = re.fullmatch(r"(\w+?)_(.+)_s(\d)(_clip)?", os.path.basename(os.path.dirname(f)))
        out.setdefault((m.group(1), m.group(2), bool(m.group(4))), []).append(json.load(open(f)))
    return out


def cell(R, key, scale=100, nd=1):
    if not R:
        return "–"
    v = [r[key] * scale for r in R]
    mu = sum(v) / len(v)
    if len(v) == 1:
        return f"{mu:.{nd}f} (1 seed)"
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
    return f"{mu:.{nd}f} ± {sd:.{nd}f}"


def mean(R, key):
    return sum(r[key] for r in R) / len(R) if R else float("nan")


def table_math(R):
    rules = [("entropy10", "Entropy (BLT)"), ("dep10", "Dependence table (paper 1)"),
             ("syntax+entropy10", "Syntax + entropy (hand-written)"), ("nov10", "**Novelty**"),
             ("novc13", "Novelty, corpus-fitted")]
    lines = ["# Table 1. Math: computed-result / final-answer exact match (%) and bits per byte, ~10% patches",
             "", "Mean ± sd over seeds (3 unless marked). 1M and 12M unclipped; 50M with gradient clipping (norm 1.0) "
             "for every rule. Same corpus (SHA-256 pinned), same 1000 computed / 660 final targets at every size.", ""]
    for metric, key, sc, nd in (("Computed results", "computed_acc", 100, 1), ("Final answers", "final_acc", 100, 1),
                                ("Bits per byte", "bpb", 1, 3)):
        lines += [f"## {metric}", "", "| size | " + " | ".join(n for _, n in rules) + " |",
                  "|---|" + "---|" * len(rules)]
        for s in SIZES:
            clip = s == "50m"
            lines.append(f"| {s.upper()} | " + " | ".join(cell(R.get((s, r, clip)), key, sc, nd) for r, _ in rules) + " |")
        lines.append("")
    lines += ["## 50M without clipping (reference; nov10 was only run clipped)", "",
              "| rule | computed | final | bpb |", "|---|---|---|---|"]
    for r, n in rules:
        X = R.get(("50m", r, False))
        if X:
            lines.append(f"| {n} | {cell(X, 'computed_acc')} | {cell(X, 'final_acc')} | {cell(X, 'bpb', 1, 3)} |")
    return "\n".join(lines) + "\n"


def table_negative(R):
    rules = [("entropy10", "Entropy"), ("nov10", "Novelty"), ("typed14", "Typed uncertainty (trained)"),
             ("final+typed14", "Typed + answer cue (trained)")]
    lines = ["# Table 5. Trained typed uncertainty vs entropy and novelty (computed / final, %)", "",
             "| size | " + " | ".join(n for _, n in rules) + " |", "|---|" + "---|" * len(rules)]
    for s in SIZES:
        clip = s == "50m"
        lines.append(f"| {s.upper()} | " + " | ".join(
            (f"{100 * mean(R[(s, r, clip)], 'computed_acc'):.1f} / {100 * mean(R[(s, r, clip)], 'final_acc'):.1f}"
             if R.get((s, r, clip)) else "–") for r, _ in rules) + " |")
    return "\n".join(lines) + "\n"


def generic(dom, sub="", prefix=None):
    out = {}
    for f in glob.glob(os.path.join(ROOT, f"results/generic/{sub or dom}/*_s[0-9].json")):
        r = json.load(open(f))
        out.setdefault(r["rule"], []).append(r)
    return out


def table_outside():
    lines = ["# Table 2. Outside math (12M, bits per byte, 2 seeds; eval patch rate in parentheses)", "",
             "| domain | rule | eval rate | bpb | bpb on new bytes | bpb on copied bytes |", "|---|---|---|---|---|---|"]
    for dom in ("prose", "code"):
        G = generic(dom)
        for rule in sorted(G, key=lambda k: mean(G[k], "val_rate")):
            X = G[rule]
            lines.append(f"| {dom} | {rule} | {100 * mean(X, 'val_rate'):.1f}% | {cell(X, 'bpb', 1, 3)} | "
                         f"{cell(X, 'bpb_new', 1, 3)} | {cell(X, 'bpb_copy', 1, 3)} |")
    A = generic("code", "code_acc")
    kinds = [("id_copy", "copied identifiers"), ("id_new", "new identifiers"), ("num_new", "new numbers"),
             ("rhs_copy", "copied values after = / return / ("), ("rhs_new", "new values after = / return / (")]
    lines += ["", "# Table 3. Code exact match of whole tokens (%), equal eval rate, 2 seeds", "",
              "| target | n | entropy10 | novelty | difference ± binomial SE |", "|---|---|---|---|---|"]
    for k, name in kinds:
        e, v = mean(A["entropy10"], f"acc_{k}"), mean(A["novelty"], f"acc_{k}")
        n = A["entropy10"][0][f"n_{k}"]
        se = math.sqrt(e * (1 - e) / n / 2 + v * (1 - v) / n / 2)
        lines.append(f"| {name} | {n:,} | {100 * e:.1f} | {100 * v:.1f} | {100 * (v - e):+.1f} ± {100 * se:.1f} |")
    return "\n".join(lines) + "\n"


def table_structure():
    lines = ["# Table 4. Where boundaries fall (held-out bytes; ~10% rate unless noted)", "",
             "Coverage = share of the category that starts a patch; share = fraction of all boundaries in the "
             "category (chance = the category's share of bytes).", ""]
    cats = ["line/statement start", "after '='/'return '/'('", "token start", "mid-identifier"]
    for dom in ("math", "code"):
        f = os.path.join(ROOT, f"results/learned_boundary/{dom}.json")
        if not os.path.exists(f):
            continue
        J = json.load(open(f))
        lines += [f"## {dom}", "", "| predictor | " + " | ".join(f"{c}: cov / share (chance)" for c in cats) + " |",
                  "|---|" + "---|" * len(cats)]
        for k in ("novelty_table", "novelty_net", "novelty_oracle", "gain_table", "gain_net"):
            r = J[k]
            lines.append(f"| {k} (rate {100 * r['rate']:.1f}%) | " + " | ".join(
                f"{100 * r[c]['covered']:.0f}% / {100 * r[c]['share']:.0f}% ({100 * r[c]['chance']:.0f}%)" for c in cats) + " |")
        lines.append("")
    f = os.path.join(ROOT, "results/learned_boundary/hnet_code.json")
    if os.path.exists(f):
        J = json.load(open(f))
        lines += ["## H-Net on code (public checkpoints; chunk start at the byte or the preceding space)", "",
                  "| model / stage | chunk rate | statement starts | values after = / return / ( | other token starts | inside identifiers |",
                  "|---|---|---|---|---|---|"]
        for name, r in J["models"].items():
            for st in sorted({k.split("_")[0] for k in r if re.fullmatch(r"s\d", k.split("_")[0])}):
                g = lambda c: f"{100 * r[f'{st}_{c}']['mean']:.0f}% ({100 * r[f'{st}_{c}_sp']['mean']:.0f}%)"
                lines.append(f"| {name} {st} | {100 * r[f'{st}_all']['mean']:.1f}% | {g('stmt')} | {g('rhs')} | {g('tok')} | {g('mid')} |")
    return "\n".join(lines) + "\n"


def svg_scale(R):
    """Figure 1: computed and final accuracy vs model size, entropy vs novelty (two panels)."""
    W, H, pad = 640, 300, 50
    xs = {s: i for i, s in enumerate(SIZES)}
    series = [("entropy10", "#c0392b", "entropy"), ("nov10", "#1f6fb2", "novelty"), ("dep10", "#7f8c8d", "dependence")]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>']
    for p, (key, title) in enumerate((("computed_acc", "Computed results (%)"), ("final_acc", "Final answers (%)"))):
        x0 = pad + p * (W // 2); pw, ph = W // 2 - pad - 20, H - 2 * pad
        X = lambda i: x0 + i * pw / (len(SIZES) - 1)
        Y = lambda v: pad + ph * (1 - v / 100)
        parts.append(f'<text x="{x0 + pw / 2}" y="{pad - 20}" text-anchor="middle" font-weight="bold">{title}</text>')
        parts.append(f'<line x1="{x0}" y1="{pad + ph}" x2="{x0 + pw}" y2="{pad + ph}" stroke="#333"/>')
        parts.append(f'<line x1="{x0}" y1="{pad}" x2="{x0}" y2="{pad + ph}" stroke="#333"/>')
        for t in (0, 25, 50, 75, 100):
            parts.append(f'<text x="{x0 - 6}" y="{Y(t) + 4}" text-anchor="end" fill="#555">{t}</text>')
        for s, i in xs.items():
            parts.append(f'<text x="{X(i)}" y="{pad + ph + 18}" text-anchor="middle">{s.upper()}</text>')
        for rule, col, lab in series:
            pts = [(X(xs[s]), Y(100 * mean(R[(s, rule, s == "50m")], key))) for s in SIZES if R.get((s, rule, s == "50m"))]
            if len(pts) > 1:
                parts.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>')
            for s in SIZES:
                for r in R.get((s, rule, s == "50m"), []):
                    parts.append(f'<circle cx="{X(xs[s]):.1f}" cy="{Y(100 * r[key]):.1f}" r="2.5" fill="{col}" opacity="0.6"/>')
            if p == 0 and pts:
                parts.append(f'<text x="{pts[-1][0] + 6:.1f}" y="{pts[-1][1] + 4:.1f}" fill="{col}">{lab}</text>')
    parts.append(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle" fill="#555">model size (50M: gradient clipping, all rules); dots = seeds</text>')
    return "\n".join(parts + ["</svg>"])


def svg_code_curve():
    """Figure 2: code, bits per byte vs eval patch rate (entropy curve, novelty points)."""
    G = generic("code")
    W, H, pad = 480, 320, 55
    pts = {k: (100 * mean(v, "val_rate"), mean(v, "bpb")) for k, v in G.items()}
    xr = (min(p[0] for p in pts.values()) - 1, max(p[0] for p in pts.values()) + 1)
    yr = (min(p[1] for p in pts.values()) - 0.02, max(p[1] for p in pts.values()) + 0.02)
    X = lambda v: pad + (W - 2 * pad) * (v - xr[0]) / (xr[1] - xr[0])
    Y = lambda v: pad + (H - 2 * pad) * (1 - (v - yr[0]) / (yr[1] - yr[0]))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W / 2}" y="22" text-anchor="middle" font-weight="bold">Code (12M): bits per byte vs patch rate</text>',
             f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="#333"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H - pad}" stroke="#333"/>',
             f'<text x="{W / 2}" y="{H - 15}" text-anchor="middle">eval patch rate (%)</text>',
             f'<text x="15" y="{H / 2}" transform="rotate(-90 15 {H / 2})" text-anchor="middle">bits per byte</text>']
    ent = sorted(v for k, v in pts.items() if k.startswith("entropy"))
    parts.append(f'<polyline fill="none" stroke="#c0392b" stroke-width="2" points="{" ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in ent)}"/>')
    for k, (a, b) in pts.items():
        col = "#c0392b" if k.startswith("entropy") else "#1f6fb2"
        parts.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="4" fill="{col}"/>')
        parts.append(f'<text x="{X(a) + 6:.1f}" y="{Y(b) - 6:.1f}" fill="{col}" font-size="10">{k}</text>')
    for t in range(math.ceil(xr[0]), int(xr[1]) + 1, 2):
        parts.append(f'<text x="{X(t):.1f}" y="{H - pad + 16}" text-anchor="middle" fill="#555">{t}</text>')
    v = math.ceil(yr[0] * 20) / 20
    while v < yr[1]:
        parts.append(f'<text x="{pad - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#555">{v:.2f}</text>'); v += 0.05
    return "\n".join(parts + ["</svg>"])


# ----------------------------------------------------------------------------- compact tables for the manuscript
# BLT-1B inference layouts (GSM8K, 294 test problems, 10% patch rate, 512-byte window restored): computed / final exact
# match, from realblt_combo.py (notes.md "Combinations on BLT-1B" and "Follow-ups on BLT-1B"); not regenerated here.
BLT1B = [("entropy (BLT default rule)", 19.0, 51.4), ("novelty (GSM8K-fitted table)", 56.5, 19.4),
         ("novelty + answer cue", 54.8, 65.6), ("typed uncertainty", 48.7, 32.0), ("typed + answer cue", 48.4, 64.6)]


def pm(R, key, sc=100, nd=1):
    if not R:
        return "–"
    v = [r[key] * sc for r in R]; mu = sum(v) / len(v)
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0
    return f"{mu:.{nd}f}" + (f" ± {sd:.{nd}f}" if len(v) > 1 else "")


def paper_math(R):
    rules = [("entropy10", "Entropy"), ("dep10", "Dependence"), ("syntax+entropy10", "Syntax + entropy"),
             ("nov10", "**Novelty**"), ("novc13", "Novelty (corpus)")]
    L = ["| size | target | " + " | ".join(n for _, n in rules) + " |", "|---|---|" + "---|" * len(rules)]
    for s in SIZES:
        for key, lab, sc, nd in (("computed_acc", "computed", 100, 1), ("final_acc", "final", 100, 1), ("bpb", "bits/byte", 1, 3)):
            L.append(f"| {s.upper() if key == 'computed_acc' else ''} | {lab} | " +
                     " | ".join(pm(R.get((s, r, s == "50m")), key, sc, nd) for r, _ in rules) + " |")
    L += ["", ": Math: exact match (%) on in-line computed results and final answers, and bits per byte, at a "
          "~11% evaluation patch rate. Mean ± sd over 3 seeds (syntax + entropy at 50M not run with clipping). 50M "
          "runs use gradient clipping for every rule."]
    return "\n".join(L) + "\n"


def paper_outside():
    L = ["| domain | rule | eval patch rate | bits/byte | computed values | copied values |", "|---|---|---|---|---|---|"]
    G = generic("prose")
    for r in ("entropy10", "novelty"):
        L.append(f"| prose | {r} | {100 * mean(G[r], 'val_rate'):.1f}% | {pm(G[r], 'bpb', 1, 3)} | – | – |")
    G = generic("code")
    for r in sorted(G, key=lambda k: mean(G[k], "val_rate")):
        L.append(f"| code | {r} | {100 * mean(G[r], 'val_rate'):.1f}% | {pm(G[r], 'bpb', 1, 3)} | – | – |")
    for sub, rules in (("trace", ("entropy10", "novelty")), ("trace_novnet", ("novelty_net",))):
        G = generic("trace", sub)
        for r in rules:
            L.append(f"| traces | {r} | {100 * mean(G[r], 'val_rate'):.1f}% | {pm(G[r], 'bpb', 1, 3)} | "
                     f"{pm(G[r], 'acc_computed')} | {pm(G[r], 'acc_copied')} |")
    L += ["", ": Outside math (12M, 2 seeds; mean ± sd). Code: entropy at four budgets (entropy_tnov, entropy10, "
          "entropy_t14, entropy_t17) and novelty at two (novelty, novelty_t10). Traces: exact match (%) on computed "
          "and copied printed values; novelty_net is the 128-byte novelty predictor (§6)."]
    return "\n".join(L) + "\n"


def paper_codeacc():
    A = generic("code", "code_acc")
    kinds = [("id_copy", "identifiers seen in the previous 128 bytes"), ("id_new", "identifiers not seen"),
             ("num_new", "numbers not seen"), ("rhs_copy", "values after = / return / ( (seen)"),
             ("rhs_new", "values after = / return / ( (not seen)")]
    L = ["| tokens | n | entropy | novelty | difference |", "|---|---|---|---|---|"]
    for k, name in kinds:
        e, v = mean(A["entropy10"], f"acc_{k}"), mean(A["novelty"], f"acc_{k}")
        n = A["entropy10"][0][f"n_{k}"]
        se = math.sqrt(e * (1 - e) / n / 2 + v * (1 - v) / n / 2)
        L.append(f"| {name} | {n:,} | {100 * e:.1f} | {100 * v:.1f} | {100 * (v - e):+.1f} ± {100 * se:.1f} |")
    L += ["", ": Code (12M, matched evaluation rate 7.6–7.7%, 2 seeds): exact match (%) of whole tokens given the "
          "true prefix; difference ± binomial standard error."]
    return "\n".join(L) + "\n"


def paper_enrichment():
    J = json.load(open(os.path.join(ROOT, "results/enrichment.json")))
    names = {"entropy10": "Entropy", "dep10": "Dependence", "syntax+entropy10": "Syntax + entropy",
             "nov10": "Novelty", "novc13": "Novelty (corpus)"}
    L = ["| rule | eval rate | enrichment B | computed-result starts covered | computed (50M) | final (50M) |",
         "|---|---|---|---|---|---|"]
    for k, n in names.items():
        r = J[k]; a = r.get("acc50m_clip")
        L.append(f"| {n} | {100 * r['rate']:.1f}% | {r['B']:.2f} | {100 * r['coverage']['computed']:.0f}% | "
                 + (f"{100 * a['computed_acc']:.1f} | {100 * a['final_acc']:.1f} |" if a else "– | – |"))
    L += ["", f": Boundary enrichment (mean surprisal at patch starts / mean surprisal; surprisal from the "
          f"patcher's byte model) on the math evaluation set at the 50M thresholds, coverage of computed-result starts, "
          f"and 50M accuracy (%, 3 seeds). Mean surprisal at computed-result starts: "
          f"{J['surprisal_at_computed_starts']:.2f} bits; all bytes: 3.12 bits."]
    return "\n".join(L) + "\n"


def paper_negative(R):
    L = ["| setting | rule | computed | final |", "|---|---|---|---|"]
    for s in ("12m", "50m"):
        for r, n in (("entropy10", "entropy"), ("typed14", "typed uncertainty"), ("final+typed14", "typed + answer cue")):
            X = R.get((s, r, s == "50m"))
            L.append(f"| trained, {s.upper()} | {n} | {pm(X, 'computed_acc')} | {pm(X, 'final_acc')} |")
    C = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results/scale/cross/12m_nov10_s*/cross_eval.json")))]
    for r, n in (("nov10", "novelty (as trained)"), ("entropy10", "entropy"), ("typed14", "typed uncertainty")):
        L.append(f"| 12M trained with novelty, rule switched at inference | {n} | {pm([c[r] for c in C], 'computed_acc')} | "
                 f"{pm([c[r] for c in C], 'final_acc')} |")
    for n, c, f in BLT1B:
        L.append(f"| BLT-1B, layout changed at inference (10%) | {n} | {c:.1f} | {f:.1f} |")
    L += ["", ": What did not work, and inference-time layouts (exact match, %). Trained rows: 3 seeds; switched "
          "rows: 2 seeds; BLT-1B: 294 GSM8K test problems [@vera2026easy]."]
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT_T, exist_ok=True); os.makedirs(OUT_F, exist_ok=True)
    R = runs()
    files = {"table1_math.md": table_math(R), "table2_3_outside_math.md": table_outside(),
             "table4_structure.md": table_structure(), "table5_typed.md": table_negative(R)}
    files.update({"paper_math.md": paper_math(R), "paper_outside.md": paper_outside(), "paper_codeacc.md": paper_codeacc(),
                  "paper_enrichment.md": paper_enrichment(), "paper_negative.md": paper_negative(R)})
    for name, txt in files.items():
        open(os.path.join(OUT_T, name), "w").write(txt)
    open(os.path.join(OUT_F, "fig1_scale.svg"), "w").write(svg_scale(R))
    open(os.path.join(OUT_F, "fig2_code_curve.svg"), "w").write(svg_code_curve())
    print("\n".join(f"-> paper2/tables/{n}" for n in files) + "\n-> paper2/figures/fig1_scale.svg, fig2_code_curve.svg")


if __name__ == "__main__":
    main()
