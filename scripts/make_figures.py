"""Report figures, rebuilt from committed result files only (no GPU, no re-training).

    python scripts/make_figures.py            # -> figures/*.pdf (for LaTeX) and figures/*.png (README / slides)

fig1_pipeline        method overview: attackers -> agent control -> detector (adv. training) -> calibration
fig2_roc_opt         ROC on control-verified optimized rewrites vs unseen-server benign tools (robust5/6a/6b)
fig3_judges_1pct     detection at 1% FPR, original vs optimized descriptions, LLM judges vs classifiers
fig4_verification    rule-judge "working" vs control-verified rewrites per attack round (data/attacks/verified.jsonl)
fig5_whitebox        genuine white-box evasions by defense stage, DeBERTa vs E5 (results/*/summary.json)
fig6_final_fpr       false alarms: base vs adversarial training x global vs registry-relative threshold (3 seeds)
fig7_contamination   detection vs poisoned tools already in the registry, max / approved-only / median anchor
fig8_fluency         median GPT-2 log-perplexity of benign tools vs white-box evaders (unconstrained vs natural)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "figures"
# validated categorical order (dataviz reference palette, light mode) + text/grid tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#a8a7a2"
FULL_W, HALF_W = 5.5, 2.7          # NeurIPS text width (in)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "lines.linewidth": 1.6, "patch.linewidth": 0,
})


def save(fig, name):
    """Save a figure as figures/<name>.pdf and .png."""
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})   # no timestamp: reruns give identical files
    fig.savefig(OUT / f"{name}.png", metadata={"Software": None})
    plt.close(fig)
    print("wrote", name)


def jl(p):
    """Read a JSONL file."""
    return [json.loads(line) for line in open(p)]


def thr_at_fpr(neg, fpr):
    """Smallest threshold with at most fpr of negatives >= it (same rule as mcptp.eval.metrics)."""
    s = np.sort(np.asarray(neg, float))[::-1]
    k = int(np.floor(fpr * len(s)))
    return float(np.nextafter(s[k], np.inf)) if k < len(s) else float(s[-1])


def bar_labels(ax, bars, vals, fmt="{:g}", missing="n/r"):
    """Value labels above bars ('n/r' for not run)."""
    for b, v in zip(bars, vals):
        txt = missing if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
        y = 0 if txt == missing else b.get_height()
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, y), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=INK2)


# --------------------------------------------------------------------------------------------------- scores
DEFENDERS = [  # (file, defender id, label, line style)
    ("robust6a", "B2_judge_8b_zs", "Qwen3-8B judge, zero-shot", "--"),
    ("robust6a", "B2_judge_8b_fs", "Qwen3-8B judge, 6-shot", "--"),
    ("robust6a", "B2_judge_14b_zs", "Qwen3-14B judge, zero-shot", "--"),
    ("robust6b", "B2_judge_14b_fs", "Qwen3-14B judge, 6-shot", "--"),
    ("robust5", "B0_tfidf_lr", "TF-IDF + LR", "-"),
    ("robust5", "B1_e5+para", "E5 classifier (fine-tuned)", "-"),
    ("robust5", "B3_plain_deberta+para", "DeBERTa classifier (fine-tuned)", "-"),
]
OPT = ["r0_opt", "r1_opt", "r1b_opt", "r2_opt", "r1t_opt", "r2t_opt"]
ORIG = [s.replace("_opt", "_orig") for s in OPT]


def load_scores():
    """Per-defender, per-seed, per-set scores from robust5/6a/6b."""
    S = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))      # S[defender][seed][set] -> scores
    for f in {d[0] for d in DEFENDERS}:
        for r in jl(RES / f / "scores.jsonl"):
            S[r["defender"]][r["seed"]][r["set"]].append(r["score"])
    return S


def fig2_roc(S):
    """Figure 2: ROC on optimized rewrites."""
    grid = np.logspace(-3, 0, 200)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.8, 2.3))
    for i, (_, d, lab, ls) in enumerate(DEFENDERS):
        tprs, aucs = [], []
        for sd in S[d].values():
            neg = np.asarray(sd["test_neg"]); pos = np.concatenate([sd[s] for s in OPT if s in sd])
            th = np.unique(np.r_[neg, pos, np.inf])[::-1]
            fpr = np.array([(neg >= t).mean() for t in th]); tpr = np.array([(pos >= t).mean() for t in th])
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(np.trapezoid(tpr, fpr))
        ax.plot(grid, np.mean(tprs, 0), ls=ls, color=C[i], label=f"{lab} ({np.mean(aucs):.2f})")
    ax.axvline(0.01, color=MUTED, lw=0.8, ls=":")
    ax.text(0.0108, 1.04, "1% FPR", color=INK2, fontsize=6.5, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1)
    ax.set_xlabel("False-positive rate on unseen-server benign tools (log)")
    ax.set_ylabel("Detection rate (optimized attacks)")
    ax.set_ylim(0, 1.12)
    ax.legend(title="Defender (AUROC on optimized rewrites)", title_fontsize=7, loc="upper left",
              bbox_to_anchor=(-0.02, -0.22), ncol=2, fontsize=6.3, alignment="left")
    save(fig, "fig2_roc_opt")


def fig3_judges(S):
    """Figure 3: detection at 1% FPR, original vs optimized."""
    labs, orig, opt = [], [], []
    for _, d, lab, _ in DEFENDERS:
        o, p = [], []
        for sd in S[d].values():
            t = thr_at_fpr(sd["test_neg"], 0.01)        # matched: 1% realized FPR on the same negatives
            o.append(np.mean(np.concatenate([sd[s] for s in ORIG if s in sd]) >= t))
            p.append(np.mean(np.concatenate([sd[s] for s in OPT if s in sd]) >= t))
        labs.append({"Qwen3-8B judge, zero-shot": "8B\nzero-shot", "Qwen3-8B judge, 6-shot": "8B\n6-shot",
                     "Qwen3-14B judge, zero-shot": "14B\nzero-shot", "Qwen3-14B judge, 6-shot": "14B\n6-shot",
                     "TF-IDF + LR": "TF-IDF\n+ LR", "E5 classifier (fine-tuned)": "E5\nfine-tuned",
                     "DeBERTa classifier (fine-tuned)": "DeBERTa-v3\nfine-tuned"}[lab])
        orig.append(np.mean(o)); opt.append(np.mean(p))
    x = np.arange(len(labs)); w = 0.38
    fig, ax = plt.subplots(figsize=(FULL_W, 2.2))
    b1 = ax.bar(x - w / 2 - 0.01, orig, w, color=C[0], label="Original MCPTox description")
    b2 = ax.bar(x + w / 2 + 0.01, opt, w, color=C[1], label="Optimized rewrite (control-verified)")
    bar_labels(ax, b1, orig, "{:.2f}"); bar_labels(ax, b2, opt, "{:.2f}")
    ax.axvline(3.5, color=MUTED, lw=0.8)
    ax.text(1.5, 1.12, "Qwen3 LLM-as-judge", ha="center", color=INK2, fontsize=7)
    ax.text(5.0, 1.12, "Trained classifiers", ha="center", color=INK2, fontsize=7)
    ax.set_xticks(x, labs); ax.set_ylim(0, 1.2); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel("Detection at 1% FPR")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.25), ncol=2)
    save(fig, "fig3_judges_1pct")


# --------------------------------------------------------------------------------------------- other figures
def fig4_verification():
    """Figure 4: rule-judge 'working' vs control-verified rewrites."""
    rows = [r for r in jl(ROOT / "data" / "attacks" / "verified.jsonl") if r["kind"] == "opt"]
    rounds = ["r0", "r1", "r1b", "r2", "r1t", "r2t"]
    names = ["r0\nblind", "r1\nadaptive", "r1b\nrepeat", "r2\n14B", "r1t\ntest", "r2t\ntest"]
    raw = [sum(r["asr_raw"] >= 0.2 for r in rows if r["round"] == k) for k in rounds]
    ver = [sum(r["asr_true"] >= 0.2 for r in rows if r["round"] == k) for k in rounds]
    x = np.arange(len(rounds)); w = 0.38
    fig, ax = plt.subplots(figsize=(HALF_W + 0.9, 2.2))
    b1 = ax.bar(x - w / 2 - 0.01, raw, w, color=C[0], label="Rule judge says “working”")
    b2 = ax.bar(x + w / 2 + 0.01, ver, w, color=C[1], label="Still works vs no-poison control")
    bar_labels(ax, b1, raw); bar_labels(ax, b2, ver)
    ax.set_xticks(x, names); ax.set_ylabel("Optimized rewrites")
    ax.set_ylim(0, max(raw) * 1.15)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
    tot = (sum(raw), sum(ver))
    ax.text(5.4, max(raw) * 1.05, f"{tot[1]}/{tot[0]} ({tot[1] / tot[0]:.0%})\nsurvive the control", ha="right",
            va="top", fontsize=7, color=INK2)
    save(fig, "fig4_verification")


def summ(run):
    """Number of genuine evasions in a white-box run summary."""
    p = RES / run / "summary.json"
    return json.load(open(p)).get("n_genuine_evasions") if p.exists() else None


def fig5_whitebox():
    """Figure 5: genuine white-box evasions by defense stage."""
    stages = [("Base\n15% budget", "wb1", "wb1e5"), ("Base\n30%", "wb3_base_b30_deb", "wb3_base_b30_e5"),
              ("Adv. rd. 1\n30%", "wb3_adv_b30_deb", "wb3_adv_b30_e5"), ("Adv. rd. 2\n30%", "wb4_adv2_b30_deb", None),
              ("Final\nsystem", "wb5_final_cal_b30", "wb5_e5_final_cal_b30"),
              ("Base", "nat_base_deb", "nat_base_e5"),
              ("Final\nsystem", "nat_final_cal_deb", "nat_final_cal_e5")]
    deb = [summ(a) for _, a, _ in stages]; e5 = [summ(b) if b else None for _, _, b in stages]
    x = np.arange(len(stages)); w = 0.38
    fig, ax = plt.subplots(figsize=(FULL_W, 2.2))
    b1 = ax.bar(x - w / 2 - 0.01, [v or 0 for v in deb], w, color=C[0], label="DeBERTa-v3")
    b2 = ax.bar(x + w / 2 + 0.01, [v or 0 for v in e5], w, color=C[1], label="multilingual-E5")
    bar_labels(ax, b1, deb); bar_labels(ax, b2, e5)
    ax.axvline(4.5, color=MUTED, lw=0.8)
    ax.text(2.0, 98, "Unconstrained word swaps", ha="center", color=INK2, fontsize=7)
    ax.text(5.5, 98, "Natural text", ha="center", color=INK2, fontsize=7)
    ax.set_xticks(x, [s[0] for s in stages]); ax.set_ylim(0, 105)
    ax.set_ylabel("Genuine evasions (of 379)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, 0.9))
    save(fig, "fig5_whitebox")


def fig6_final():
    """Figure 6: false alarms, base vs adversarial training x global vs calibrated."""
    runs = {"DeBERTa-v3": ["cal_adv2_deb", "cal_adv2_s43", "cal_adv2_s44"],
            "multilingual-E5": ["cal_adv2_e5_s42", "cal_adv2_e5_s43", "cal_adv2_e5_s44"]}
    variants = [("base", "global", "Base, global threshold"), ("base", "rel", "Base + calibration"),
                ("adv", "global", "Adv. trained, global"), ("adv", "rel", "Adv. trained + calibration (final)")]
    sets = [("test_neg@0.01", "Unseen-server\nbenign tools"), ("test_hardneg@0.01", "Hard negatives\n(“call X first”)")]
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.2), sharey=True)
    for ax, (bb, rs) in zip(axs, runs.items()):
        J = [json.load(open(RES / r / "cal_adv.json")) for r in rs]
        x = np.arange(len(sets)); w = 0.2
        for i, (v, m, lab) in enumerate(variants):
            vals = np.array([[j[v][k][m] for k, _ in sets] for j in J]) * 100
            mu, sd = vals.mean(0), vals.std(0)
            bars = ax.bar(x + (i - 1.5) * (w + 0.015), mu, w, yerr=sd, color=C[i], label=lab,
                          error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 1.5})
            if v == "adv" and m == "rel":
                for b, a, e in zip(bars, mu, sd):
                    ax.annotate(f"{a:.1f}", (b.get_x() + b.get_width() / 2, a + e), xytext=(0, 2),
                                textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.axhline(1.0, color=INK2, lw=0.8, ls=":", label="1% target" if bb.startswith("DeB") else None)
        ax.set_xticks(x, [s[1] for s in sets]); ax.set_title(bb, loc="left")
    axs[0].set_ylabel("False-positive rate (%)\nat 1% validation target")
    h, lab_ = axs[0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))
    save(fig, "fig6_final_fpr")


def fig7_contam():
    """Figure 7: detection vs contaminated registry."""
    rules = [("max", "max anchor (as deployed)", "o"), ("approved", "approved-only anchor", "s"),
             ("median", "median anchor", "^")]
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.1), sharey=True)
    for ax, (tag, bb) in zip(axs, (("contam_deb", "DeBERTa-v3"), ("contam_e5", "multilingual-E5"))):
        S = json.load(open(RES / tag / "contam.json"))["mean_std"]["naive"]
        fprs = []
        for i, (r, lab, mk) in enumerate(rules):
            ks = sorted(S[r], key=lambda k: int(k[2:]))
            xs = np.array([int(k[2:]) for k in ks]) + (i - 1) * 0.06      # small dodge: approved/median overlap at 1.0
            mu = [S[r][k]["test_pos"][0] for k in ks]; sd = [S[r][k]["test_pos"][1] for k in ks]
            ax.errorbar(xs, mu, yerr=sd, color=C[i], marker=mk, ms=4, capsize=1.5, elinewidth=0.8, label=lab,
                        mec="white", mew=0.6)
            fprs.append(f"{lab.split(' (')[0].replace(' anchor', '')}: {S[r]['K=0']['test_neg'][0]:.1%}")
        ax.text(3.1, 0.62, "clean FPR\n" + "\n".join(fprs), ha="right", va="top", fontsize=6.3, color=INK2)
        ax.set_xticks([0, 1, 2, 3]); ax.set_ylim(-0.03, 1.05); ax.set_title(bb, loc="left")
    axs[0].set_ylabel("Detection on test poisonings")
    fig.supxlabel("Poisoned tools already in the server registry", fontsize=8, color=INK2, y=-0.04)
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig7_contamination")


def fig8_fluency():
    """Figure 8: GPT-2 log-perplexity of evaders."""
    def med(tag, prefix):
        j = json.load(open(RES / tag / "defense_results.json"))["fluency"]["median_nll"]
        return next(v for k, v in j.items() if prefix in k and k.endswith("_evaders"))
    base = json.load(open(RES / "fluency_nat_deb" / "defense_results.json"))["fluency"]["median_nll"]
    rows = [("Benign test tools", base["test_neg"], MUTED), ("Black-box LLM rewrites", base["blackbox_opt"], MUTED),
            ("White-box, unconstrained — DeBERTa", med("fluency_nat_deb", "wb3_base_b30_deb"), C[0]),
            ("White-box, unconstrained — E5", med("fluency_nat_e5", "wb3_base_b30_e5"), C[1]),
            ("White-box, natural — DeBERTa", med("fluency_nat_deb", "nat_base_deb"), C[0]),
            ("White-box, natural — E5", med("fluency_nat_e5", "nat_base_e5"), C[1])]
    fig, ax = plt.subplots(figsize=(HALF_W + 0.6, 1.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col) in zip(y, rows):
        ax.hlines(yi, 3.5, v, color=GRID, lw=1.2)
        ax.plot(v, yi, "o", color=col, ms=6, mec="white", mew=1)
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.5,
                    color=INK2)
    ax.axvline(rows[0][1], color=INK2, lw=0.8, ls=":")
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(3.5, 7.0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median GPT-2 log-perplexity (nats/token, evaders)")
    save(fig, "fig8_fluency")


def fig1_pipeline():
    """Figure 1: method overview."""
    fig, ax = plt.subplots(figsize=(FULL_W, 1.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")
    W, GAP, Y, H = 18.2, 2.25, 13, 18

    def box(i, title, body, col):
        x = i * (W + GAP)
        ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=col, lw=1.1))
        ax.add_patch(FancyBboxPatch((x, Y + H - 4.2), W, 4.2, boxstyle="round,pad=0,rounding_size=1.2", fc=col,
                                    ec=col, lw=0))
        ax.text(x + W / 2, Y + H - 2.1, title, ha="center", va="center", fontsize=6.8, weight="bold", color="white")
        ax.text(x + W / 2, Y + H - 6.2, body, ha="center", va="top", fontsize=5.15, color=INK2, linespacing=1.35)
        return x

    xs = [box(0, "MCPTox seeds", "45 real servers\nserver-disjoint split\n+ MCPTox-Para\n(paraphrased test)", C[6]),
          box(1, "Attackers", "black-box LLM loop\n(Qwen3-4B / 14B)\nwhite-box word swaps\n(15%, 30%, natural)", C[1]),
          box(2, "Agent control", "Qwen3-8B, Llama-3.1\nwith vs. without tool\nkeep only attacks\nthat still hijack", C[3]),
          box(3, "Detector", "DeBERTa-v3 / E5\nfine-tuned classifier\n+ adversarial training\n(2 rounds)", C[0]),
          box(4, "Calibration", "score − max score of\napproved tools on\nthe same server\n→ accept / reject", C[2])]
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + W + 0.2, Y + H / 2), (x + W + GAP - 0.2, Y + H / 2), arrowstyle="-|>",
                                     mutation_scale=7, color=INK2, lw=0.9))
    # feedback loop under the boxes: detector -> attackers
    ax.add_patch(FancyArrowPatch((xs[3] + W / 2, Y - 0.3), (xs[1] + W / 2, Y - 0.3), arrowstyle="-|>",
                                 mutation_scale=7, color=C[0], lw=0.9, connectionstyle="arc3,rad=-0.12"))
    ax.text((xs[1] + xs[3]) / 2 + W / 2, 1.0,
            "scores / gradients guide the attacker;  genuine evasions on train seeds → adversarial training data",
            ha="center", fontsize=5.8, color=INK2)
    save(fig, "fig1_pipeline")


def main():
    """Build every figure."""
    S = load_scores()
    fig1_pipeline(); fig2_roc(S); fig3_judges(S); fig4_verification(); fig5_whitebox(); fig6_final()
    fig7_contam(); fig8_fluency()


if __name__ == "__main__":
    main()
