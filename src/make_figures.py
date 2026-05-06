#!/usr/bin/env python3
"""Generate the headline figure for the report.

Single-panel bar plot: pre/post accuracy by protocol. Each protocol gets its
own colour (pre = lighter, post = darker), with 95% topic-clustered bootstrap
CIs as errorbars. Δ gain (pp) and bold McNemar significance stars are
annotated above each pair (cross-subtopic gets no significance test since
pre/post are on different items).

Output: docs/figures/headline_results.png and headline_results.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_full import (build_data, build_pooled_single_judge_debate,
                            _bootstrap_topic_cluster, RNG, N_BOOT)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (14, 9),
    #    "font.family": "serif",
        "mathtext.fontset": "cm",
        "legend.fontsize": "medium",
        "legend.title_fontsize": 18,
        "axes.titlesize": 18,
        "axes.labelsize": "large",
        "ytick.labelsize": 15,
        "xtick.labelsize": 15,
        # colour-consistent theme
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage[version=3]{mhchem}"

# ── Helpers ──────────────────────────────────────────────────────

def _wilson_ci(k, n, alpha=0.05):
    """Wilson score CI for a binomial proportion (no clustering)."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (centre - half, centre + half)


def _bootstrap_topic_ci_for_acc(df, col, n_boot=2000):
    """Topic-clustered bootstrap CI on a single-arm proportion."""
    topics = df["topic"].unique()
    out = np.empty(n_boot)
    for i in range(n_boot):
        sampled = RNG.choice(topics, size=len(topics), replace=True)
        boot = pd.concat([df[df["topic"] == t] for t in sampled],
                         ignore_index=True)
        out[i] = boot[col].mean()
    return tuple(np.percentile(out, [2.5, 97.5]))


def _mcnemar_p(df):
    b = int(((df["pre_correct"] == 1) & (df["post_correct"] == 0)).sum())
    c = int(((df["pre_correct"] == 0) & (df["post_correct"] == 1)).sum())
    if b + c == 0:
        return np.nan
    return stats.binomtest(min(b, c), b + c, 0.5).pvalue


def _balance_mjh(mjh, rng):
    """Drop random all-pre-correct rooms until pre-acc within ±0.02 of 0.5."""
    rooms = mjh.groupby("al_survey_id")["pre_correct"]
    all_correct = sorted(s for s, vals in rooms if (vals == 1).all())
    rng.shuffle(all_correct)
    sub = mjh.copy()
    pa = sub["pre_correct"].mean()
    for room in all_correct:
        if abs(pa - 0.5) <= 0.02 or pa <= 0.5:
            break
        cand = sub[sub["al_survey_id"] != room]
        new_pa = cand["pre_correct"].mean()
        if abs(new_pa - 0.5) < abs(pa - 0.5):
            sub = cand
            pa = new_pa
    return sub


def collect_protocol_rows(data):
    """Build the per-protocol summary rows for Panel A.

    For each protocol slice, returns: n, pre_acc, pre_ci, post_acc, post_ci,
    delta, mcnemar_p (or NaN if not applicable). Topic-clustered CIs.
    """
    rows = []
    proto_label = {
        "consultancy":    "Consultancy",
        "in_subtopic":    "In-subtopic",
        "cross_subtopic": "Cross-subtopic\n(post on different item)",
        "hybrid_mj":      "Hybrid multi-judge",
        "multi_judge_h":  "Multi-judge humans (full)",
    }
    paired = {"in_subtopic", "consultancy", "multi_judge_h", "hybrid_mj"}
    for proto, label in proto_label.items():
        sub = data[data["protocol"] == proto]
        if len(sub) == 0:
            continue
        n = len(sub)
        pre_acc = sub["pre_correct"].mean()
        post_acc = sub["post_correct"].mean()
        pre_ci = _bootstrap_topic_ci_for_acc(sub, "pre_correct")
        post_ci = _bootstrap_topic_ci_for_acc(sub, "post_correct")
        mc_p = _mcnemar_p(sub) if proto in paired else np.nan
        rows.append({
            "label": label, "n": n,
            "pre": pre_acc, "pre_ci": pre_ci,
            "post": post_acc, "post_ci": post_ci,
            "delta": post_acc - pre_acc,
            "mcnemar_p": mc_p,
            "kind": "protocol",
        })

    # Pooled single-judge debate
    pooled = build_pooled_single_judge_debate(data)
    rows.append({
        "label": "Pooled single-judge debate\n(in-sub + multi-judge post-debate)",
        "n": len(pooled),
        "pre": pooled["pre_correct"].mean(),
        "pre_ci": _bootstrap_topic_ci_for_acc(pooled, "pre_correct"),
        "post": pooled["post_correct"].mean(),
        "post_ci": _bootstrap_topic_ci_for_acc(pooled, "post_correct"),
        "delta": pooled["post_correct"].mean() - pooled["pre_correct"].mean(),
        "mcnemar_p": _mcnemar_p(pooled),
        "kind": "pooled",
    })

    # MJH balanced (median across multiple seeds, or one representative)
    mjh = data[data["protocol"] == "multi_judge_h"].copy()
    rng = np.random.default_rng(42)
    n_seeds = 200
    balanced_results = []
    for seed in range(n_seeds):
        rrng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        sub = _balance_mjh(mjh, rrng)
        balanced_results.append({
            "n": len(sub),
            "pre": sub["pre_correct"].mean(),
            "post": sub["post_correct"].mean(),
            "mcnemar_p": _mcnemar_p(sub),
        })
    bal = pd.DataFrame(balanced_results)
    # 2.5/97.5% pct on pre and post across seeds for the CI
    rows.append({
        "label": "Multi-judge humans (balanced-pre)",
        "n": int(bal["n"].median()),
        "pre": bal["pre"].median(),
        "pre_ci": tuple(bal["pre"].quantile([0.025, 0.975])),
        "post": bal["post"].median(),
        "post_ci": tuple(bal["post"].quantile([0.025, 0.975])),
        "delta": bal["post"].median() - bal["pre"].median(),
        "mcnemar_p": bal["mcnemar_p"].median(),
        "kind": "balanced",
    })
    return rows


def collect_contrast_rows(data):
    """Build Panel-B forest-plot rows: cross-protocol Δ contrasts + CIs."""
    pooled = build_pooled_single_judge_debate(data)
    cons = data[data["protocol"] == "consultancy"]
    in_sub = data[data["protocol"] == "in_subtopic"]
    cross = data[data["protocol"] == "cross_subtopic"]
    mjh = data[data["protocol"] == "multi_judge_h"]
    hyb = data[data["protocol"] == "hybrid_mj"]

    def _two_arm_delta_ci(a, b):
        """Topic-clustered bootstrap CI for (acc_a − acc_b)."""
        a_id, b_id = "A", "B"
        merged = pd.concat([a.assign(__arm=a_id), b.assign(__arm=b_id)],
                           ignore_index=True)
        topics = merged["topic"].unique()
        out = np.empty(N_BOOT)
        for i in range(N_BOOT):
            sampled = RNG.choice(topics, size=len(topics), replace=True)
            boot = pd.concat([merged[merged["topic"] == t] for t in sampled],
                             ignore_index=True)
            ga = boot[boot["__arm"] == a_id]
            gb = boot[boot["__arm"] == b_id]
            if len(ga) == 0 or len(gb) == 0:
                out[i] = np.nan
            else:
                out[i] = ga["post_correct"].mean() - gb["post_correct"].mean()
        return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))

    def _fisher_p(a, b, alternative="two-sided"):
        ka, na = int(a["post_correct"].sum()), len(a)
        kb, nb = int(b["post_correct"].sum()), len(b)
        return stats.fisher_exact([[ka, na-ka],[kb, nb-kb]],
                                    alternative=alternative).pvalue

    contrasts = []
    contrasts.append({
        "label": "Pooled debate vs Consultancy",
        "delta": pooled["post_correct"].mean() - cons["post_correct"].mean(),
        "ci": _two_arm_delta_ci(pooled, cons),
        "p": _fisher_p(pooled, cons),
        "p_label": "Fisher 2-sided",
    })
    contrasts.append({
        "label": "MJH (full) vs Consultancy",
        "delta": mjh["post_correct"].mean() - cons["post_correct"].mean(),
        "ci": _two_arm_delta_ci(mjh, cons),
        "p": _fisher_p(mjh, cons),
        "p_label": "Fisher 2-sided",
    })
    contrasts.append({
        "label": "In-subtopic vs Consultancy",
        "delta": in_sub["post_correct"].mean() - cons["post_correct"].mean(),
        "ci": _two_arm_delta_ci(in_sub, cons),
        "p": _fisher_p(in_sub, cons),
        "p_label": "Fisher 2-sided",
    })
    contrasts.append({
        "label": "Cross-subtopic vs Consultancy",
        "delta": cross["post_correct"].mean() - cons["post_correct"].mean(),
        "ci": _two_arm_delta_ci(cross, cons),
        "p": _fisher_p(cross, cons),
        "p_label": "Fisher 2-sided",
    })
    contrasts.append({
        "label": "Hybrid MJ vs Consultancy",
        "delta": hyb["post_correct"].mean() - cons["post_correct"].mean(),
        "ci": _two_arm_delta_ci(hyb, cons),
        "p": _fisher_p(hyb, cons),
        "p_label": "Fisher 2-sided",
    })
    contrasts.append({
        "label": "MJH (full) vs In-subtopic",
        "delta": mjh["post_correct"].mean() - in_sub["post_correct"].mean(),
        "ci": _two_arm_delta_ci(mjh, in_sub),
        "p": _fisher_p(mjh, in_sub),
        "p_label": "Fisher 2-sided",
    })
    contrasts.append({
        "label": "In-subtopic vs Cross-subtopic",
        "delta": in_sub["post_correct"].mean() - cross["post_correct"].mean(),
        "ci": _two_arm_delta_ci(in_sub, cross),
        "p": _fisher_p(in_sub, cross),
        "p_label": "Fisher 2-sided",
    })
    return contrasts


# ── Plotting ──────────────────────────────────────────────────────

def sig_marker(p, thresholds=(0.05, 0.01, 0.001)):
    if pd.isna(p):
        return ""
    if p < thresholds[2]:
        return "***"
    if p < thresholds[1]:
        return "**"
    if p < thresholds[0]:
        return "*"
    return "n.s."


def plot_headline(rows):
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # (orig label in `rows`, display label, (light, dark) palette)
    panel_filter = [
        ("Pooled single-judge debate\n(in-sub + multi-judge post-debate)",
         "Debate-Subtopic",         ("#f4a3a3", "#d62728")),
        ("Cross-subtopic\n(post on different item)",
         "Cross-Subtopic", ("#ffd29a", "#ff7f0e")),
        ("Consultancy",
         "Consultancy",    ("#cccccc", "#555555")),
        ("Hybrid multi-judge",
         "Hybrid MJ",      ("#a6c8e0", "#1f77b4")),
        ("Multi-judge humans (balanced-pre)",
         "MJH",            ("#9ad59a", "#2ca02c")),
    ]
    rows_by_label = {r["label"]: r for r in rows}
    panel_filter = [(o, d, c) for o, d, c in panel_filter if o in rows_by_label]
    labels         = [o for o, _, _ in panel_filter]
    display_labels = [d for _, d, _ in panel_filter]
    palettes       = [c for _, _, c in panel_filter]

    x = np.arange(len(labels))
    bar_width = 0.36

    for i, label in enumerate(labels):
        r = rows_by_label[label]
        c_pre, c_post = palettes[i]
        pre_err = [[r["pre"] - r["pre_ci"][0]], [r["pre_ci"][1] - r["pre"]]]
        post_err = [[r["post"] - r["post_ci"][0]], [r["post_ci"][1] - r["post"]]]
        ax.bar(i - bar_width/2, r["pre"], bar_width,
               color=c_pre, edgecolor="black", linewidth=0.6,
               yerr=pre_err, capsize=4,
               error_kw={"lw": 1.0, "ecolor": "black"}, zorder=3)
        ax.bar(i + bar_width/2, r["post"], bar_width,
               color=c_post, edgecolor="black", linewidth=0.6,
               yerr=post_err, capsize=4,
               error_kw={"lw": 1.0, "ecolor": "black"}, zorder=3)

        # Annotation above the higher of the two CIs
        top = max(r["pre_ci"][1], r["post_ci"][1])
        sig = sig_marker(r["mcnemar_p"]) if not pd.isna(r["mcnemar_p"]) else ""
        delta_str = f"Δ={r['delta']*100:+.1f} pp"
        ax.text(i, top + 0.035, delta_str, ha="center", va="bottom",
                fontsize=13, color="#444444")
        if sig.startswith("*"):
            ax.text(i, top + 0.085, sig, ha="center", va="bottom",
                    fontsize=15, fontweight="bold", color="black")

    ax.axhline(0.5, color="#aaaaaa", linestyle="--", lw=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Protocol")
    ax.set_ylabel("Accuracy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "headline_results.png")
    out_pdf = os.path.join(OUT_DIR, "headline_results.pdf")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def run():
    data, *_ = build_data()
    print("Computing per-protocol rows (with topic-cluster bootstrap CIs)...")
    rows = collect_protocol_rows(data)
    print("Plotting...")
    out_png, out_pdf = plot_headline(rows)
    print(f"\nWrote:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    run()
