"""
CANONICAL significance analysis for the open-source debate experiments (this repo:
Qwen / Llama / gemma). Replaces the earlier naive analysis that pooled the normal +
inverted debates as independent observations.

WHY THE OLD WAY WAS WRONG. Each claim is judged ONCE pre-debate, then debated TWICE:
once with normal arguments and once with inverted/swapped arguments (the swap exists
only to cancel argument-position bias -- paper sec 3.2, "200 distinct debates"). The
pre-debate verdict is IDENTICAL across the two runs (verified 100%); only the post
verdict differs. So the two debates per claim are NOT independent. Pooling them as
~195 independent paired units per group (what the spreadsheet's McNemar did)
under-estimates the standard error -> inflated significance (pseudoreplication).

CORRECT analysis = cluster on the claim. Collapse each claim to ONE unit:
    pre_acc_c  = pre_correct                  (same in both runs)  in {0,1}
    post_acc_c = mean(post_correct over runs)                      in {0,.5,1}
    delta_c    = post_acc_c - pre_acc_c                            in {-1,-.5,0,.5,1}
and test H0: median(delta)=0. Primary test = exact-style paired PERMUTATION (sign
flip), robust to the heavy ties; reported alongside the sign test, Wilcoxon
signed-rank and a cluster bootstrap (all agree). n = number of claims (~99/group),
NOT ~195. Finally Holm-Bonferroni across the family of independent cells, matching
the paper's declared methodology (sec 4.2: paired tests + cluster-robust + Holm).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from load_results import RESULTS_DIR, load_evaluations

OUT_CSV = os.path.join(RESULTS_DIR, "significance_clustered.csv")
MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-3-4b-it"]
KEY = ["topic", "statement_1", "statement_2", "trait_key"]
ALPHA = 0.05
RNG = np.random.default_rng(0)
NBOOT = 10000

# row -> (cond, group_mode); row is the cell's row in the collaborators' results workbook, kept as an ID
os_blocks = {
 "Llama-3.1-8B-Instruct": {38:("C1","m"),
    41:("C2","m"),42:("C2","s"),43:("C2","b"), 50:("C3","m"),51:("C3","s"),52:("C3","b"),
    59:("C4","m"),60:("C4","s"),61:("C4","b"), 68:("C5","m"),69:("C5","s"),70:("C5","b"),
    77:("C6","m"),78:("C6","s"),79:("C6","b")},
 "Qwen2.5-7B-Instruct": {39:("C1","m"),
    44:("C2","m"),45:("C2","s"),46:("C2","b"), 53:("C3","m"),54:("C3","s"),55:("C3","b"),
    62:("C4","m"),63:("C4","s"),64:("C4","b"), 71:("C5","m"),72:("C5","s"),73:("C5","b"),
    80:("C6","m"),81:("C6","s"),82:("C6","b")},
 "gemma-3-4b-it": {40:("C1","m"),
    47:("C2","m"),48:("C2","s"),49:("C2","b"), 56:("C3","m"),57:("C3","s"),58:("C3","b"),
    65:("C4","m"),66:("C4","s"),67:("C4","b"), 74:("C5","m"),75:("C5","s"),76:("C5","b"),
    83:("C6","m"),84:("C6","s"),85:("C6","b")},
}
GROUPS = {"m": ["mainstream"], "s": ["skeptical"], "b": ["mainstream", "skeptical"]}

osdata = {m: load_evaluations(m) for m in MODELS}


def collapse(df, cond, groups):
    """One row per claim (cluster): pre (0/1) and post (mean over runs)."""
    sub = df[(df.condition == cond) & (df.group.isin(groups))].copy()
    sub["pre_i"] = sub.pre_correct.astype(int)
    sub["post_i"] = sub.post_correct.astype(int)
    g = sub.groupby(KEY + ["group"], as_index=False).agg(
        pre=("pre_i", "mean"), post=("post_i", "mean"), runs=("post_i", "size"))
    g["delta"] = g.post - g.pre
    return g


def old_independent_mcnemar(df, cond, groups):
    sub = df[(df.condition == cond) & (df.group.isin(groups))]
    pre = sub.pre_correct.astype(bool).to_numpy()
    post = sub.post_correct.astype(bool).to_numpy()
    b10 = int((pre & ~post).sum()); b01 = int((~pre & post).sum())
    diag = max(0, len(sub) - b10 - b01)
    p = float(mcnemar([[0, b10], [b01, diag]], exact=True).pvalue)
    return len(sub), b10, b01, p


def cluster_boot_p(delta):
    """2-sided bootstrap p for H0: mean(delta)=0, resampling claims (clusters)."""
    d = delta.to_numpy()
    n = len(d)
    obs = d.mean()
    centered = d - obs
    idx = RNG.integers(0, n, size=(NBOOT, n))
    boot = centered[idx].mean(axis=1)
    return float((np.sum(np.abs(boot) >= abs(obs)) + 1) / (NBOOT + 1))


def perm_sign_flip_p(delta):
    """Exact-style 2-sided paired permutation test: under H0 each claim's delta
    sign is equally likely +/-. Natural exact analog for the {-1,-.5,0,.5,1}
    paired data (no distributional assumption; robust to the heavy ties)."""
    d = delta[delta != 0].to_numpy()
    if d.size == 0:
        return 1.0
    obs = d.mean()
    signs = RNG.integers(0, 2, size=(NBOOT, d.size)) * 2 - 1
    perm = (signs * np.abs(d)).mean(axis=1)
    return float((np.sum(np.abs(perm) >= abs(obs)) + 1) / (NBOOT + 1))


rows = []
for model, blocks in os_blocks.items():
    df = osdata[model]
    for r, (cond, gm) in blocks.items():
        groups = GROUPS[gm]
        n_ind, b10, b01, p_old = old_independent_mcnemar(df, cond, groups)
        g = collapse(df, cond, groups)
        delta = g.delta
        nz = delta[delta != 0]
        # Wilcoxon signed-rank (primary)
        p_wil = float(stats.wilcoxon(g.post, g.pre, zero_method="wilcox").pvalue) if (nz.size >= 2) else np.nan
        # sign test (robustness): among non-zero deltas, are pos != neg?
        npos = int((delta > 0).sum()); nneg = int((delta < 0).sum())
        p_sign = float(stats.binomtest(npos, npos + nneg, 0.5).pvalue) if (npos + nneg) > 0 else np.nan
        p_perm = perm_sign_flip_p(delta)          # primary (robust to ties)
        p_boot = cluster_boot_p(delta)
        old_sig = p_old < ALPHA
        new_sig = (p_perm < ALPHA) if np.isfinite(p_perm) else False
        rows.append(dict(
            row=r, model=model, cond=cond, group=gm,
            n_claims=len(g), n_independent_pooled=n_ind,
            mean_delta_pp=round(delta.mean() * 100, 2),
            p_old_independent_mcnemar=p_old,
            p_new_perm=p_perm, p_new_sign=p_sign, p_new_wilcoxon=p_wil, p_new_clusterboot=p_boot,
            old="Significant" if old_sig else "n.s.",
            new="Significant" if new_sig else "n.s.",
            flips="<<< FLIPS" if old_sig != new_sig else ""))

res = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)

# Multiplicity: Holm-Bonferroni across the independent decision cells (exclude the
# 'both'/Average aggregates, which are not independent of their subgroups).
# Family = the 33 open-source cells = 3 models x [1 unconditioned + 5 conditions x 2 groups].
from statsmodels.stats.multitest import multipletests
res["p_clustered"] = res["p_new_perm"]                 # primary clustered test
res["p_holm"] = np.nan
cells = res.index[res.group != "b"]
rej, p_holm, _, _ = multipletests(res.loc[cells, "p_clustered"].values, method="holm")
res.loc[cells, "p_holm"] = p_holm
res["significant_clustered"] = res["p_clustered"] < ALPHA
res["significant_clustered_holm"] = res["p_holm"] < ALPHA   # NaN (averages) -> False
res["direction"] = np.where(res.mean_delta_pp > 0, "Improvement",
                     np.where(res.mean_delta_pp < 0, "Deterioration", "No change"))
res.to_csv(OUT_CSV, index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
show = res[["row", "model", "cond", "group", "n_claims", "n_independent_pooled",
            "mean_delta_pp", "p_old_independent_mcnemar", "p_new_perm",
            "p_new_wilcoxon", "p_new_clusterboot", "old", "new", "flips"]]
print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

nflip = (res.old != res.new).sum()
lost = ((res.old == "Significant") & (res.new == "n.s.")).sum()
gained = ((res.old == "n.s.") & (res.new == "Significant")).sum()
print(f"\nRows: {len(res)}  |  primary test = permutation (sign-flip)."
      f"  Verdict changes under clustering: {nflip}"
      f"  (lost significance: {lost}, gained: {gained})")
# robustness: how many flip under EACH clustered test
for col in ["p_new_perm", "p_new_sign", "p_new_wilcoxon", "p_new_clusterboot"]:
    lo = ((res.p_old_independent_mcnemar < ALPHA) & (res[col] >= ALPHA)).sum()
    print(f"    lost-significance count under {col:18}: {lo}")
# headline counts on the 33 independent decision cells
cells_df = res[res.group != "b"]
print(f"\nIndependent decision cells (n={len(cells_df)}):  significant @0.05  ->  "
      f"old pooled McNemar: {(cells_df.p_old_independent_mcnemar<ALPHA).sum()}  |  "
      f"clustered: {cells_df.significant_clustered.sum()}  |  "
      f"clustered+Holm: {cells_df.significant_clustered_holm.sum()}")
print("-> results/significance_clustered.csv")
