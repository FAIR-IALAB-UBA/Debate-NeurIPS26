"""
CANONICAL confidence-change analysis for the open-source debate experiments
(this repo: Qwen / Llama / gemma). Replaces the earlier version that treated the
~2340 normal+inverted per-claim rows as independent (pseudoreplication).

Question: when accuracy improves, does confidence worsen? (post-debate vs pre-debate)

Same clustering fix as the significance analysis: each claim is collapsed to ONE
unit (pre_conf is identical across the normal/inverted runs -- verified 100%; post
is averaged over the two runs). Tests are then run on independent claims, not on
the doubled normal+inverted rows.

  pre_conf_c  = pre_conf                         (same in both runs)
  post_conf_c = mean(post_conf over runs)
  dconf_c     = post_conf_c - pre_conf_c
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from load_results import RESULTS_DIR as ACC_DIR, load_evaluations

MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "gemma-3-4b-it"]
KEY = ["topic", "statement_1", "statement_2", "trait_key", "group", "condition"]
ALPHA = 0.05

osdata = {m: load_evaluations(m) for m in MODELS}
# C1 (unconditioned) is identical for mainstream/skeptical (no persona) -> count once,
# consistent with the significance analysis.
osdata = {m: df[~((df.condition == "C1") & (df.group == "skeptical"))].copy()
          for m, df in osdata.items()}


def collapse_claims(df):
    """One row per claim: pre/post accuracy and confidence averaged over the two runs."""
    d = df.copy()
    d["pre_c"] = d.pre_correct.astype(int)
    d["post_c"] = d.post_correct.astype(int)
    g = d.groupby(KEY, as_index=False).agg(
        pre_acc=("pre_c", "mean"), post_acc=("post_c", "mean"),
        pre_conf=("pre_conf", "mean"), post_conf=("post_conf", "mean"),
        runs=("post_c", "size"))
    g["dacc"] = g.post_acc - g.pre_acc
    g["dconf"] = g.post_conf - g.pre_conf
    return g


print("=" * 100)
print("CONFIDENCE CHANGE (clustered: 1 unit per claim, normal+inverted collapsed)")
print("=" * 100)

# ---- per-claim pooled, per model ----
all_claims = []
per_model = []
for m, df in osdata.items():
    g = collapse_claims(df)
    g["model"] = m
    all_claims.append(g)
    w = stats.wilcoxon(g.pre_conf, g.post_conf, zero_method="wilcox")
    per_model.append(dict(model=m, n_claims=len(g),
                          mean_dconf_pp=round(g.dconf.mean(), 2),
                          pct_conf_down=round((g.dconf < 0).mean() * 100),
                          wilcoxon_p=w.pvalue))
    print(f"\n[{m}] n_claims={len(g)}  mean dConf={g.dconf.mean():+.2f} pp "
          f"({(g.dconf<0).mean()*100:.0f}% lose confidence)  Wilcoxon p={w.pvalue:.2g}")
    # decomposition by per-claim accuracy outcome
    for name, mask in {
        "accuracy improved (dacc>0)": g.dacc > 0,
        "accuracy unchanged (dacc=0)": g.dacc == 0,
        "accuracy worsened (dacc<0)": g.dacc < 0,
    }.items():
        s = g[mask]
        if len(s):
            print(f"      {name:30} n={len(s):4d}  mean dConf={s.dconf.mean():+.2f} pp")

claims = pd.concat(all_claims, ignore_index=True)
pd.DataFrame(per_model).to_csv(os.path.join(ACC_DIR, "confidence_per_model.csv"), index=False)

# ---- pooled across all open-source claims ----
print("\n" + "-" * 100)
wins = stats.wilcoxon(claims.pre_conf, claims.post_conf, zero_method="wilcox")
t = stats.ttest_1samp(claims.dconf, 0)
print(f"POOLED (all open-source claims, n={len(claims)}): mean dConf={claims.dconf.mean():+.2f} pp, "
      f"{(claims.dconf<0).mean()*100:.0f}% lose confidence; Wilcoxon p={wins.pvalue:.2g}; "
      f"t-test p={t.pvalue:.2g}")

# ---- the question: does confidence worsen MORE when accuracy improves? ----
imp = claims[claims.dacc > 0]; wor = claims[claims.dacc < 0]
print(f"\nWhen accuracy IMPROVES (per claim, n={len(imp)}): mean dConf={imp.dconf.mean():+.2f} pp "
      f"({(imp.dconf<0).mean()*100:.0f}% lose confidence)")
print(f"When accuracy WORSENS (per claim, n={len(wor)}): mean dConf={wor.dconf.mean():+.2f} pp "
      f"({(wor.dconf<0).mean()*100:.0f}% lose confidence)")
mw = stats.mannwhitneyu(imp.dconf, wor.dconf, alternative="two-sided")
sp = stats.spearmanr(claims.dacc, claims.dconf)
print(f"dConf(improved) vs dConf(worsened): Mann-Whitney p={mw.pvalue:.2g}")
print(f"per-claim corr(dAcc, dConf): Spearman rho={sp.correlation:+.3f} (p={sp.pvalue:.2g})")

# ---- cell-level (1 obs per model x condition x group): the cross-experiment view ----
cell = claims.groupby(["model", "condition", "group"], as_index=False).agg(
    dacc_pp=("dacc", lambda x: x.mean() * 100), dconf_pp=("dconf", "mean"))
pe = stats.pearsonr(cell.dacc_pp, cell.dconf_pp)
print(f"\nCell-level (n={len(cell)} model x condition x group): "
      f"mean dConf={cell.dconf_pp.mean():+.2f} pp, "
      f"{(cell.dconf_pp<0).mean()*100:.0f}% of cells lose confidence; "
      f"corr(dAcc,dConf) Pearson r={pe[0]:+.3f} (p={pe[1]:.2g})")
cell.to_csv(os.path.join(ACC_DIR, "confidence_cells.csv"), index=False)

# summary row for the report / sheet
pd.DataFrame([dict(
    n_claims=len(claims), mean_dconf_pp=round(claims.dconf.mean(), 2),
    pct_claims_conf_down=round((claims.dconf < 0).mean() * 100),
    pooled_wilcoxon_p=wins.pvalue, pooled_ttest_p=t.pvalue,
    n_claims_acc_improved=len(imp), mean_dconf_acc_improved=round(imp.dconf.mean(), 2),
    pct_conf_down_acc_improved=round((imp.dconf < 0).mean() * 100),
    n_claims_acc_worsened=len(wor), mean_dconf_acc_worsened=round(wor.dconf.mean(), 2),
    pct_conf_down_acc_worsened=round((wor.dconf < 0).mean() * 100),
    mannwhitney_improved_vs_worsened_p=mw.pvalue,
    spearman_dacc_dconf=round(sp.correlation, 3), spearman_p=sp.pvalue,
    n_cells=len(cell), cell_pearson_r=round(pe[0], 3), cell_pearson_p=pe[1],
)]).to_csv(os.path.join(ACC_DIR, "confidence_summary.csv"), index=False)
print("\n-> results/confidence_per_model.csv, confidence_cells.csv, confidence_summary.csv")
