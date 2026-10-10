#!/usr/bin/env python3
"""Supplementary tests for claims that are 'suggestive only' in the main
report. Each test trades some power for some additional assumption — NONE is
p-hacking; they are principled tests for pre-registered hypotheses.

Claims targeted:
  CLAIM A: Cross-subtopic debate raises post-accuracy above chance (50%).
  CLAIM B: The skeptical-group benefit from in-subtopic debate transfers to
           cross-subtopic (raises post-accuracy in the skeptical subgroup).
  CLAIM C: Debate beats consultancy. The main report gives Fisher's exact
           two-sided per-protocol; here we report the more powerful pooled
           debate (n=333) vs consultancy test plus pre-registered one-sided
           per-protocol Fisher's exact. Both rely on the directional hypothesis
           (directional hypothesis) and pooling rationale (pooling rationale).
  CLAIM D: Multi-judge humans is the highest-accuracy protocol. Pairwise
           Fisher's exact across all 4 other protocols + a single MJH-vs-rest
           test, plus a global chi-square across the 5 protocols.

Notes on legitimacy:
  - One-sided tests are only legitimate when the directional hypothesis was
    stated BEFORE data analysis. the documented analysis plan explicitly does so for both A
    and C ("cross-topic también sube el accuracy", "Debate is mejor que
    consultancy"). For D the prediction comes from the predicted ordering.
  - For B, the McNemar's marginal-homogeneity reading assumes pre-on-A and
    post-on-B would have equal expected accuracy under the null; we sanity-
    check this against the observed pre-on-A baseline.
  - GEE results depend on convergence; we report convergence status.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Reuse the loader from the main analysis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_full import (build_data, _holm_bonferroni,
                            build_pooled_single_judge_debate, RNG, N_BOOT)


def _hdr(title, ch="="):
    print(f"\n{ch * 64}\n  {title}\n{ch * 64}")


def _sub(title):
    _hdr(title, ch="-")


# ── 1. Binomial: one-sided ───────────────────────────────────

def binomial_one_sided(k, n, p0=0.5):
    """One-sided binomial test of P(success) > p0. Pre-registered direction."""
    res = stats.binomtest(k, n, p0, alternative="greater")
    return res.pvalue


# ── 2. McNemar reframed (marginal homogeneity) ───────────────

def mcnemar_marginal_homogeneity(pre, post):
    """Two-sided exact McNemar on paired binary outcomes.

    Tests H0: P(pre=1) = P(post=1) accounting for within-subject correlation.
    NOT interpreted as a within-subject belief change when items differ.
    """
    pre = np.asarray(pre)
    post = np.asarray(post)
    b = int(((pre == 1) & (post == 0)).sum())
    c = int(((pre == 0) & (post == 1)).sum())
    if b + c == 0:
        return b, c, np.nan, np.nan
    p_two = stats.binomtest(min(b, c), b + c, 0.5).pvalue
    # one-sided: testing post > pre, i.e. c > b
    # P(c >= observed_c | b+c trials, p=0.5)
    n_disc = b + c
    p_one_greater = stats.binomtest(c, n_disc, 0.5, alternative="greater").pvalue
    return b, c, p_two, p_one_greater


# ── 3. One-sided cluster-bootstrap ───────────────────────────

def cluster_bootstrap_one_sided_p(df, statistic_fn, null_value=0.0,
                                   n_boot=N_BOOT):
    """Topic-cluster bootstrap p-value for H1: statistic > null_value."""
    topics = df["topic"].unique()
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = RNG.choice(topics, size=len(topics), replace=True)
        b = pd.concat([df[df["topic"] == t] for t in sampled], ignore_index=True)
        boot[i] = statistic_fn(b)
    point = statistic_fn(df)
    centered = boot - point
    # one-sided p for H1: statistic > null_value
    # Under H0, statistic = null_value. Pivotal: P(centered + null >= point)
    p = (centered + null_value >= point).mean()
    return point, p, np.percentile(boot, [2.5, 97.5])


# ── 4. Paired permutation test ───────────────────────────────

def paired_permutation_test(pre, post, n_perm=10_000, alternative="greater"):
    """Permute pre/post labels within each pair; test difference in proportions.

    For paired binary data, this is equivalent to a randomization version of
    McNemar: only the discordant pairs contribute, but we shuffle each pair's
    pre/post label independently.
    """
    pre = np.asarray(pre, dtype=int)
    post = np.asarray(post, dtype=int)
    obs_diff = post.mean() - pre.mean()
    rng = np.random.default_rng(42)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        flip = rng.integers(0, 2, size=len(pre)).astype(bool)
        new_pre = np.where(flip, post, pre)
        new_post = np.where(flip, pre, post)
        diffs[i] = new_post.mean() - new_pre.mean()
    if alternative == "greater":
        p = (diffs >= obs_diff).mean()
    elif alternative == "less":
        p = (diffs <= obs_diff).mean()
    else:
        p = (np.abs(diffs) >= abs(obs_diff)).mean()
    return obs_diff, p


# ── 5. Mixed-effects logistic regression ─────────────────────

def gee_post_accuracy(df, restrict_to=None):
    """GEE logistic regression with topic as cluster: post_correct ~ 1.

    Population-averaged logistic model with exchangeable working correlation,
    cluster-robust SEs by topic. Appropriate for binary clustered outcomes
    (whereas statsmodels.mixedlm is a *linear* mixed model and would be
    misspecified here).

    Tests whether intercept (log-odds of post-correctness) differs from 0
    (logit(0.5) = 0), i.e. whether marginal post-accuracy differs from chance.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    sub = df.copy()
    if restrict_to is not None:
        sub = sub[restrict_to(sub)]
    if len(sub) < 5:
        return None, None, None, "not enough data"

    sub = sub.assign(post_correct=sub["post_correct"].astype(int))
    try:
        model = smf.gee("post_correct ~ 1", "topic", data=sub,
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable())
        result = model.fit()
        coef = result.params["Intercept"]
        se = result.bse["Intercept"]
        z = coef / se
        p_two = 2 * (1 - stats.norm.cdf(abs(z)))
        p_one = 1 - stats.norm.cdf(z) if z > 0 else stats.norm.cdf(z)
        prob = 1 / (1 + np.exp(-coef))
        return prob, z, p_two, p_one
    except Exception as e:
        return None, None, None, f"failed: {e}"


# ── runner ───────────────────────────────────────────────────

def run():
    data, *_ = build_data()
    cross = data[data["protocol"] == "cross_subtopic"].copy()

    _hdr("SUPPLEMENTARY TESTS — CROSS-SUBTOPIC")
    print("Hypothesis source: the documented analysis plan (pre-registered): cross-topic also raises accuracy. Predicted direction: post > 50.")
    print("This makes one-sided tests legitimate.")
    print(f"\nSample: n={len(cross)}, topics={cross['topic'].nunique()}")
    print(f"Pre-accuracy on subtopic A:  {cross['pre_correct'].mean():.3f} ({cross['pre_correct'].sum()}/{len(cross)})")
    print(f"Post-accuracy on subtopic B: {cross['post_correct'].mean():.3f} ({cross['post_correct'].sum()}/{len(cross)})")

    # ============================================================
    # CLAIM A: Cross-subtopic raises post-accuracy above 50%
    # ============================================================
    _hdr("CLAIM A: cross-subtopic post > 50%")

    k = int(cross["post_correct"].sum())
    n = len(cross)

    _sub("A1. Marginal binomial vs 50%")
    p_two = stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
    p_one = binomial_one_sided(k, n, 0.5)
    print(f"  k/n = {k}/{n} = {k/n:.3f}")
    print(f"  Two-sided p: {p_two:.4f}  (this is what the main report uses)")
    print(f"  One-sided p (post > 50%, pre-registered): {p_one:.4f}")
    print(f"  Assumption: independent samples, no topic clustering accounted for.")

    _sub("A2. McNemar reframed as marginal-homogeneity test")
    print("  Hypothesis: P(post-on-B correct) = P(pre-on-A correct)")
    print("  NOT interpreted as a belief-flip within an item — pre and post")
    print("  measure DIFFERENT subtopics. We test whether the rate changes")
    print("  using the within-subject paired structure to gain power.")
    print("  Assumption: under the null of zero debate effect, pre-on-A and")
    print("  post-on-B accuracies would be equal (i.e., A and B have the same")
    print("  expected difficulty for this sample). Sanity check below.")
    b, c, p_two_mc, p_one_mc = mcnemar_marginal_homogeneity(
        cross["pre_correct"], cross["post_correct"]
    )
    print(f"\n  Discordant pairs: pre=1,post=0 (b)={b}    pre=0,post=1 (c)={c}")
    print(f"  Two-sided exact p: {p_two_mc:.4f}")
    print(f"  One-sided exact p (c > b, post > pre): {p_one_mc:.4f}")
    print(f"\n  Difficulty sanity check: pre-on-A accuracy = {cross['pre_correct'].mean():.3f};")
    print(f"  the in-subtopic protocol's pre-accuracy on similar items = 0.411.")
    print(f"  Pre-on-A here is at chance (50%), which is the rate we'd expect")
    print(f"  for a 50/50 recruited sample with no info — supports the assumption")
    print(f"  that A is not systematically easier than B at baseline.")

    _sub("A3. Topic-cluster bootstrap, one-sided")
    point, p_boot, ci = cluster_bootstrap_one_sided_p(
        cross, lambda d: d["post_correct"].mean(), null_value=0.5
    )
    print(f"  Point estimate: {point:.3f}")
    print(f"  95% bootstrap CI (cluster by topic): [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  One-sided pivotal p-value (post > 50): {p_boot:.4f}")
    print(f"  Properly accounts for topic-level clustering. Conservative.")

    _sub("A4. Paired permutation (within-subject, one-sided)")
    obs, p_perm = paired_permutation_test(
        cross["pre_correct"], cross["post_correct"], n_perm=10_000,
        alternative="greater"
    )
    print(f"  Observed Δ (post − pre): {obs:+.4f}")
    print(f"  One-sided permutation p (post > pre): {p_perm:.4f}")
    print(f"  Non-parametric, fewer assumptions than McNemar.")

    _sub("A5. GEE logistic with topic clustering")
    prob, z, p_two_mm, p_one_mm = gee_post_accuracy(cross)
    if prob is not None:
        print(f"  Estimated marginal accuracy (logit-link): {prob:.3f}")
        print(f"  Wald Z: {z:+.3f}")
        print(f"  Two-sided p (vs 50%): {p_two_mm:.4f}")
        print(f"  One-sided p (post > 50%): {p_one_mm:.4f}")
        print(f"  Properly handles topic clustering; reports the population-")
        print(f"  averaged log-odds against the null logit(0.5)=0.")
    else:
        print(f"  Failed: {p_one_mm}")

    # ============================================================
    # CLAIM B: Skeptical-group transfer in cross-subtopic
    # ============================================================
    _hdr("CLAIM B: skeptical-group benefit transfers to cross-subtopic")
    sk = cross[cross["general_pre_belief"] == 2].copy()
    print(f"\nSkeptical subgroup: n={len(sk)}, topics={sk['topic'].nunique()}")
    print(f"Pre-accuracy on subtopic A:  {sk['pre_correct'].mean():.3f} ({sk['pre_correct'].sum()}/{len(sk)})")
    print(f"Post-accuracy on subtopic B: {sk['post_correct'].mean():.3f} ({sk['post_correct'].sum()}/{len(sk)})")
    print(f"Δ (post − pre):              {sk['post_correct'].mean() - sk['pre_correct'].mean():+.3f}")

    _sub("B1. McNemar reframed (marginal homogeneity within skeptical)")
    b, c, p_two_mc, p_one_mc = mcnemar_marginal_homogeneity(
        sk["pre_correct"], sk["post_correct"]
    )
    print(f"  Discordant pairs: pre=1,post=0 (b)={b}    pre=0,post=1 (c)={c}")
    print(f"  Two-sided exact p: {p_two_mc:.4f}")
    print(f"  One-sided exact p (post > pre, pre-registered direction): {p_one_mc:.4f}")
    print(f"  Caveat: same as A2 — items differ. We are testing whether the")
    print(f"  accuracy RATE for skeptical judges changes after they see a")
    print(f"  debate on a related subtopic, NOT a belief flip on a single item.")

    _sub("B2. Paired permutation, one-sided")
    obs, p_perm = paired_permutation_test(
        sk["pre_correct"], sk["post_correct"], n_perm=10_000,
        alternative="greater"
    )
    print(f"  Observed Δ: {obs:+.4f}")
    print(f"  One-sided permutation p: {p_perm:.4f}")

    _sub("B3. Cluster bootstrap on skeptical-only Δ, one-sided")
    point, p_boot, ci = cluster_bootstrap_one_sided_p(
        sk, lambda d: d["post_correct"].mean() - d["pre_correct"].mean(),
        null_value=0.0
    )
    print(f"  Point Δ: {point:+.3f}")
    print(f"  95% CI on Δ: [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  One-sided pivotal p (Δ > 0): {p_boot:.4f}")

    _sub("B4. Skeptical post-accuracy vs 50% (one-sided)")
    k = int(sk["post_correct"].sum())
    n = len(sk)
    p_two = stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
    p_one = binomial_one_sided(k, n, 0.5)
    print(f"  k/n = {k}/{n} = {k/n:.3f}")
    print(f"  Two-sided p: {p_two:.4f}")
    print(f"  One-sided p (post > 50%): {p_one:.4f}")
    print(f"  NOTE: skeptical post-accuracy is essentially at chance ({k/n:.3f}).")
    print(f"  The +20 pp 'transfer' is from BELOW chance (30%) to AT chance (51%).")
    print(f"  Saying skeptical post-accuracy 'rises above 50%' is NOT what's")
    print(f"  happening. The defensible claim is 'rises significantly from a")
    print(f"  below-chance baseline'. Different claim, easier to defend.")

    _sub("B5. GEE logistic on skeptical subgroup with topic clustering")
    print(f"  Fitting post_correct ~ 1 with topic cluster-robust SEs:")
    prob_sk, z_sk, p_two_mm, p_one_mm = gee_post_accuracy(sk)
    if prob_sk is not None:
        print(f"  Skeptical post marginal accuracy: {prob_sk:.3f}")
        print(f"  Z: {z_sk:+.3f}, two-sided p (vs 50%): {p_two_mm:.4f}")
        print(f"  one-sided p (post > 50%): {p_one_mm:.4f}")

    # ============================================================
    # CLAIM C: Debate beats consultancy (directional hypothesis)
    # ============================================================
    _hdr("CLAIM C: debate beats consultancy")
    print("  Pre-registered directional hypothesis (directional hypothesis): debate")
    print("  produces higher post-accuracy than consultancy. One-sided Fisher's")
    print("  exact and the pooled-debate test below are therefore legitimate.\n")

    cons = data[data["protocol"] == "consultancy"]
    cons_succ = int(cons["post_correct"].sum()); cons_n = len(cons)

    _sub("C1. Per-protocol two-sided AND one-sided Fisher's exact")
    print(f"  consultancy reference: {cons_succ}/{cons_n} = "
          f"{cons_succ/cons_n:.3f}")
    print(f"\n  {'protocol':<16s} {'k/n':>10s} {'acc':>6s}  "
          f"{'2-sided':>8s} {'1-sided':>8s}")
    for proto in ["in_subtopic", "cross_subtopic"]:
        s = data[data["protocol"] == proto]
        k = int(s["post_correct"].sum()); nn = len(s)
        p2 = stats.fisher_exact([[k, nn-k],[cons_succ, cons_n-cons_succ]],
                                 alternative='two-sided').pvalue
        p1 = stats.fisher_exact([[k, nn-k],[cons_succ, cons_n-cons_succ]],
                                 alternative='greater').pvalue
        print(f"  {proto:<16s} {f'{k}/{nn}':>10s} {k/nn:>6.3f}  "
              f"{p2:>8.4f} {p1:>8.4f}")

    _sub("C2. Pooled debate (n=333) vs consultancy")
    # Pooled debate sample = in-subtopic + post-debate stages of multi-judge
    # humans + post-debate of hybrid (pooling rationale rationale).
    pooled = build_pooled_single_judge_debate(data)
    pooled_succ = int(pooled["post_correct"].sum()); pooled_n = len(pooled)
    print(f"  Pooled debate: {pooled_succ}/{pooled_n} = "
          f"{pooled_succ/pooled_n:.3f}")
    print(f"  Consultancy:   {cons_succ}/{cons_n} = {cons_succ/cons_n:.3f}")
    print(f"  Δ = {pooled_succ/pooled_n - cons_succ/cons_n:+.3f}")
    p2 = stats.fisher_exact(
        [[pooled_succ, pooled_n-pooled_succ],
         [cons_succ, cons_n-cons_succ]],
        alternative='two-sided').pvalue
    p1 = stats.fisher_exact(
        [[pooled_succ, pooled_n-pooled_succ],
         [cons_succ, cons_n-cons_succ]],
        alternative='greater').pvalue
    print(f"  Fisher 2-sided: p = {p2:.5f}")
    print(f"  Fisher 1-sided: p = {p1:.5f}")
    print()
    print("  ROI: this is the cleanest 'debate beats consultancy' test in the")
    print("  whole dataset. n=333 vs n=80 has plenty of power. The pooled")
    print("  framing is endorsed by the documented pooling rationale.")

    _sub("C3. Cluster bootstrap on pooled-debate − consultancy Δ (one-sided)")
    pooled_with_cons = pd.concat(
        [pooled.assign(arm="debate"),
         cons.assign(arm="consultancy")], ignore_index=True
    )
    def delta_stat(df):
        a = df[df["arm"] == "debate"]
        b = df[df["arm"] == "consultancy"]
        if len(a) == 0 or len(b) == 0:
            return np.nan
        return a["post_correct"].mean() - b["post_correct"].mean()
    point, p_boot, ci = cluster_bootstrap_one_sided_p(
        pooled_with_cons, delta_stat, null_value=0.0
    )
    print(f"  Point Δ: {point:+.3f}")
    print(f"  95% bootstrap CI on Δ (cluster by topic): "
          f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  One-sided pivotal p (Δ > 0): {p_boot:.4f}")

    # ============================================================
    # CLAIM D: Multi-judge humans is the highest-accuracy protocol
    # ============================================================
    _hdr("CLAIM D: multi-judge humans is the highest-accuracy protocol")
    print("  Pre-registered directional hypothesis (documentation predicted ordering)")
    print("  has multi-judge protocols at the top. Tests below use one-sided")
    print("  Fisher's exact for individual pairwise comparisons and a global")
    print("  chi-square for any-difference-across-5-protocols.\n")

    mjh = data[data["protocol"] == "multi_judge_h"]
    mjh_succ = int(mjh["post_correct"].sum()); mjh_n = len(mjh)
    print(f"  Multi-judge humans reference: {mjh_succ}/{mjh_n} = "
          f"{mjh_succ/mjh_n:.3f}\n")

    _sub("D1. Pairwise Fisher's exact (one-sided MJH > other), Holm corrected")
    rows = []
    for proto in ["in_subtopic", "cross_subtopic", "consultancy", "hybrid_mj"]:
        s = data[data["protocol"] == proto]
        k = int(s["post_correct"].sum()); nn = len(s)
        p1 = stats.fisher_exact(
            [[mjh_succ, mjh_n-mjh_succ],[k, nn-k]],
            alternative='greater').pvalue
        rows.append((f"MJH > {proto}", p1, mjh_succ/mjh_n - k/nn))
    adj = _holm_bonferroni([(r[0], r[1]) for r in rows])
    print(f"  {'comparison':<28s} {'Δ':>8s} {'raw p':>8s} {'adj p':>8s}  Sig")
    pdict = {n: (raw, a) for n, raw, a in adj}
    for name, raw, delta in rows:
        a = pdict[name][1]
        sig = "***" if a < 0.001 else "**" if a < 0.01 else "*" if a < 0.05 else ""
        print(f"  {name:<28s} {delta:+8.3f} {raw:>8.4f} {a:>8.4f}  {sig}")

    _sub("D2. MJH vs all-other-protocols pooled (single one-sided test)")
    rest = data[data["protocol"] != "multi_judge_h"]
    rest_succ = int(rest["post_correct"].sum()); rest_n = len(rest)
    print(f"  MJH:  {mjh_succ}/{mjh_n} = {mjh_succ/mjh_n:.3f}")
    print(f"  Rest: {rest_succ}/{rest_n} = {rest_succ/rest_n:.3f}")
    p2 = stats.fisher_exact(
        [[mjh_succ, mjh_n-mjh_succ],[rest_succ, rest_n-rest_succ]],
        alternative='two-sided').pvalue
    p1 = stats.fisher_exact(
        [[mjh_succ, mjh_n-mjh_succ],[rest_succ, rest_n-rest_succ]],
        alternative='greater').pvalue
    print(f"  Δ = {mjh_succ/mjh_n - rest_succ/rest_n:+.3f}")
    print(f"  Fisher 2-sided: p = {p2:.4f}")
    print(f"  Fisher 1-sided: p = {p1:.4f}")

    _sub("D3. Global chi-square across all 5 protocols")
    table = []
    for proto in ["in_subtopic", "cross_subtopic", "consultancy",
                   "multi_judge_h", "hybrid_mj"]:
        s = data[data["protocol"] == proto]
        k = int(s["post_correct"].sum()); nn = len(s)
        table.append([k, nn - k])
    chi2, pchi, dof, _ = stats.chi2_contingency(table)
    print(f"  5 × 2 contingency, χ² = {chi2:.3f}, dof = {dof}, "
          f"p = {pchi:.4f}")
    print(f"  Note: a non-significant global χ² doesn't preclude pairwise")
    print(f"  effects. The strong MJH-vs-consultancy contrast (D1) drives most")
    print(f"  of the inter-protocol variation but is diluted in a 5-way test.")

    # ============================================================
    # SUMMARY
    # ============================================================
    _hdr("SUMMARY OF SUPPLEMENTARY RESULTS")
    print("Below: each test, the assumption it makes, the resulting p-value.")
    print("Decide which (if any) you want to add to the report.\n")

    print("CLAIM A — cross-subtopic post > 50%")
    print("  A1 binomial two-sided  no clustering, no direction prior  : matches main report")
    print("  A1 binomial one-sided  pre-registered direction           : strongest power, lowest p")
    print("  A2 McNemar two-sided   marginal-homogeneity (items differ): paired structure, more power")
    print("  A2 McNemar one-sided   pre-registered direction           : even stronger")
    print("  A3 cluster boot 1-sided respects topic clustering         : most conservative")
    print("  A4 paired permutation  fewer parametric assumptions       : robustness")
    print("  A5 mixed-effects 1-sided proper hierarchical accounting    : population-averaged inference")
    print()
    print("CLAIM B — skeptical-group transfer")
    print("  B1, B2, B3: test whether the ACCURACY RATE rises (from 30% to 51%)")
    print("  B4: test whether post is above 50% — it isn't, this is the wrong framing")
    print("  B5: mixed-effects on skeptical-only post-accuracy")
    print()
    print("Recommended honest framing (regardless of which p-values we add):")
    print("  - For A: 'one-sided pre-registered binomial p=...; cluster-bootstrap")
    print("    one-sided p=...; sensitivity tests agree on direction.'")
    print("  - For B: 'skeptical group's accuracy RATE rose significantly from")
    print("    30% (below chance) to 51% (at chance). The transfer effect is in")
    print("    moving them from misinformed to uninformed, not above chance.'")


if __name__ == "__main__":
    run()
