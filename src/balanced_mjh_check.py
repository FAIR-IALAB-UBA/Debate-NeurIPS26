#!/usr/bin/env python3
"""Robustness check: re-run the multi-judge-humans (MJH) headline tests on a
random subsample where the *pre*-belief accuracy starts at ~50%, matching the
other protocols.

Why this matters
----------------
MJH's full-sample pre-accuracy is 56.7% (80 / 141). The other four protocols
all start near 50% (cross 50.6, consultancy 51.2, hybrid 50.5, in-subtopic
41.3 — the only one off the other way). Comparing a 66.0% post against a
~57% baseline of comparable protocols is unfair to the other protocols, since
MJH had a head start on the easy direction.

What this script does
---------------------
For each random seed (default 1000):
1. Identify all MJH rooms (debate-rooms = LimeSurvey survey IDs) where every
   recruited participant was pre-correct (i.e. room-level pre-accuracy = 1.0).
2. Randomly drop a subset of those rooms — entire rooms, not individuals,
   per the request — until the remaining MJH sub-sample's pre-accuracy is
   within ±2 pp of 50%.
3. Re-run the MJH headline tests on the balanced sub-sample:
     - binomial post-acc vs 50%
     - McNemar pre→post-deliberation
     - Fisher's exact MJH-vs-rest (using the OTHER protocols as the baseline)
     - Fisher's exact MJH-vs-consultancy
4. Across all seeds, report the distribution of n, pre-acc, post-acc, Δ, and
   each test's p-value (median + 5th/95th percentiles).

Reading the output
------------------
If MJH's headline result is mostly an artifact of the inflated starting line,
the balanced-subsample p-values will move away from significance. If it's
robust, they'll stay significant in most seeds. We treat 'most seeds
significant' as a reasonable robustness standard.

This script does NOT modify any other claims; it's only relevant to MJH.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_full import build_data

N_SEEDS = 1000
TARGET_PRE_ACC = 0.50
TARGET_TOL = 0.02       # accept any sub-sample with |pre_acc − 0.5| ≤ 0.02


def balance_one_seed(mjh, rng):
    """Drop random all-pre-correct rooms until the sub-sample's pre-accuracy
    falls within ±TARGET_TOL of TARGET_PRE_ACC, or we run out of all-correct
    rooms to drop. Return the trimmed dataframe."""
    rooms = mjh.groupby("al_survey_id")["pre_correct"]
    all_correct = sorted(s for s, vals in rooms if (vals == 1).all())
    rng.shuffle(all_correct)

    sub = mjh.copy()
    pa = sub["pre_correct"].mean()
    # Iterate room-by-room: drop a room only if it improves the imbalance
    # without overshooting too far below 50%.
    for room in all_correct:
        if abs(pa - TARGET_PRE_ACC) <= TARGET_TOL:
            break
        if pa <= TARGET_PRE_ACC:
            break
        candidate = sub[sub["al_survey_id"] != room]
        new_pa = candidate["pre_correct"].mean()
        # Only drop the room if new_pa is closer to 0.5 than current pa,
        # OR still ≥ 0.5 (to avoid undershoot when one big room would push
        # us far below 0.5).
        if abs(new_pa - TARGET_PRE_ACC) < abs(pa - TARGET_PRE_ACC):
            sub = candidate
            pa = new_pa
    return sub


def run():
    data, *_ = build_data()
    mjh = data[data["protocol"] == "multi_judge_h"].copy()
    rest = data[data["protocol"] != "multi_judge_h"].copy()
    cons = data[data["protocol"] == "consultancy"].copy()

    n_full = len(mjh)
    pre_full = mjh["pre_correct"].mean()
    post_full = mjh["post_correct"].mean()
    print("=" * 64)
    print("  BALANCED-PRE MJH ROBUSTNESS CHECK")
    print("=" * 64)
    print(f"  Full MJH:  n={n_full}  pre={pre_full:.3f}  post={post_full:.3f}")
    print(f"  Target balanced sub-sample: pre-acc within ±{TARGET_TOL} of "
          f"{TARGET_PRE_ACC}")
    print(f"  Strategy: randomly drop entire all-pre-correct rooms until")
    print(f"  the sub-sample's pre-accuracy hits the target band.")
    print(f"  Seeds: {N_SEEDS}")
    print()

    rng_master = np.random.default_rng(42)
    results = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        sub = balance_one_seed(mjh, rng)
        n_sub = len(sub)
        pre_sub = sub["pre_correct"].mean()
        post_sub = sub["post_correct"].mean()

        # binomial post vs 50%
        k = int(sub["post_correct"].sum())
        p_binom = stats.binomtest(k, n_sub, 0.5,
                                   alternative="greater").pvalue

        # McNemar pre→post-deliberation
        b = int(((sub["pre_correct"] == 1) & (sub["post_correct"] == 0)).sum())
        c = int(((sub["pre_correct"] == 0) & (sub["post_correct"] == 1)).sum())
        if b + c > 0:
            p_mcnemar = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        else:
            p_mcnemar = np.nan

        # Fisher's exact MJH-vs-rest (one-sided MJH > rest)
        rest_k = int(rest["post_correct"].sum())
        rest_n = len(rest)
        p_fisher_rest = stats.fisher_exact(
            [[k, n_sub - k], [rest_k, rest_n - rest_k]],
            alternative="greater").pvalue

        # Fisher's exact MJH-vs-consultancy (one-sided MJH > cons)
        cons_k = int(cons["post_correct"].sum())
        cons_n = len(cons)
        p_fisher_cons = stats.fisher_exact(
            [[k, n_sub - k], [cons_k, cons_n - cons_k]],
            alternative="greater").pvalue

        results.append({
            "n": n_sub, "pre": pre_sub, "post": post_sub, "delta": post_sub-pre_sub,
            "b": b, "c": c,
            "p_binom_post>50": p_binom,
            "p_mcnemar": p_mcnemar,
            "p_fisher_vs_rest": p_fisher_rest,
            "p_fisher_vs_cons": p_fisher_cons,
        })

    df = pd.DataFrame(results)

    def pct(s, q):
        return np.percentile(s, q)

    print(f"  {'metric':<28s}  {'median':>10s}  {'5th-pct':>10s}  {'95th-pct':>10s}")
    for col in ["n", "pre", "post", "delta",
                "p_binom_post>50", "p_mcnemar",
                "p_fisher_vs_rest", "p_fisher_vs_cons"]:
        s = df[col].dropna()
        med = pct(s, 50)
        lo = pct(s, 5)
        hi = pct(s, 95)
        if col in ("n", "b", "c"):
            print(f"  {col:<28s}  {med:>10.0f}  {lo:>10.0f}  {hi:>10.0f}")
        else:
            print(f"  {col:<28s}  {med:>10.4f}  {lo:>10.4f}  {hi:>10.4f}")

    print()
    print(f"  Fraction of seeds with each p < 0.05:")
    for col in ["p_binom_post>50", "p_mcnemar",
                "p_fisher_vs_rest", "p_fisher_vs_cons"]:
        frac = (df[col] < 0.05).mean()
        print(f"    {col:<28s}  {frac:.1%}")

    print()
    print("Headline-significance check:")
    print("  A claim survives this robustness check if the test stays below")
    print("  α=0.05 in 'most' (≥80% by convention) of the seeds. Treat the")
    print("  fraction column above as the relevant evidence.")

    print()
    print("Comparison to full sample (un-balanced):")
    n_full = len(mjh)
    k_full = int(mjh["post_correct"].sum())
    b_full = int(((mjh["pre_correct"] == 1) & (mjh["post_correct"] == 0)).sum())
    c_full = int(((mjh["pre_correct"] == 0) & (mjh["post_correct"] == 1)).sum())
    p_binom_full = stats.binomtest(k_full, n_full, 0.5,
                                     alternative="greater").pvalue
    p_mc_full = stats.binomtest(min(b_full, c_full), b_full + c_full, 0.5).pvalue
    rest_k = int(rest["post_correct"].sum()); rest_n = len(rest)
    cons_k = int(cons["post_correct"].sum()); cons_n = len(cons)
    p_fr_full = stats.fisher_exact(
        [[k_full, n_full - k_full], [rest_k, rest_n - rest_k]],
        alternative="greater").pvalue
    p_fc_full = stats.fisher_exact(
        [[k_full, n_full - k_full], [cons_k, cons_n - cons_k]],
        alternative="greater").pvalue
    print(f"  full MJH: n={n_full}, pre={mjh['pre_correct'].mean():.3f}, "
          f"post={mjh['post_correct'].mean():.3f}")
    print(f"  full MJH p-values: binom={p_binom_full:.4f} "
          f"mcnemar={p_mc_full:.4f} fisher_vs_rest={p_fr_full:.4f} "
          f"fisher_vs_cons={p_fc_full:.4f}")


if __name__ == "__main__":
    run()
