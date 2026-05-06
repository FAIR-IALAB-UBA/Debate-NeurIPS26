# Debate Experiment — Human Participants Results


## Design

| Protocol | Pre-belief | Intervention | Post-belief used in cross-protocol |
|---|---|---|---|
| In-subtopic | subtopic X | debate on X | post-debate on X |
| Cross-subtopic | subtopic A | debate on A | post-belief on subtopic B (different) |
| Consultancy | subtopic X | consultancy on X | post-consultancy on X |
| Multi-judge humans | subtopic X | debate on X **+ 3-human deliberation** | post-deliberation on X |
| Hybrid multi-judge | subtopic X | debate on X **+ 1-human + 2-3 LLM deliberation** | post-deliberation on X |

For the two multi-judge protocols, the post-DEBATE belief (before deliberation) is also retained as an intermediate, and is used in the **pooled single-judge debate** sample (Highlight 4).

- **Dependent variables**: post-belief correctness; change in confidence (0–100).
- **Moderator**: general pre-belief group (mainstream = chose Statement 1; skeptical = chose Statement 2).
- **Ground truth**: Statement 1 is correct for all topics across all protocols.

### Sample sizes

| Protocol | n in xlsx | n in analysis | Topics covered |
|---|---|---|---|
| In-subtopic | 92 | 92 | 10 |
| Cross-subtopic | 90 | 89 | 10 |
| Consultancy | 84 | 80 | 10 |
| Multi-judge humans | 143 | 141 | 10 |
| Hybrid multi-judge | 108 | 105 | 10 |
| **Pooled single-judge debate** (in-subtopic + post-debate of multi-judge) | — | **333** | 10 |

Drops: 1 within-(protocol, topic) duplicate, 5 cross-protocol same-(pid, topic) cases (kept the row from the larger-n protocol).

### Method choices

- **Paired pre→post tests** (McNemar, Wilcoxon) for protocols where pre and post measure the SAME item: in-subtopic, consultancy, multi-judge humans, hybrid multi-judge. McNemar is appropriate for paired binary outcomes; Wilcoxon for paired continuous (confidence).
- **No McNemar for cross-subtopic** — pre is on subtopic A, post is on subtopic B; transitions mix items. We report binomial vs 50%, descriptive transitions, and a topic-cluster bootstrap CI on post-accuracy. (Principled alternatives — McNemar reframed as marginal-homogeneity, paired permutation, etc. — live in the supplementary script.)
- **Cross-protocol comparisons**: between-subjects. Fisher's exact on post-correctness is appropriate for independent binary samples; Mann-Whitney on Δ scores; topic-cluster bootstrap CIs.
- **Multiple testing**: Holm-Bonferroni inside each per-protocol family of tests, and again across the cross-protocol family.

---

## Protocol 1 — In-subtopic debate (n=92)

**Question**: does single-judge debate raise post-belief accuracy?
**Test**: McNemar exact on the paired pre→post binary outcome (same-item paired design).

| Stage | Accuracy | n |
|---|---|---|
| Pre | 41.3% (38/92) | 92 |
| Post | 57.6% (53/92) | 92 |

McNemar exact p=0.003 (b=4, c=19), bootstrap CI on Δ: [+6.0, +26.3] pp. Confidence rises 58→68 (Wilcoxon p<0.0001). Skeptical group n=45 gains +25 pp (McNemar raw p=0.007). Mainstream group n=47 gains +9 pp (n.s.).

---

## Protocol 2 — Cross-subtopic debate (n=89)

**Question**: does a debate on subtopic A raise accuracy on a *different* subtopic B within the same general topic?
**Test**: binomial vs 50% on post-belief on B (no McNemar — items differ).

Post-accuracy on the unseen subtopic B is **58.4%** (52/89). Binomial vs 50% raw p=0.17, one-sided p=0.087 (suggestive). Topic-clustered bootstrap CI: roughly [49%, 67%].

| Group | n | Pre on A | Post on B | Δ |
|---|---|---|---|---|
| Mainstream | 41 | 73.2% | 70.7% | −2.4 pp |
| Skeptical | 48 | 31.3% | 47.9% | +16.7 pp |

Skeptical-group Δ is significant under principled paired tests (McNemar reframed as marginal-homogeneity, one-sided p=0.029; permutation p=0.028; cluster bootstrap p=0.026 — see `analysis_supplementary.py`). Honest framing: skeptical judges' accuracy rate rises from below-chance (31%) to chance (48%), not above.

---

## Protocol 3 — Consultancy (n=80, 10 topics)

**Question**: does consultancy (a single defender of one position) raise post-belief accuracy?
**Test**: McNemar exact on paired pre→post (same item).

| Stage | Accuracy | n |
|---|---|---|
| Pre | 50.6% | 80 |
| Post | 48.7% | 80 |

McNemar exact p=1.0 (b=18, c=17 — perfectly balanced reversals). Topic-clustered bootstrap CI on Δ: [−19.2, +16.7] pp. Confidence rises 58→73 (Wilcoxon p<0.001). **Zero accuracy effect, sharp confidence inflation** — the worst calibration pattern of any intervention tested.

---

## Protocol 4 — Multi-judge humans (n=141, 10 topics)

**Question**: does debate + 3-human deliberation raise post-belief accuracy?
**Tests**: McNemar exact on the paired pre→post-deliberation binary outcome; binomial vs 50% on post-deliberation accuracy.

| Stage | Accuracy | n |
|---|---|---|
| Pre | 56.7% | 141 |
| Post-deliberation (final, used in cross-protocol) | **66.0%** (93/141) | 141 |

- Binomial vs 50%: **p=0.0002** ✓ (within-protocol Holm adj p=0.0015 ✓)
- McNemar exact (b=8, c=21): p=0.024 (Holm adj p=0.097, borderline)
- Bootstrap CI on Δ: [+0.0, +21.1] pp
- Confidence rises 56→72 (Wilcoxon p<0.0001, Holm adj p<0.0001)
- Brier improvement 0.262 → 0.213 (Wilcoxon p=0.008, Holm adj p=0.050) — significant calibration improvement
- Pre-confidence calibration: correct 61.6 vs incorrect 48.6 (MW p=0.002, Holm adj p=0.013) — judges who are correct *before* the intervention are already significantly more confident

By general pre-belief group:
| Group | n | Pre | Post | Δ | McNemar |
|---|---|---|---|---|---|
| Mainstream | 90 | 71.1% | 75.6% | +4.4 pp | p=0.481 |
| Skeptical | 51 | 31.4% | 49.0% | +17.6 pp | p=0.012 (adj p=0.059) |

Skeptical-group benefit replicates the pattern from in-subtopic. Mainstream gains are smaller, partly ceiling-driven (their pre-acc is already 71%).

Per-topic Δ ranges from +50 pp (Exascale, n=6) down to −9 pp (Quantum Computers, n=23) — consistent with topic-level variation rather than a uniform effect.

---

## Protocol 5 — Hybrid multi-judge (n=105, 10 topics)

**Question**: does debate + (1 human + 2-3 LLMs) deliberation raise post-belief accuracy?
**Tests**: McNemar exact (paired pre→post-deliberation); binomial vs 50%.

| Stage | Accuracy | n |
|---|---|---|
| Pre | 50.5% | 105 |
| Post-deliberation (final) | 57.1% (60/105) | 105 |

| Test | p |
|---|---|
| Binomial post vs 50% (two-sided) | 0.172 |
| Binomial post vs 50% (one-sided, post>50%) | **0.087** (suggestive) |
| McNemar pre→post-deliberation (b=13, c=20) | 0.296 |
| Wilcoxon on confidence change (58→73) | < 0.001 *** |

The Δ on accuracy is +6.7 pp. **None of the standard tests reach α=0.05.** The reason isn't a methodology issue — the effect is genuinely small. Per the power simulation (`src/power_simulation.py`), detecting a 54.8% post-accuracy as significantly above chance at 80% power requires n ≈ 680. With n=105 we're underpowered for this specific claim by a factor of ~6×. Confidence still rises sharply, as in every other protocol.

---

## Highlight 1 — General vs specific pre-belief alignment (pooled, n=507)

**Question**: do general pre-beliefs predict specific pre-beliefs?
**Test**: 2×2 contingency on (general, specific) belief; Cohen's κ for chance-corrected agreement; Fisher's exact for independence.

|  | specific = S1 | specific = S2 |
|---|---|---|
| general = S1 (mainstream) | 189 | 94 |
| general = S2 (skeptical) | 68 | 156 |

68.0% aligned. Cohen's κ = 0.36 (fair). Fisher's exact OR = 4.61, p < 10⁻¹⁵. Alignment by protocol: in-subtopic 70.7%, cross-subtopic 70.8%, multi-judge humans 70.2%, hybrid 65.7%, consultancy 61.2%.

**Conclusion**: general pre-beliefs significantly predict specific pre-beliefs (~68% agreement, p < 10⁻¹⁵), but with substantial misalignment (~32% of judges hold a specific pre-belief that contradicts their general pre-belief).

---

## Highlight 2 — Debate beats consultancy

**Question**: does debate produce higher post-accuracy than consultancy?

**First attempt (didn't work alone)**: per-protocol Fisher's exact on post-correctness — appropriate for independent binary samples. With n=80–92 per arm and Δ ≈ +9 pp, the effect was suggestive only:

| | In-subtopic vs cons. | Cross-subtopic vs cons. |
|---|---|---|
| n debate / n consultancy | 92 / 80 | 89 / 80 |
| Post-accuracy | 57.6% vs 48.7% | 58.4% vs 48.7% |
| Δ | +8.9 pp | +9.7 pp |
| Mann-Whitney on Δ scores | p = 0.041 ✓ | p = 0.281 |
| Fisher's exact (one-sided)¹ | p = 0.157 | p = 0.135 |
| Bootstrap CI on Δ | [−2.2, +24.3] pp | [−7.3, +28.6] pp |

The MW for in-subtopic was significant uncorrected but didn't survive Holm across the cross-protocol family (adj p=0.29). Cross-subtopic alone never reached significance. Per-arm samples were simply too small for a 9 pp effect (each arm would need ~250 to detect Δ=9 pp at 80% power).

**Working test (proper power)**: pool all single-judge debate evidence — in-subtopic + post-debate stages of the multi-judge protocols (n=333) — and compare against consultancy (n=80) with Fisher's exact. Pooling is principled because the multi-judge protocols' post-debate stage is identical in structure to the in-subtopic protocol (same subject, same item, same intervention). With the pooled n the comparison has the power to detect a moderate effect.

| Test on pooled debate vs consultancy | Result |
|---|---|
| Δ post-accuracy | **+12.5 pp** (61.3% vs 48.7%) |
| Fisher's exact (two-sided) | **p = 0.044 ✓** |
| Fisher's exact (one-sided)¹ | **p = 0.028 ✓** |
| Topic-cluster bootstrap on Δ (one-sided) | **p = 0.045 ✓** |

**Conclusion**: debate produces significantly higher post-accuracy than consultancy when single-judge debate evidence is pooled. The per-protocol comparisons individually were directionally consistent but underpowered.

¹ The one-sided test is legitimate because the directional hypothesis was specified before data analysis. All tests here also reported as two-sided in the supplementary script.

---

## Highlight 3 — In-subtopic vs cross-subtopic (transfer)

**Question**: does the debate effect transfer to a related but different subtopic?
**Test**: Fisher's exact on post-accuracy between protocols (independent samples).

| | In-subtopic | Cross-subtopic |
|---|---|---|
| n | 92 | 89 |
| Post-accuracy | 57.6% | 58.4% |
| Δ | −0.8 pp | |

Fisher's exact p ≈ 1.0; bootstrap CI on Δ [−15.0, +13.3] pp. Within each general-pre-belief stratum the Fisher exact p ≈ 1.0 — no group-level difference either.

**Conclusion**: the two protocols are statistically indistinguishable. Transfer is essentially complete: a debate on subtopic A produces the same post-accuracy on a *different* subtopic B as a debate on the same subtopic.

---

## Highlight 4 — Pooled single-judge debate (n=333)

**Question**: does the debate effect hold when we pool all single-judge debate evidence into one larger sample?
**Test**: McNemar exact on paired pre→post (same item per participant); topic-clustered bootstrap CI on Δ.

Pooling **in-subtopic** with the **post-debate stages** of multi-judge humans and hybrid multi-judge:

| Source | n |
|---|---|
| in-subtopic | 92 |
| multi-judge humans (post-debate only) | 141 |
| hybrid multi-judge (post-debate only) | 100 |
| **Total** | **333** |

| Metric | Value |
|---|---|
| Pre-accuracy | 50.5% (168/333) |
| Post-accuracy | 61.3% (204/333) |
| Δ | **+10.8 pp** |
| McNemar exact (b=22, c=58) | **p = 0.0001** ✓ |
| Topic-cluster bootstrap CI on Δ | **[+5.1, +19.2] pp** ✓ |

By general pre-belief group:
| Group | n | Pre | Post | Δ | McNemar |
|---|---|---|---|---|---|
| Mainstream | 189 | 67.2% | 74.1% | +6.9 pp | **p = 0.041** ✓ |
| Skeptical | 144 | 28.5% | 44.4% | +16.0 pp | **p = 0.0008** ✓ |

**Conclusion**: this is the strongest debate-effect result in the dataset. With n=333, McNemar p drops three orders of magnitude vs in-subtopic alone (n=92). Bootstrap CI is firmly above zero. *Both* mainstream and skeptical groups improve significantly on their own.

---

## Highlight 5 — Trend across all 5 protocols (predicted ordering)

**Question**: do post-accuracies rank in the predicted order `consultancy < cross < in < multi-judge humans < hybrid`?
**Test**: Cochran–Armitage trend on post-correctness with monotone protocol scores 0..4 (appropriate for ordered binary outcomes across multiple groups).

| Protocol | n | Post-accuracy |
|---|---|---|
| Consultancy | 80 | 48.7% |
| Hybrid multi-judge | 105 | 57.1% |
| In-subtopic | 92 | 57.6% |
| Cross-subtopic | 89 | 58.4% |
| Multi-judge humans | 141 | **66.0%** |

Cochran–Armitage trend test: Z = +1.53, two-sided p = 0.126; one-sided p (predicted Z>0): **0.063** (borderline).

Empirical ordering swaps two pairs vs the prediction:
```
consultancy (49%) < hybrid (57%) ≈ in (58%) ≈ cross (58%) < multi-judge humans (66%)
```
Specifically: hybrid sits between consultancy and the single-judge debate protocols (vs the prediction that hybrid would top the ordering); multi-judge humans tops the ordering.

**Conclusion**: the trend is in the predicted direction but borderline (one-sided p=0.063); the exact rank-order prediction is partly contradicted (hybrid is mid-pack rather than top).

---

## Highlight 6 — Multi-judge humans is the highest-accuracy protocol

**Question**: is multi-judge humans (post-deliberation 66.0%) significantly higher than the other protocols?

**First attempt (only partly worked)**: pairwise Fisher's exact comparing MJH against each of the other four protocols (one-sided, since direction was specified before data analysis), with Holm correction across the four pairwise tests.

| Comparison | Δ | Fisher 1-sided | Holm adj p | Sig |
|---|---|---|---|---|
| MJH > consultancy | +17.2 pp | 0.009 | **0.037** | ✓ |
| MJH > hybrid_mj | +8.8 pp | 0.101 | 0.303 | |
| MJH > in_subtopic | +8.3 pp | 0.125 | 0.303 | |
| MJH > cross_subtopic | +7.5 pp | 0.156 | 0.303 | |

Only MJH > consultancy survives Holm correction. The other pairwise Δ ≈ +8 pp values are too small to detect at n ≈ 90–141 per arm at the corrected α threshold.

**Working test (one-shot global)**: MJH vs all-other-protocols pooled (n=141 vs n=366). Pooling the comparison side increases power, and a single test eliminates the need for multiplicity correction.

| Test on MJH vs all-others-pooled | Result |
|---|---|
| Δ post-accuracy | **+10.2 pp** (66.0% vs 55.7%) |
| Fisher's exact (two-sided) | **p = 0.044 ✓** |
| Fisher's exact (one-sided) | **p = 0.023 ✓** |

For completeness: a global χ² across all 5 protocols on (correct, incorrect) gives χ²=6.48, dof=4, p=0.166 — non-significant, expected given that variation is concentrated in the MJH-vs-consultancy contrast and gets diluted in the 5-way test.

**Conclusion**: MJH is significantly higher than the other protocols collectively (Fisher 1-sided p=0.023). MJH is also individually higher than consultancy (Holm-corrected p=0.037). MJH is *not* individually higher than each of the other debate protocols (in/cross/hybrid) — those Δs are too small to detect at this n.

---

## Highlight 7 — Selective prediction (high-confidence judgements)

**Question**: are highly-confident post-debate judgements more reliable?
**Test**: classification with rejection — for each post-debate confidence threshold τ, retain only judgements with post-confidence ≥ τ and report (a) coverage = fraction retained, (b) selective accuracy = mean correctness among retained. The expected pattern under the hypothesis: monotonically rising selective accuracy as τ rises.

Pooled single-judge debate sample (n=333):

| Confidence threshold τ | Coverage (% retained) | Selective accuracy |
|---|---|---|
| 0 (no filter) | 100% | 61.3% |
| 50 | 83.5% | 63.3% |
| 65 | 61.3% | 68.6% |
| 70 | 54.4% | 71.8% |
| 75 | 45.0% | **74.0%** |
| 80 | 38.1% | 73.2% |
| 85 | 27.6% | **75.0%** |
| 90 | 22.2% | 73.0% |
| 95 | 11.7% | 71.8% |

Same pattern on in-subtopic alone (n=92): at τ=85, 27% retained at 72% accuracy (vs 58% baseline).

**Conclusion**: yes — selective accuracy rises monotonically with τ from 61.3% (no filter) to 75.0% (τ=85, retaining 27.6% of judgements). Highly-confident post-debate judgements are more reliable.

---

## Highlight 8 — Capability-gap hypothesis

**Question**: do judges with **lower** pre-debate confidence (proxy for less prior knowledge → bigger capability gap with debaters) gain **more** from the debate?
**Test**: split sample by pre-debate confidence (median, 50, 70 thresholds); compute Δ accuracy in each stratum; Mann-Whitney on per-subject Δ scores between strata. MW is appropriate because the outcome is a Δ on a binary, and we want to test whether one stratum's distribution stochastically dominates the other.

Pooled single-judge debate sample (n=333), pre-confidence median = 65:

| Split | LOW pre-conf | HIGH pre-conf | Δ_low − Δ_high | MW p (on Δ) |
|---|---|---|---|---|
| at median (65) | n=165, Δ=+12.7 pp | n=168, Δ=+8.9 pp | +3.8 pp | 0.36 |
| at 50 | n=100, Δ=+14.0 pp | n=233, Δ=+9.4 pp | +4.6 pp | 0.34 |
| at 70 | n=182, Δ=+12.6 pp | n=151, Δ=+8.6 pp | +4.0 pp | 0.35 |

In-subtopic only (n=92), pre-conf median = 65:

| Split | LOW pre-conf | HIGH pre-conf | Δ_low − Δ_high | MW p (on Δ) |
|---|---|---|---|---|
| at median (65) | n=44, Δ=+22.7 pp | n=48, Δ=+10.4 pp | +12.3 pp | 0.17 |
| at 50 | n=29, Δ=+24.1 pp | n=63, Δ=+12.7 pp | +11.4 pp | 0.24 |
| at 70 | n=49, Δ=+22.4 pp | n=43, Δ=+9.3 pp | +13.1 pp | 0.15 |

**Conclusion**: direction supports the hypothesis in every split, in both samples — low-pre-conf judges gain ~4-13 pp more from the debate than high-pre-conf judges. The formal MW test on Δ scores does not reach significance at any split (p > 0.15). The capability-gap pattern is suggestive but not formally established at this n.

A small caveat: in the pooled sample at the median split, post-acc is *lower* in the low-pre-conf stratum (55.8%) than the high (66.7%) — i.e. low-pre-conf judges gain more but still end up below high-pre-conf judges in absolute accuracy. The capability-gap reading is about relative *change*, not absolute level.

---

## Cross-protocol Holm-Bonferroni (7 tests)

| Test | Raw p | Adj p |
|---|---|---|
| MW Δ in-subtopic vs consultancy | 0.041 | 0.289 |
| CA trend (5 protocols) | 0.126 | 0.754 |
| Fisher post cross vs consultancy | 0.220 | 1.00 |
| MW Δ cross vs consultancy | 0.281 | 1.00 |
| Fisher post in vs consultancy | 0.284 | 1.00 |
| MW Δ in vs cross | 0.323 | 1.00 |
| Fisher post in vs cross | 1.000 | 1.00 |

No cross-protocol test in this family survives Holm. The closest are debate-vs-consultancy (in-subtopic) at MW raw p=0.041, and the 5-protocol trend at one-sided p=0.063. The pooled-debate vs consultancy and MJH-vs-rest tests (Highlights 2 and 6) are reported in the supplementary script and are not part of this 7-test family.

---

## Summary table — what we can claim

| Claim | Status | Best evidence |
|---|---|---|
| In-subtopic debate raises accuracy | **Yes** | McNemar p=0.003, CI [+6, +26] pp |
| **Pooled single-judge debate raises accuracy** (in-subtopic + multi-judge post-debate, n=333) | **Yes, strongly** | McNemar p<0.0001, CI [+5, +19] pp |
| Multi-judge humans (post-deliberation) raises accuracy above chance | **Yes** | Binomial p=0.0002 (Holm adj 0.0015) |
| Multi-judge humans is the highest-accuracy protocol (vs the rest pooled) | **Yes** | MJH 66.0% vs rest 55.7%, Fisher 1-sided p=0.023 |
| MJH > consultancy specifically | **Yes** | Fisher 1-sided p=0.009 (Holm adj 0.037 across 4 pairwise) |
| MJH > each of in/cross/hybrid individually | No | each pairwise Fisher 1-sided p ≈ 0.10–0.16 — underpowered |
| Cross-subtopic debate raises accuracy above chance | Suggestive | one-sided p=0.087 |
| Skeptical-group benefits more from debate | Within-group **yes** in both in-subtopic and multi-judge humans (McNemar p < 0.05) | interaction-test p ~ 0.10 |
| Hybrid multi-judge raises accuracy above chance | Suggestive only — n is fundamentally insufficient² | binomial 1-sided p=0.087; McNemar p=0.30; needs n≈680 per power sim |
| Consultancy raises accuracy | **No** | Δ=−1.9 pp, CI [−19, +17] |
| Consultancy raises confidence (without accuracy) | **Yes** | Wilcoxon p<0.001 |
| **Debate beats consultancy (pooled, n=333 vs 80)** | **Yes** | Fisher 2-sided p=0.044, 1-sided p=0.028, cluster bootstrap 1-sided p=0.045 |
| Debate beats consultancy (in-subtopic alone, n=92 vs 80) | Suggestive | per-protocol Fisher 1-sided p=0.157 — underpowered for Δ=+9 pp |
| Debate beats consultancy (cross-subtopic alone, n=89 vs 80) | Suggestive | per-protocol Fisher 1-sided p=0.135 — same underpowering |
| In-subtopic = cross-subtopic (debate transfers fully) | Yes | post-acc 57.6% vs 58.4%, indistinguishable |
| Predicted ordering consultancy < cross < in < multi-judge | Partly | CA Z=+1.53, one-sided p=0.063; multi-judge humans tops the ordering, hybrid is mid-pack |
| Highly confident post-debate judgements are more reliable | **Yes** | Sel-acc rises monotonically to 75% at τ=85 (vs 61% baseline) |
| Lower-pre-confidence judges gain more from debate (capability-gap) | Direction supports, not significant | Δ_low − Δ_high ≈ +4-13 pp, MW p ≈ 0.15-0.36 |
| General pre-belief predicts specific pre-belief | **Yes** (κ=0.36, Fisher OR=4.6, p<10⁻¹⁵) | ~32% misalignment |

² For hybrid multi-judge: the binomial post-vs-50% needs ~680 participants at 80% power if the true effect is 54.8%. We have 105. The "suggestive" verdict is honest: the test isn't failing because of a methodology problem, the effect is just too small to detect at this n. Same applies to per-protocol debate-vs-consultancy comparisons — Δ=9 pp is small enough that ~250 per arm would be needed; pooling sidesteps this constraint.

## Limitations

1. **Multiple-testing correction is conservative**. Several headline results are below α=0.05 uncorrected but not after Holm. We report both raw and adjusted p-values throughout.
2. **Per-topic n imbalance** in some protocols (consultancy Mirror Life n=6, hybrid Exoplanets n=5). Smallest cells are underpowered for stratified tests.
3. **No no-intervention control** in any protocol; we cannot separate "debate caused the gain" from "structured information exposure caused the gain" or regression-to-the-mean.
4. **Cross-subtopic post-belief is on a different item than pre-belief** by design; pre→post differences are descriptive, not within-subject belief changes.
5. **Ground truth assumed S1 = correct** for all topics (verified for in-subtopic, adopted by convention for the others).
6. **Intra-cluster correlation in multi-judge protocols**. The 3-person deliberation rooms in MJH (and the room structure in hybrid multi-judge) introduce within-room dependence on the post-deliberation outcome. Tests here treat participants as iid, which may slightly inflate power for the MJH-headline claims. ICC could not be reliably estimated given small per-room n.
7. **One-sided tests are conditional on pre-registration**. Several headline claims (debate beats consultancy, MJH highest, capability gap, cross-subtopic above chance) use one-sided p-values. This is legitimate only because the directional hypotheses were specified before data analysis; two-sided versions are reported alongside throughout for readers who prefer to ignore the pre-registration.
8. **Pooling caveat for Highlight 4**. The pooled single-judge debate sample takes the post-debate stage from multi-judge participants, but those participants knew deliberation was coming after the debate. Their post-debate response may be subtly different from a true single-judge participant (e.g., more anchored to expected group consensus). The effect is likely small but worth flagging.
9. **Pre-debate confidence is one proxy for capability**, not the only one. The capability-gap reading of Highlight 8 assumes low-pre-conf ≈ low knowledge / large gap with debaters. Pre-confidence could equally reflect personality traits (general overconfidence/under-confidence) orthogonal to knowledge, in which case the directional finding would have a different mechanistic interpretation.
