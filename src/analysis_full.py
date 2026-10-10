#!/usr/bin/env python3
"""Unified statistical analysis for the debate experiment — all five protocols.

Protocols
---------
  in_subtopic     pre-belief on subtopic X → debate on X → post-belief on X
  cross_subtopic  pre on subtopic A → debate on A → post-belief on subtopic B
                  (different subtopic, same general topic)
  consultancy     pre on X → consultancy → post on X
  multi_judge_h   pre on X → debate → post-debate → 3-human deliberation →
                  post-deliberation (final). Cross-protocol post = deliberation.
  hybrid_mj       same as multi_judge_h but deliberation is 1 human + 2-3 LLMs.

Outputs
-------
  Per-protocol section (Métricas pasos 1-3): general/specific/post accuracy,
    confidence calibration, belief reversals, alignment, stratified by general
    pre-belief group, per-topic.
  Highlight 1 — general vs specific pre-belief alignment, pooled.
  Highlight 2 — debate beats consultancy, in BOTH in-subtopic (2a) and
                cross-subtopic (2b)  [directional hypothesis].
  Highlight 3 — does in-subtopic effect transfer to cross-subtopic.
  Trend test  — predicted ordering consultancy < cross < in_subtopic
                (< multi_judge_h < hybrid_mj when those have data).
  Pooled single-judge debate — pooling rationale — in-subtopic + post-debate stages
                of multi-judge protocols, larger combined sample.
  Q1 selective prediction (post-debate confidence) — coverage/accuracy at each
                post-debate confidence threshold τ.
  Q2 capability gap (pre-debate confidence stratification) — compare Δ accuracy between low- and
                high-pre-confidence strata.

Method choices
--------------
  - Paired pre/post tests (McNemar, Wilcoxon) for protocols where pre and post
    measure the SAME item: in_subtopic, consultancy, multi_judge_h, hybrid_mj.
  - For cross_subtopic the pre and post items DIFFER; we report the binomial
    post vs 50%, descriptive transitions, and a topic-cluster bootstrap CI on
    post-accuracy. McNemar would mix items and is omitted.
  - Cross-protocol comparisons are between-subjects. Fisher's exact on
    post-correctness; Mann-Whitney on Δ scores; topic-cluster bootstrap CIs.
  - Multiple-testing: Holm-Bonferroni inside each protocol's family of tests
    and again across the cross-protocol family.

Cross-protocol deduplication: same pid + same topic across two protocols →
keep the row from the protocol with the LARGER total n (drop the duplicate
from the smaller protocol). Same pid + DIFFERENT topics → keep all.

Ground truth: Statement 1 is correct for all topics in all protocols.
Survey ID quirk: source-list says 637776 for hybrid Embryo Models but the
  real LimeSurvey ID is 637767 (digit transposition). Mapped here.
LimeSurvey export formats: most CSVs are "question text" (column header
  contains the full question); some multi-judge humans CSVs are "code" format
  (column header is e.g. PROLIFICPID, PreSBMQ1[PreSBM1S1]). The block detector
  handles both.

Data files (all anonymized):
  docs/valid_participants.csv          per-row metadata (anon_id, protocol,
                                       survey_id, topic, general_pre_belief)
  docs/data_*/results-survey<id>.csv   per-survey response data; the prolific
                                       ID column has been replaced by the
                                       same anon_id used in the metadata file
The pipeline depends only on these two file groups; the original spreadsheet
and PDFs are not needed at runtime. To regenerate from the private inputs,
run `python src/anonymize_data.py`.
"""

import os
import csv
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(BASE, "..", "docs")
DATA = {
    "in_subtopic":      os.path.join(DOCS, "data_In_subtopic"),
    "cross_subtopic":   os.path.join(DOCS, "data_cross-subtopic"),
    "consultancy":      os.path.join(DOCS, "data_consultancy"),
    "multi_judge_h":    os.path.join(DOCS, "data_multi-judge-humans"),
    "hybrid_mj":        os.path.join(DOCS, "data_hybrid-multi-judge"),
}

# Map xlsx survey ID → actual CSV survey ID for cases where xlsx has a typo.
# xlsx says 637776 for hybrid Embryo Models; LimeSurvey survey is actually 637767.
XLSX_TO_CSV_SURVEY_ID = {
    637776: 637767,
}

# (PS_id, AL_id) per topic. PS_id is None for consultancy (single combined survey).
TOPICS = {
    "in_subtopic": {
        "Supersonic Aircrafts": (256316, 611329),
        "Mirror Life":          (848936, 843144),
        "Embryo Models":        (251316, 794297),
        "Bioprinting":          (999839, 211685),
        "Black Holes":          (395497, 922129),
        "Exoplanets":           (299539, 196555),
        "Spacecraft Civilians": (731215, 956794),
        "Exascale Computing":   (388398, 954439),
        "Quantum Computers":    (273328, 979424),
        "Nuclear Fusion":       (985711, 252599),
    },
    "cross_subtopic": {
        "Supersonic Aircrafts": (625985, 933923),
        "Nuclear Fusion":       (957838, 594895),
        "Quantum Computers":    (741729, 447814),
        "Spacecraft Civilians": (536195, 358136),
        "Mirror Life":          (117232, 783582),
        "Embryo Models":        (592888, 947761),
        "Black Holes":          (888129, 631795),
        "Exoplanets":           (299539, 444565),  # PS shared with in-subtopic
        "Exascale Computing":   (765461, 155479),
        "Bioprinting":          (115774, 749581),
    },
    # Consultancy: single survey per topic. No separate PS.
    "consultancy": {
        "Bioprinting":          (None, 538362),
        "Quantum Computers":    (None, 184477),
        "Exascale Computing":   (None, 539526),
        "Spacecraft Civilians": (None, 543147),
        "Embryo Models":        (None, 391357),
        "Mirror life":          (None, 976151),  # xlsx spelling
        "Black Holes":          (None, 849391),
        "Nuclear Fusion":       (None, 187811),
        "Supersonic Aircrafts": (None, 765238),  # added in updated xlsx
        "Exoplanets":           (None, 893559),  # added in updated xlsx
    },
    # Multi-judge humans: one survey per debate-room (so multiple surveys per
    # topic). 52 surveys × ~3 participants = ~143 valid pids across all 10
    # topics now. Some surveys are exported in LimeSurvey "code" format (column
    # codes like 'PROLIFICPID', 'PreSBMQ1[PreSBM1S1]') rather than "question
    # text" format (column headers with the full question). The loader handles
    # both formats.
    "multi_judge_h": {
        "Bioprinting":          (None, [216299, 458888]),
        "Black Holes":          (None, [342828, 577215, 942549]),
        "Embryo Models":        (None, [271178, 495236, 631852, 717113, 793217]),
        "Exascale Computing":   (None, [318345, 365149, 961926]),
        "Exoplanets":           (None, [193427, 459169, 678651]),
        "Mirror Life":          (None, [156122, 225146, 635263, 646856, 747336,
                                          861545, 965546]),
        "Nuclear Fusion":       (None, [244221, 314137, 394323, 412589, 894127,
                                          981898, 993551]),
        "Quantum Computers":    (None, [313313, 323679, 347238, 447965, 853599,
                                          877597, 882523, 934262]),
        "Spacecraft Civilians": (None, [237625, 285825, 297251, 335145, 865432,
                                          899348]),
        "Supersonic Aircrafts": (None, [194326, 211995, 463475, 567926, 866177,
                                          895111, 931292, 999926]),
    },
    # Hybrid multi-judge: one survey per topic.
    "hybrid_mj": {
        "Supersonic Aircrafts": (None, 633357),
        "Nuclear Fusion":       (None, 386595),
        "Quantum Computers":    (None, 527567),
        "Spacecraft Civilians": (None, 832991),
        "Mirror Life":          (None, 874873),
        "Embryo Models":        (None, 637776),  # xlsx → 637767 in CSV (see map above)
        "Black Holes":          (None, 423932),
        "Exoplanets":           (None, 438134),
        "Exascale Computing":   (None, 881832),
        "Bioprinting":          (None, 683686),
    },
}

# Block-parsing offsets within a 12-column subtopic block.
# Same offsets for all 3 protocols; only the BLOCK_START differs.
BLOCK_SIZE = 12
PRE_S1, PRE_S2, PRE_CONF   = 1, 2, 3
POST_S1, POST_S2, POST_CONF = 9, 10, 11
BLOCK_START = {
    "in_subtopic":    16,
    "cross_subtopic": 16,
    "consultancy":    20,  # 4 extra cols for general pre-belief
}

GROUND_TRUTH = 1
N_BOOT = 10_000
RNG = np.random.default_rng(42)

# Anonymized valid-participants metadata (replaces the original xlsx for
# runtime use). Generated by `anonymize_data.py`.
META_PATH = os.path.join(DOCS, "valid_participants.csv")


# ── data loading ─────────────────────────────────────────────

def load_xlsx_participants():
    """Return dict[protocol] -> list of valid-participant rows.

    Reads docs/valid_participants.csv (anonymized). Each row has columns:
      anon_id, protocol, survey_id, topic, general_pre_belief
    where general_pre_belief ∈ {1=mainstream, 2=skeptical, ''=unknown}.
    Returned dicts use the same field names as the legacy xlsx loader so the
    rest of the pipeline doesn't need to change ('prolific_id' here holds the
    anon_id).

    The function name is preserved for backwards compatibility with
    analysis_supplementary.py and any downstream code; the data source is now
    a single anonymized CSV instead of the xlsx.
    """
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Expected anonymized metadata at {META_PATH}. Run "
            f"`python src/anonymize_data.py` first to generate it from the "
            f"private xlsx, or place a pre-generated valid_participants.csv "
            f"at that path.")
    out = {p: [] for p in DATA}
    with open(META_PATH, newline="") as f:
        for row in csv.DictReader(f):
            proto = row["protocol"]
            if proto not in out:
                continue
            gpb = row["general_pre_belief"].strip()
            color = int(gpb) if gpb in ("1", "2") else None
            out[proto].append({
                "prolific_id":        row["anon_id"].strip(),  # legacy key name
                "al_survey_id":       int(row["survey_id"]),
                "general_pre_belief": color,
                "topic_xlsx":         (row["topic"].strip() or None),
            })
    return out


def _belief_from_pair(s1_val, s2_val):
    """Return 1 if S1 chosen, 2 if S2 chosen, NaN if ambiguous/blank."""
    s1_yes = (s1_val == "Yes")
    s2_yes = (s2_val == "Yes")
    if s1_yes and not s2_yes:
        return 1
    if s2_yes and not s1_yes:
        return 2
    return np.nan


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _detect_blocks(df, protocol):
    """Detect per-block column indices by scanning column headers.

    Some surveys vary in block size:
      - Standard consultancy: 12-col blocks
      - Consultancy 765238: 11-col blocks (lacks 'chat room ID' column)
      - Multi-judge humans: single block with 3 belief stages (debate +
        deliberation)
      - Hybrid multi-judge: 19-col blocks with 3 belief stages

    Header-based detection handles all of these. For each block we try to find:
      pre-belief, post-debate-belief, post-deliberation-belief
    Multi-judge protocols have all three; the others have pre + post-debate.

    Returns: list of dicts with keys (some may be None when absent)
      pre_s1, pre_s2, pre_conf, pre_header,
      post_s1, post_s2, post_conf, post_header,        — post-debate (or
                                                          consultancy)
      delib_s1, delib_s2, delib_conf, delib_header     — post-deliberation,
                                                          None if N/A
    """
    cols = [str(c) for c in df.columns]
    has_general_belief_block = protocol in ("consultancy", "multi_judge_h",
                                             "hybrid_mj")

    # Detect format: question-text vs LimeSurvey code format.
    # Question-text: c0 = "Response ID". Code: c0 = "id".
    is_code_format = cols[0].lower() == "id"

    if is_code_format:
        # Code-format CSVs (LimeSurvey raw export). Use stable question codes
        # exposed by survey design:
        #   PreSBMQ*[PreSBM*S1] / [PreSBM*S2]    pre-specific belief S1/S2
        #   PostSBMQ*[PostSBM*S1] / [PostSBM*S2] post-debate belief S1/S2
        #   PreSBMB*  / PostSBMB*                pre/post-debate confidence
        #   PostDelSBMB*                         post-deliberation confidence
        # The post-deliberation S1/S2 columns are NOT named with a SBMQ prefix;
        # they appear as G##Q##[SQ001]/[SQ002] but are positioned right before
        # PostDelSBMB. We locate them positionally relative to PostDelSBMB.
        pre_s1_cols  = [i for i, c in enumerate(cols)
                        if c.startswith("PreSBMQ") and c.endswith("S1]")]
        post_s1_cols = [i for i, c in enumerate(cols)
                        if c.startswith("PostSBMQ") and c.endswith("S1]")]
        delib_conf_cols = [i for i, c in enumerate(cols)
                           if c.startswith("PostDelSBMB")]
        # Post-deliberation S1 sits at delib_conf - 2, S2 at delib_conf - 1.
        delib_s1_cols = [i - 2 for i in delib_conf_cols]
    else:
        # Question-text format (the default human-readable export).
        # Post-debate / consultancy: "Based on the consultancy interaction"
        # OR "Based on the content of the debate" + [Statement 1].
        post_s1_cols = [i for i, c in enumerate(cols)
                        if '[Statement 1]' in c
                        and ('Based on the consultancy' in c
                             or 'Based on the content of the debate' in c)]
        # Post-deliberation: "Based on the deliberation" + [Statement 1]
        delib_s1_cols = [i for i, c in enumerate(cols)
                         if '[Statement 1]' in c
                         and 'Based on the deliberation' in c]
        # Pre-belief: contains "Which statement do you think is correct" +
        # [Statement 1] + NOT "Based on..." (post variants also match the
        # phrase otherwise).
        pre_s1_cols = [i for i, c in enumerate(cols)
                       if 'Which statement do you think is correct' in c
                       and '[Statement 1]' in c
                       and 'Based on' not in c]
        # Drop the first one if there's a general-belief block before topics.
        if has_general_belief_block and pre_s1_cols:
            pre_s1_cols = pre_s1_cols[1:]

    blocks = []
    n = min(len(pre_s1_cols), len(post_s1_cols))
    has_delib = len(delib_s1_cols) >= n
    for k in range(n):
        p = pre_s1_cols[k]
        q = post_s1_cols[k]
        d = delib_s1_cols[k] if has_delib else None
        block = {
            "pre_s1": p, "pre_s2": p + 1, "pre_conf": p + 2, "pre_header": p - 1,
            "post_s1": q, "post_s2": q + 1, "post_conf": q + 2, "post_header": q - 1,
            "delib_s1": d,
            "delib_s2": d + 1 if d is not None else None,
            "delib_conf": d + 2 if d is not None else None,
            "delib_header": d - 1 if d is not None else None,
        }
        blocks.append(block)
    return blocks


def _find_filled_block(row, blocks):
    """Return the first block whose post-belief S1 or S2 has a non-null value.

    For multi-judge protocols we prefer a block where deliberation is filled,
    since that's the final belief; if no such block exists fall back to one
    where post-debate is filled.
    """
    # Prefer deliberation-filled blocks when the protocol has them
    has_delib = any(b["delib_s1"] is not None for b in blocks)
    if has_delib:
        for b in blocks:
            if b["delib_s1"] is None:
                continue
            if (pd.notna(row.iloc[b["delib_s1"]])
                    or pd.notna(row.iloc[b["delib_s2"]])):
                return b
    for b in blocks:
        if pd.notna(row.iloc[b["post_s1"]]) or pd.notna(row.iloc[b["post_s2"]]):
            return b
    return None


def load_protocol_csvs(protocol, participants):
    """Read all CSVs for a protocol and extract per-participant rows.

    Block layout detected per-CSV from column headers. Multi-judge protocols
    additionally extract a post-deliberation belief; for them, `post_belief` in
    the returned data is the post-deliberation belief (the final belief), and
    `post_debate_belief` is retained as an intermediate column.

    Some entries in TOPICS map a topic to a list of survey IDs (multi-judge
    humans, where each debate room has its own survey).
    """
    rows = []
    pid_lookup = {(p["prolific_id"], p["al_survey_id"]): p for p in participants[protocol]}

    has_delib_protocol = protocol in ("multi_judge_h", "hybrid_mj")

    for topic, (_, sid_or_list) in TOPICS[protocol].items():
        survey_ids = sid_or_list if isinstance(sid_or_list, list) else [sid_or_list]
        for al_id in survey_ids:
            # Apply xlsx→csv typo correction (harmless if not present)
            csv_id = XLSX_TO_CSV_SURVEY_ID.get(al_id, al_id)
            path = os.path.join(DATA[protocol], f"results-survey{csv_id}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep=";")
            pid_col = next(
                (c for c in df.columns if "Prolific ID" in c or c == "PROLIFICPID"),
                None)
            if pid_col is None:
                continue
            blocks = _detect_blocks(df, protocol)

            for _, r in df.iterrows():
                pid = str(r[pid_col]).strip()
                # The xlsx may use the original (uncorrected) survey ID for lookup
                key = (pid, al_id)
                key_alt = (pid, csv_id)
                if key not in pid_lookup and key_alt not in pid_lookup:
                    continue
                lookup_entry = pid_lookup.get(key) or pid_lookup.get(key_alt)

                b = _find_filled_block(r, blocks)
                if b is None:
                    continue

                pre_b = _belief_from_pair(r.iloc[b["pre_s1"]], r.iloc[b["pre_s2"]])
                post_debate_b = _belief_from_pair(r.iloc[b["post_s1"]],
                                                   r.iloc[b["post_s2"]])

                # For multi-judge protocols, post-deliberation is the final belief
                if has_delib_protocol and b["delib_s1"] is not None:
                    delib_b = _belief_from_pair(r.iloc[b["delib_s1"]],
                                                 r.iloc[b["delib_s2"]])
                    delib_conf = _num(r.iloc[b["delib_conf"]])
                else:
                    delib_b = np.nan
                    delib_conf = np.nan

                # `post_belief` for cross-protocol use is post-deliberation when
                # available, else post-debate.
                post_belief = delib_b if (has_delib_protocol
                                           and not pd.isna(delib_b)) else post_debate_b
                post_conf   = (delib_conf if (has_delib_protocol
                                              and not pd.isna(delib_conf))
                               else _num(r.iloc[b["post_conf"]]))

                rows.append({
                    "protocol": protocol,
                    "prolific_id": pid,
                    "al_survey_id": al_id,  # keep xlsx-side id for joins
                    "csv_survey_id": csv_id,
                    "topic": topic,
                    "block_start": b["pre_s1"] - 1,
                    "specific_pre_belief":  pre_b,
                    "pre_confidence":       _num(r.iloc[b["pre_conf"]]),
                    "post_debate_belief":   post_debate_b,
                    "post_debate_confidence": _num(r.iloc[b["post_conf"]]),
                    "post_deliberation_belief":     delib_b,
                    "post_deliberation_confidence": delib_conf,
                    "post_belief":     post_belief,
                    "post_confidence": post_conf,
                    "pre_subtopic_text":  str(df.columns[b["pre_header"]])[:120],
                    "post_subtopic_text": str(df.columns[b["post_header"]])[:120],
                    "general_pre_belief": lookup_entry["general_pre_belief"],
                })
    return pd.DataFrame(rows)


def build_data():
    """Build the unified per-participant dataframe across all protocols.

    Dedup applies the documentation page-4 rule:
      step 1 — within (protocol, topic), drop same-pid duplicates (keep first).
      step 2 — same pid + same topic across DIFFERENT protocols → keep the row
               from the protocol with the larger total n (drop the duplicate
               from the smaller protocol). Same pid + DIFFERENT topics → keep
               all (legitimate independent participation).

    Note: 'el del experimento que más respuestas tenga' in the documentation is
    grammatically ambiguous; we interpret it as 'keep the larger experiment',
    i.e. preserve the participant in the more statistically powerful protocol.
    """
    parts = load_xlsx_participants()
    frames = []
    n_in_xlsx = {}
    for proto in DATA:
        n_in_xlsx[proto] = len(parts[proto])
        df = load_protocol_csvs(proto, parts)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Drop rows where pre or post belief is unparseable
    n_before = len(data)
    data = data.dropna(subset=["specific_pre_belief", "post_belief"])
    n_dropped_parse = n_before - len(data)

    # Step 1: within-(protocol, topic) duplicates (true retakes/multiple-rows)
    n_before = len(data)
    data = data.drop_duplicates(subset=["protocol", "topic", "prolific_id"],
                                 keep="first")
    n_dropped_within = n_before - len(data)

    # Step 2: cross-protocol same-(pid, topic) — keep row from protocol with
    # larger total n. Compute total-n per protocol BEFORE this step so the
    # ranking reflects the dataset as a whole, not after-iterating-prunings.
    proto_n = data.groupby("protocol").size().to_dict()
    data = data.assign(__proto_n=data["protocol"].map(proto_n))
    n_before = len(data)
    data = (data
            .sort_values("__proto_n", ascending=False)
            .drop_duplicates(subset=["prolific_id", "topic"], keep="first")
            .drop(columns="__proto_n")
            .reset_index(drop=True))
    n_dropped_cross = n_before - len(data)

    data["pre_correct"]  = (data["specific_pre_belief"] == GROUND_TRUTH).astype(int)
    data["post_correct"] = (data["post_belief"]         == GROUND_TRUTH).astype(int)
    # Post-debate-only correctness for protocols where it differs from final post
    data["post_debate_correct"] = (
        data["post_debate_belief"] == GROUND_TRUTH
    ).astype("Int64")
    data["gen_correct"]  = (data["general_pre_belief"] == GROUND_TRUTH).astype("Int64")
    data["change"]       = data["post_correct"] - data["pre_correct"]
    return data, n_in_xlsx, n_dropped_parse, n_dropped_within, n_dropped_cross


# ── statistics helpers ───────────────────────────────────────

def _hdr(title, ch="="):
    print(f"\n{ch * 64}\n  {title}\n{ch * 64}")


def _sub(title):
    _hdr(title, ch="-")


def _mcnemar(sub):
    b = ((sub["pre_correct"] == 1) & (sub["post_correct"] == 0)).sum()
    c = ((sub["pre_correct"] == 0) & (sub["post_correct"] == 1)).sum()
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else np.nan
    return int(b), int(c), p


def _bootstrap_topic_cluster(data, statistic_fn, n_boot=N_BOOT):
    """Cluster-bootstrap CI by topic. statistic_fn(df) -> scalar."""
    topics = data["topic"].unique()
    out = np.empty(n_boot)
    for i in range(n_boot):
        sampled = RNG.choice(topics, size=len(topics), replace=True)
        boot = pd.concat([data[data["topic"] == t] for t in sampled],
                         ignore_index=True)
        out[i] = statistic_fn(boot)
    return np.percentile(out, [2.5, 97.5])


def _brier(conf, correct):
    return ((conf / 100.0 - correct) ** 2).mean()


def _holm_bonferroni(pvals):
    sorted_pv = sorted(pvals, key=lambda x: x[1])
    m = len(sorted_pv)
    out, max_so_far = [], 0.0
    for i, (name, p) in enumerate(sorted_pv):
        if pd.isna(p):
            out.append((name, p, p))
            continue
        adj = min(p * (m - i), 1.0)
        adj = max(adj, max_so_far)
        max_so_far = adj
        out.append((name, p, adj))
    return out


def _cochran_armitage(success, n, weights):
    """Cochran–Armitage trend test on proportions.

    success: array of #successes per ordered group
    n:       array of group sizes
    weights: monotone scores per group (e.g., [0,1,2])

    Returns (Z, two-sided p).
    """
    success = np.asarray(success, dtype=float)
    n = np.asarray(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    N = n.sum()
    R = success.sum()
    p_bar = R / N
    T = (w * success).sum() - p_bar * (w * n).sum()
    var = p_bar * (1 - p_bar) * ((w**2 * n).sum() - ((w * n).sum())**2 / N)
    if var <= 0:
        return np.nan, np.nan
    Z = T / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    return Z, p


def _fisher_2x2(a, b, c, d):
    """Fisher's exact two-sided on 2×2 [[a, b], [c, d]]. Returns (OR, p)."""
    return stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")


# ── per-protocol analyses ────────────────────────────────────

def analyze_protocol(data, protocol, label, paired_pre_post):
    """Run the full per-protocol analysis. Returns dict of p-values for FDR.

    paired_pre_post=True for in_subtopic and consultancy (same Q pre/post).
    paired_pre_post=False for cross_subtopic (different Q).
    """
    sub = data[data["protocol"] == protocol].copy()
    n = len(sub)
    pvals = {}

    _hdr(f"PROTOCOL: {label.upper()}  (n = {n})")
    print(f"Topics covered: {sub['topic'].nunique()}")
    print(sub.groupby("topic").size().to_string())

    # Métricas Paso 1 ─ general pre-belief distribution
    _sub("1. General pre-belief distribution (Paso 1)")
    gn = sub["general_pre_belief"].notna().sum()
    s1 = (sub["general_pre_belief"] == 1).sum()
    s2 = (sub["general_pre_belief"] == 2).sum()
    if gn > 0:
        print(f"  Mainstream (S1): {s1}/{gn} ({100*s1/gn:.1f}%)")
        print(f"  Skeptical (S2):  {s2}/{gn} ({100*s2/gn:.1f}%)")
    else:
        print("  No general pre-belief recorded for this protocol.")

    # Métricas Paso 2 ─ specific pre-belief accuracy
    _sub("2. Specific pre-belief accuracy (Paso 2)")
    pre_acc = sub["pre_correct"].mean()
    bt = stats.binomtest(int(sub["pre_correct"].sum()), n, 0.5)
    print(f"  Pre-belief accuracy: {pre_acc:.3f} ({sub['pre_correct'].sum()}/{n})")
    print(f"  Binomial vs 50%: p = {bt.pvalue:.4f}")
    pvals["pre vs 50"] = bt.pvalue

    # Pre-confidence calibration
    _sub("3. Pre-confidence calibration (Paso 2)")
    cc = sub.loc[sub["pre_correct"] == 1, "pre_confidence"].dropna()
    ci = sub.loc[sub["pre_correct"] == 0, "pre_confidence"].dropna()
    print(f"  Correct pre-belief mean conf:   {cc.mean():.1f}  (n={len(cc)})")
    print(f"  Incorrect pre-belief mean conf: {ci.mean():.1f}  (n={len(ci)})")
    if len(cc) > 0 and len(ci) > 0:
        u, p = stats.mannwhitneyu(cc, ci, alternative="two-sided")
        print(f"  Mann-Whitney U: U={u:.0f}, p={p:.4f}")
        pvals["pre conf calibration"] = p

    # Post accuracy
    _sub("4. Post-belief accuracy (Paso 3)")
    post_acc = sub["post_correct"].mean()
    bt2 = stats.binomtest(int(sub["post_correct"].sum()), n, 0.5)
    print(f"  Post accuracy: {post_acc:.3f} ({sub['post_correct'].sum()}/{n})")
    print(f"  Binomial vs 50%: p = {bt2.pvalue:.4f}")
    pvals["post vs 50"] = bt2.pvalue

    if paired_pre_post:
        delta = post_acc - pre_acc
        print(f"  Δaccuracy (post − pre on same Q): {delta:+.3f}")
        b, c, mc_p = _mcnemar(sub)
        print(f"  McNemar exact (b={b}, c={c}): p = {mc_p:.4f}")
        pvals["McNemar pre→post"] = mc_p
        try:
            lo, hi = _bootstrap_topic_cluster(
                sub, lambda d: d["post_correct"].mean() - d["pre_correct"].mean()
            )
            print(f"  Topic-cluster bootstrap CI on Δ: [{lo:+.3f}, {hi:+.3f}]")
        except Exception as e:
            print(f"  (bootstrap skipped: {e})")
    else:
        # Cross-subtopic: pre and post measure DIFFERENT subtopics. Report
        # both accuracies side by side, but DO NOT run McNemar — a 1↔0
        # transition mixes two distinct questions and is not a coherent
        # within-subject belief change.
        print(f"  Pre-accuracy on subtopic A:  {pre_acc:.3f} (different question)")
        print(f"  Post-accuracy on subtopic B: {post_acc:.3f}")
        print("  No McNemar reported: pre and post measure different subtopics,")
        print("  so a flip is not a within-subject belief change about the same item.")
        try:
            lo, hi = _bootstrap_topic_cluster(sub, lambda d: d["post_correct"].mean())
            print(f"  Topic-cluster bootstrap CI on POST-accuracy: [{lo:.3f}, {hi:.3f}]")
        except Exception as e:
            print(f"  (bootstrap skipped: {e})")

    # Belief reversals: descriptive table
    _sub("5. Belief transitions table")
    a = ((sub["pre_correct"] == 1) & (sub["post_correct"] == 1)).sum()
    b = ((sub["pre_correct"] == 1) & (sub["post_correct"] == 0)).sum()
    c = ((sub["pre_correct"] == 0) & (sub["post_correct"] == 1)).sum()
    d = ((sub["pre_correct"] == 0) & (sub["post_correct"] == 0)).sum()
    if not paired_pre_post:
        print("  (Cross-subtopic: pre and post are different items — read as")
        print("   joint distribution, NOT a belief-change matrix.)")
    print(f"  Correct→Correct:    {a}")
    print(f"  Correct→Incorrect:  {b}")
    print(f"  Incorrect→Correct:  {c}")
    print(f"  Incorrect→Incorrect:{d}")

    # Confidence change
    _sub("6. Confidence change")
    m = sub["pre_confidence"].notna() & sub["post_confidence"].notna()
    pre_c, post_c = sub.loc[m, "pre_confidence"], sub.loc[m, "post_confidence"]
    print(f"  Mean pre-confidence:  {pre_c.mean():.1f}")
    print(f"  Mean post-confidence: {post_c.mean():.1f}")
    if len(pre_c) > 1:
        w, p = stats.wilcoxon(pre_c, post_c)
        print(f"  Wilcoxon signed-rank: W={w:.0f}, p={p:.4f}")
        if not paired_pre_post:
            print("  (Confidence is about the participant's stance; meaningful even")
            print("   though the post item differs from the pre item — both reflect")
            print("   the participant's certainty in their chosen statement.)")
        pvals["Wilcoxon conf"] = p

    # Brier
    _sub("7. Brier score")
    if m.sum() > 1:
        pre_b = _brier(sub.loc[m, "pre_confidence"],  sub.loc[m, "pre_correct"])
        post_b = _brier(sub.loc[m, "post_confidence"], sub.loc[m, "post_correct"])
        print(f"  Pre Brier:  {pre_b:.4f}")
        print(f"  Post Brier: {post_b:.4f}")
        if paired_pre_post:
            pre_bs  = ((sub.loc[m,"pre_confidence"]/100 - sub.loc[m,"pre_correct"])**2).values
            post_bs = ((sub.loc[m,"post_confidence"]/100 - sub.loc[m,"post_correct"])**2).values
            wb, pbr = stats.wilcoxon(pre_bs, post_bs)
            print(f"  Wilcoxon on individual Brier: W={wb:.0f}, p={pbr:.4f}")
            pvals["Brier improvement"] = pbr
        else:
            print("  (No paired Wilcoxon: pre and post Brier scored on different items.)")

    # By general pre-belief group
    _sub("8. Accuracy by general pre-belief group (Paso 1-3)")
    for label_g, val in [("Mainstream (S1)", 1), ("Skeptical (S2)", 2)]:
        g = sub[sub["general_pre_belief"] == val]
        if len(g) == 0:
            print(f"  {label_g}: n=0, skipped")
            continue
        pa = g["pre_correct"].mean()
        ppa = g["post_correct"].mean()
        print(f"  {label_g}  n={len(g):3d}  pre={pa:.3f}  post={ppa:.3f}  Δ={ppa-pa:+.3f}")
        if paired_pre_post and len(g) >= 4:
            bg, cg, pg = _mcnemar(g)
            print(f"    McNemar (b={bg}, c={cg}): p = {pg:.4f}")
            pvals[f"McNemar {label_g[:11]}"] = pg

    # Interaction — does the change differ between groups?
    if paired_pre_post:
        m1 = sub[sub["general_pre_belief"] == 1]
        m2 = sub[sub["general_pre_belief"] == 2]
        if len(m1) > 0 and len(m2) > 0:
            u_int, p_int = stats.mannwhitneyu(m1["change"], m2["change"],
                                              alternative="two-sided")
            print(f"\n  Interaction (Δ differs between groups?)")
            print(f"  Mainstream Δ mean: {m1['change'].mean():+.3f}")
            print(f"  Skeptical Δ mean:  {m2['change'].mean():+.3f}")
            print(f"  Mann-Whitney U on Δ scores: U={u_int:.0f}, p={p_int:.4f}")
            pvals["Δ-interaction"] = p_int
    else:
        m1 = sub[sub["general_pre_belief"] == 1]
        m2 = sub[sub["general_pre_belief"] == 2]
        if len(m1) > 0 and len(m2) > 0:
            u_int, p_int = stats.mannwhitneyu(m1["post_correct"], m2["post_correct"],
                                              alternative="two-sided")
            print(f"\n  Post-accuracy by general pre-belief (between-group)")
            print(f"  Mann-Whitney U on post_correct: U={u_int:.0f}, p={p_int:.4f}")
            pvals["post-by-group"] = p_int

    # Per-topic
    _sub("9. Per-topic")
    for t in sorted(sub["topic"].unique()):
        ts = sub[sub["topic"] == t]
        pa = ts["pre_correct"].mean()
        ppa = ts["post_correct"].mean()
        print(f"  {t:25s} n={len(ts):2d}  pre={pa:.3f}  post={ppa:.3f}  Δ={ppa-pa:+.3f}")

    # Holm correction within protocol
    _sub("10. Holm-Bonferroni within protocol")
    adj = _holm_bonferroni(list(pvals.items()))
    print(f"  {'Test':<28s}  {'raw p':>8s}  {'adj p':>8s}  Sig")
    for name, raw, a in adj:
        if pd.isna(raw):
            continue
        sig = "***" if a < 0.001 else "**" if a < 0.01 else "*" if a < 0.05 else ""
        print(f"  {name:<28s}  {raw:8.4f}  {a:8.4f}  {sig}")

    return pvals


# ── cross-protocol analyses ──────────────────────────────────

def cross_protocol_alignment(data):
    """Highlight #1: General-vs-specific alignment, pooled across protocols."""
    _hdr("HIGHLIGHT 1: GENERAL vs SPECIFIC ALIGNMENT (pooled)")
    pool = data[data["general_pre_belief"].notna()].copy()
    pool["aligned"] = (pool["general_pre_belief"] == pool["specific_pre_belief"])
    n = len(pool)
    al = pool["aligned"].sum()
    print(f"  Pooled n = {n} across {pool['protocol'].nunique()} protocols")
    print(f"  Aligned (general == specific): {al}/{n} ({100*al/n:.1f}%)")
    print(f"  Misaligned:                    {n-al}/{n} ({100*(n-al)/n:.1f}%)")

    # 2×2: rows = general (1/2), cols = specific (1/2). Cohen's kappa.
    a = ((pool["general_pre_belief"]==1) & (pool["specific_pre_belief"]==1)).sum()
    b = ((pool["general_pre_belief"]==1) & (pool["specific_pre_belief"]==2)).sum()
    c = ((pool["general_pre_belief"]==2) & (pool["specific_pre_belief"]==1)).sum()
    d = ((pool["general_pre_belief"]==2) & (pool["specific_pre_belief"]==2)).sum()
    p_obs = (a + d) / n
    p_exp = (((a + b) * (a + c)) + ((c + d) * (b + d))) / (n * n)
    kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else np.nan
    OR, p_fisher = _fisher_2x2(a, b, c, d)
    print(f"\n  2×2 contingency (general × specific):")
    print(f"            specific=1   specific=2")
    print(f"  general=1    {a:3d}          {b:3d}")
    print(f"  general=2    {c:3d}          {d:3d}")
    print(f"  Cohen's κ = {kappa:.3f}")
    print(f"  Fisher's exact (2-sided): OR = {OR:.2f}, p = {p_fisher:.4g}")

    # Stratified by general belief value
    for val, lbl in [(1, "General correct (mainstream)"),
                     (2, "General incorrect (skeptical)")]:
        g = pool[pool["general_pre_belief"] == val]
        if len(g) == 0:
            continue
        same = (g["specific_pre_belief"] == val).sum()
        print(f"\n  {lbl}: specific also matches general = {same}/{len(g)} "
              f"({100*same/len(g):.1f}%)")

    # By protocol
    print("\n  Alignment by protocol:")
    for proto in pool["protocol"].unique():
        g = pool[pool["protocol"] == proto]
        al = g["aligned"].sum()
        print(f"    {proto:15s}  {al}/{len(g)} ({100*al/len(g):.1f}%)")

    return {"alignment kappa": kappa, "alignment fisher": p_fisher}


def cross_protocol_compare(data, label, p1, p2, predicted_higher=None):
    """Highlights #2 and #3 framework: compare two protocols on post-accuracy + Δ.

    p1, p2: protocol names
    predicted_higher: name of the protocol predicted to have higher accuracy
    """
    _hdr(f"COMPARE: {p1.upper()} vs {p2.upper()}  ({label})")
    a = data[data["protocol"] == p1]
    b = data[data["protocol"] == p2]
    print(f"  n({p1}) = {len(a)}, n({p2}) = {len(b)}")

    # Post accuracy
    post_a = a["post_correct"].sum()
    post_b = b["post_correct"].sum()
    pa = post_a / len(a)
    pb = post_b / len(b)
    delta = pa - pb
    print(f"  Post-accuracy {p1}: {pa:.3f} ({post_a}/{len(a)})")
    print(f"  Post-accuracy {p2}: {pb:.3f} ({post_b}/{len(b)})")
    print(f"  Δ = {delta:+.3f}  ({p1} − {p2})")

    OR, p_fisher = _fisher_2x2(post_a, len(a)-post_a, post_b, len(b)-post_b)
    print(f"  Fisher's exact on post-correct: OR={OR:.2f}, p = {p_fisher:.4f}")

    # Cluster bootstrap on Δ post-accuracy (resample topics within each protocol)
    def _delta(df):
        ga = df[df["protocol"] == p1]
        gb = df[df["protocol"] == p2]
        if len(ga) == 0 or len(gb) == 0:
            return np.nan
        return ga["post_correct"].mean() - gb["post_correct"].mean()
    pooled = pd.concat([a, b], ignore_index=True)
    try:
        lo, hi = _bootstrap_topic_cluster(pooled, _delta)
        print(f"  Topic-cluster bootstrap CI on Δ post-accuracy: [{lo:+.3f}, {hi:+.3f}]")
    except Exception as e:
        print(f"  (bootstrap skipped: {e})")

    # Mann-Whitney on Δ (change scores). Only meaningful when both protocols
    # have a paired pre/post. Cross-subtopic's Δ score is descriptive (different
    # questions), but we still report MW on it for symmetry.
    u, p_u = stats.mannwhitneyu(a["change"], b["change"], alternative="two-sided")
    print(f"  Mann-Whitney U on Δ scores: U={u:.0f}, p={p_u:.4f}")

    # Stratified by general pre-belief
    print("\n  Stratified by general pre-belief group:")
    for val, lbl in [(1, "Mainstream"), (2, "Skeptical")]:
        ga = a[a["general_pre_belief"] == val]
        gb = b[b["general_pre_belief"] == val]
        if len(ga) == 0 or len(gb) == 0:
            print(f"    {lbl}: insufficient data ({len(ga)} vs {len(gb)})")
            continue
        pa_g = ga["post_correct"].mean()
        pb_g = gb["post_correct"].mean()
        OR_g, pf_g = _fisher_2x2(ga["post_correct"].sum(), len(ga) - ga["post_correct"].sum(),
                                  gb["post_correct"].sum(), len(gb) - gb["post_correct"].sum())
        print(f"    {lbl:11s}  {p1}: {pa_g:.3f} (n={len(ga)})   "
              f"{p2}: {pb_g:.3f} (n={len(gb)})   Fisher p={pf_g:.4f}")

    if predicted_higher:
        sig = "✓" if (p_fisher < 0.05 and (delta > 0) == (predicted_higher == p1)) else "·"
        direction = p1 if delta > 0 else p2
        print(f"\n  Predicted higher: {predicted_higher}; observed higher: {direction}  [{sig}]")

    return {f"Fisher post {p1} vs {p2}": p_fisher,
            f"MW Δ {p1} vs {p2}": p_u}


def cross_protocol_trend(data):
    """Highlight #3 trend: predicted ordering of post-accuracy by protocol.

    Base order: consultancy < cross_subtopic < in_subtopic. When multi-judge
    protocols have data, they're appended at the high end (multi-judge debate
    + deliberation should match or exceed single-judge debate by convention's
    hypothesis). Cochran–Armitage on post_correct with ordered scores.
    """
    _hdr("HIGHLIGHT 3 (TREND): predicted ordering of protocols")
    order = ["consultancy", "cross_subtopic", "in_subtopic"]
    if (data["protocol"] == "multi_judge_h").sum() >= 5:
        order.append("multi_judge_h")
    if (data["protocol"] == "hybrid_mj").sum() >= 5:
        order.append("hybrid_mj")
    print(f"  Tested ordering: {' < '.join(order)}")
    weights = list(range(len(order)))
    success, n = [], []
    for proto in order:
        s = data[data["protocol"] == proto]
        success.append(int(s["post_correct"].sum()))
        n.append(len(s))
        print(f"  {proto:15s}  n={len(s):3d}  post-accuracy={s['post_correct'].mean():.3f}")

    Z, p = _cochran_armitage(success, n, weights)
    print(f"\n  Cochran–Armitage trend test (one ordering):")
    print(f"    Z = {Z:+.3f},  two-sided p = {p:.4f}")
    if not pd.isna(Z):
        # one-sided p (testing predicted direction)
        p_one = stats.norm.cdf(-Z) if Z < 0 else 1 - stats.norm.cdf(Z)
        # We predicted Z>0 (post-accuracy increases with the ordered weights).
        p_predicted = 1 - stats.norm.cdf(Z) if Z > 0 else stats.norm.cdf(Z)
        # Use the predicted direction one-sided
        p_one_sided_predicted = 1 - stats.norm.cdf(Z)
        print(f"    one-sided p (predicted direction Z>0): {p_one_sided_predicted:.4f}")

    # Stratified by general pre-belief
    for val, lbl in [(1, "Mainstream"), (2, "Skeptical")]:
        sub = data[data["general_pre_belief"] == val]
        success_v, n_v = [], []
        for proto in order:
            s = sub[sub["protocol"] == proto]
            success_v.append(int(s["post_correct"].sum()))
            n_v.append(len(s))
        if min(n_v) > 0:
            Zv, pv = _cochran_armitage(success_v, n_v, weights)
            accs = [s/(nv if nv else 1) for s, nv in zip(success_v, n_v)]
            print(f"\n  {lbl} stratum:")
            for proto, nv, acc in zip(order, n_v, accs):
                print(f"    {proto:15s}  n={nv:2d}  post={acc:.3f}")
            print(f"    CA trend Z={Zv:+.3f}, two-sided p = {pv:.4f}")

    return {f"CA trend ({len(order)} protocols)": p}


# ── pooled single-judge debate — pooled single-judge debate ───────────

def build_pooled_single_judge_debate(data):
    """Pool in-subtopic + post-DEBATE stage of multi-judge protocols.

    Pooling rationale: the multi-judge experiments include an
    individual debate stage BEFORE deliberation. If we take just that stage
    (pre-belief → debate → post-debate-belief) from the multi-judge protocols
    and pool with in-subtopic, we get a much larger single-judge debate sample
    (>320 participants).

    Returned dataframe has the same shape as a single-protocol slice of `data`,
    but for multi-judge rows the `post_belief` / `post_correct` / `post_confidence`
    columns are replaced by their post-DEBATE counterparts (so deliberation
    is excluded).
    """
    parts = []
    # in-subtopic — already debate-only
    parts.append(data[data["protocol"] == "in_subtopic"].copy())
    # multi-judge protocols — swap post_belief for post_debate_belief
    for proto in ("multi_judge_h", "hybrid_mj"):
        sub = data[data["protocol"] == proto].copy()
        if len(sub) == 0:
            continue
        # Use the post-debate-only fields for the pooled "post" measurement
        sub["post_belief"]      = sub["post_debate_belief"]
        sub["post_correct"]     = sub["post_debate_correct"].astype("Int64")
        sub["post_confidence"]  = sub["post_debate_confidence"]
        # Drop rows where post-debate belief is missing (some participants
        # only completed up to deliberation, never logged post-debate)
        sub = sub.dropna(subset=["post_belief", "post_correct"])
        sub["post_correct"] = sub["post_correct"].astype(int)
        sub["change"] = sub["post_correct"] - sub["pre_correct"]
        parts.append(sub)
    pooled = pd.concat(parts, ignore_index=True)
    return pooled


def analyze_pooled_in_subtopic(data):
    """Run a single-judge-debate analysis on the pooled sample."""
    pooled = build_pooled_single_judge_debate(data)
    if len(pooled) == 0:
        return {}
    _hdr("POOLED SINGLE-JUDGE DEBATE  (in-subtopic + post-debate of multi-judge)")
    print(f"  Pooled n = {len(pooled)} (in-subtopic + post-debate stages of")
    print(f"  multi-judge humans and hybrid multi-judge, before deliberation).")
    print(f"  Counts by source protocol:")
    print(pooled.groupby("protocol").size().to_string(header=False))
    n = len(pooled)
    pre_acc  = pooled["pre_correct"].mean()
    post_acc = pooled["post_correct"].mean()
    print(f"\n  Pre-accuracy:  {pre_acc:.3f} ({pooled['pre_correct'].sum()}/{n})")
    print(f"  Post-accuracy: {post_acc:.3f} ({pooled['post_correct'].sum()}/{n})")
    print(f"  Δ = {post_acc - pre_acc:+.3f}")
    b, c, mc_p = _mcnemar(pooled)
    print(f"  McNemar exact (b={b}, c={c}): p = {mc_p:.4f}")
    # Topic-cluster bootstrap on Δ
    try:
        lo, hi = _bootstrap_topic_cluster(
            pooled, lambda d: d["post_correct"].mean() - d["pre_correct"].mean()
        )
        print(f"  Topic-cluster bootstrap CI on Δ: [{lo:+.3f}, {hi:+.3f}]")
    except Exception as e:
        print(f"  (bootstrap skipped: {e})")
    # By general pre-belief group
    print("\n  By general pre-belief group:")
    for val, lbl in [(1, "Mainstream (S1)"), (2, "Skeptical (S2)")]:
        g = pooled[pooled["general_pre_belief"] == val]
        if len(g) == 0:
            continue
        b_g, c_g, p_g = _mcnemar(g)
        print(f"    {lbl}  n={len(g):3d}  pre={g['pre_correct'].mean():.3f}  "
              f"post={g['post_correct'].mean():.3f}  "
              f"Δ={g['post_correct'].mean() - g['pre_correct'].mean():+.3f}  "
              f"McNemar p={p_g:.4f}")
    return {"Pooled McNemar": mc_p}


# ── confidence calibration — confidence calibration ──────────────

def selective_prediction_analysis(data):
    """Q1: Are highly-confident post-debate judgements more accurate?

    Reference: Khan et al. 2024 (arxiv 2402.06782) Insight 8 — rejecting
    judgements below confidence τ retains some fraction at higher accuracy.
    For each threshold τ in {0..95 step 5}, we report:
      - coverage    = #(post_confidence ≥ τ) / n
      - selective accuracy = mean(post_correct) among retained
    Run on (a) in-subtopic alone, (b) the pooled single-judge debate sample.
    """
    _hdr("Q1. SELECTIVE PREDICTION (post-debate confidence)")
    print("  For each post-debate confidence threshold τ, retain only judgements")
    print("  with post-confidence ≥ τ and report:")
    print("    coverage = fraction retained")
    print("    selective_acc = accuracy among retained")
    print()
    samples = [
        ("in-subtopic only",
         data[data["protocol"] == "in_subtopic"].copy()),
        ("pooled (in-subtopic + multi-judge post-debate)",
         build_pooled_single_judge_debate(data)),
    ]
    thresholds = list(range(0, 100, 5))
    for label, sub in samples:
        sub = sub.dropna(subset=["post_confidence", "post_correct"]).copy()
        n_total = len(sub)
        if n_total == 0:
            continue
        print(f"  ── {label} (n = {n_total}) ──")
        print(f"  {'τ':>4s}  {'kept':>6s}  {'coverage':>9s}  {'sel.acc':>9s}")
        for tau in thresholds:
            keep = sub["post_confidence"] >= tau
            kept = int(keep.sum())
            if kept == 0:
                print(f"  {tau:>4d}  {kept:>6d}  {0:>9.3f}  {'-':>9s}")
                continue
            cov = kept / n_total
            sel_acc = sub.loc[keep, "post_correct"].mean()
            print(f"  {tau:>4d}  {kept:>6d}  {cov:>9.3f}  {sel_acc:>9.3f}")
        print()


def capability_gap_analysis(data):
    """Q2: Do judges with LOWER pre-debate confidence achieve higher post-acc?

    Hypothesis (capability-gap): bigger capability asymmetry between judge and
    debaters → larger debate effect. Lower pre-debate confidence is a proxy
    for lower judge capability. Test:
      - split sample by pre-debate confidence (median, then 50/100 hard cutoffs)
      - compare post-accuracy between low-pre-conf and high-pre-conf strata
      - also compare Δ (post − pre) — the debate gain.
    """
    _hdr("Q2. CAPABILITY-GAP HYPOTHESIS (pre-debate confidence stratification)")
    print("  Hypothesis: judges with LOWER pre-debate confidence (≈ less prior")
    print("  knowledge, larger gap with debaters) gain MORE from the debate.")
    print("  Predicts: Δ accuracy is larger in the low-pre-conf stratum.")
    print()
    samples = [
        ("in-subtopic only",
         data[data["protocol"] == "in_subtopic"].copy()),
        ("pooled (in-subtopic + multi-judge post-debate)",
         build_pooled_single_judge_debate(data)),
    ]
    for label, sub in samples:
        sub = sub.dropna(subset=["pre_confidence", "post_correct",
                                  "pre_correct"]).copy()
        if len(sub) == 0:
            continue
        median_conf = sub["pre_confidence"].median()
        print(f"  ── {label} (n = {len(sub)}, pre-conf median = "
              f"{median_conf:.1f}) ──")
        for split_name, mask_low in [
            (f"split at median ({median_conf:.0f})",
             sub["pre_confidence"] < median_conf),
            ("split at 50",  sub["pre_confidence"] < 50),
            ("split at 70",  sub["pre_confidence"] < 70),
        ]:
            low  = sub[mask_low]
            high = sub[~mask_low]
            if len(low) < 5 or len(high) < 5:
                print(f"    {split_name}: n_low={len(low)}, n_high={len(high)} "
                      f"— too small, skipped")
                continue
            pre_low,  pre_high  = low["pre_correct"].mean(),  high["pre_correct"].mean()
            post_low, post_high = low["post_correct"].mean(), high["post_correct"].mean()
            d_low,    d_high    = post_low - pre_low,         post_high - pre_high
            # Test: Δ between low and high strata (Mann-Whitney on change scores)
            u, p_u = stats.mannwhitneyu(low["change"], high["change"],
                                         alternative="two-sided")
            # Also Fisher on post-correct
            a = int(low["post_correct"].sum()); A = len(low)
            b = int(high["post_correct"].sum()); B = len(high)
            _, p_f = _fisher_2x2(a, A - a, b, B - b)
            print(f"    {split_name}:")
            print(f"      LOW  pre-conf  n={A:3d}  pre={pre_low:.3f}  "
                  f"post={post_low:.3f}  Δ={d_low:+.3f}")
            print(f"      HIGH pre-conf  n={B:3d}  pre={pre_high:.3f}  "
                  f"post={post_high:.3f}  Δ={d_high:+.3f}")
            print(f"      Δ_low − Δ_high = {d_low - d_high:+.3f}   "
                  f"(MW on Δ p={p_u:.4f}, Fisher post-acc p={p_f:.4f})")
        print()


# ── main ─────────────────────────────────────────────────────

def run():
    data, n_xlsx, n_drop_parse, n_drop_within, n_drop_cross = build_data()

    _hdr("SAMPLE OVERVIEW (all protocols)")
    proto_order = ["in_subtopic", "cross_subtopic", "consultancy",
                   "multi_judge_h", "hybrid_mj"]
    for proto in proto_order:
        n = (data["protocol"] == proto).sum()
        print(f"  {proto:15s}  valid list n={n_xlsx[proto]:3d}  →  in analysis={n}")
    print(f"  Total dropped (unparseable belief): {n_drop_parse}")
    print(f"  Total dropped (within-protocol+topic duplicates): {n_drop_within}")
    print(f"  Total dropped (cross-protocol same-(pid,topic), keep larger): "
          f"{n_drop_cross}")

    # Per-protocol — each section runs its own Holm correction internally.
    analyze_protocol(data, "in_subtopic",
                     "in-subtopic (debate, same Q)", paired_pre_post=True)
    analyze_protocol(data, "cross_subtopic",
                     "cross-subtopic (debate, different Q)",
                     paired_pre_post=False)
    analyze_protocol(data, "consultancy",
                     "consultancy (same Q)", paired_pre_post=True)
    if (data["protocol"] == "multi_judge_h").sum() > 0:
        analyze_protocol(data, "multi_judge_h",
                         "multi-judge humans (debate + deliberation, same Q)",
                         paired_pre_post=True)
    if (data["protocol"] == "hybrid_mj").sum() > 0:
        analyze_protocol(data, "hybrid_mj",
                         "hybrid multi-judge (debate + human+LLM deliberation, same Q)",
                         paired_pre_post=True)

    # Multi-judge: also report the post-DEBATE accuracy as an intermediate, for
    # decoupling debate-only effect from debate+deliberation effect.
    for proto in ["multi_judge_h", "hybrid_mj"]:
        sub = data[data["protocol"] == proto]
        if len(sub) == 0:
            continue
        _hdr(f"INTERMEDIATE: {proto} post-DEBATE (before deliberation)", ch="-")
        n_d = sub["post_debate_correct"].notna().sum()
        if n_d == 0:
            print("  No post-debate data parsed.")
            continue
        post_dbt_acc = sub["post_debate_correct"].dropna().astype(int).mean()
        print(f"  Post-debate (before deliberation) accuracy: {post_dbt_acc:.3f} "
              f"(n with debate data = {n_d})")
        post_delib_acc = sub["post_correct"].mean()
        print(f"  Post-deliberation (final) accuracy:         {post_delib_acc:.3f} "
              f"(n = {len(sub)})")
        print(f"  Δ deliberation effect (final − post-debate): "
              f"{post_delib_acc - post_dbt_acc:+.3f}")

    # ── Pooled single-judge debate — pooling rationale ──
    analyze_pooled_in_subtopic(data)

    # ── Cross-protocol highlights ──
    h1 = cross_protocol_alignment(data)

    # Highlight 2 — Debate beats consultancy. the claim (directional hypothesis)
    # asks for this in BOTH in-subtopic and cross-subtopic, so each is a
    # co-equal sub-claim, not a primary + secondary.
    h2a = cross_protocol_compare(
        data, "Highlight 2a — in-subtopic debate vs consultancy",
        p1="in_subtopic", p2="consultancy",
        predicted_higher="in_subtopic")
    h2b = cross_protocol_compare(
        data, "Highlight 2b — cross-subtopic debate vs consultancy",
        p1="cross_subtopic", p2="consultancy",
        predicted_higher="cross_subtopic")

    # Highlight 3 — does the in-subtopic effect transfer to cross-subtopic?
    h3 = cross_protocol_compare(
        data, "Highlight 3 — in-subtopic vs cross-subtopic",
        p1="in_subtopic", p2="cross_subtopic",
        predicted_higher="in_subtopic")

    # Predicted-ordering trend test (consultancy < cross < in < multi-judge)
    h_trend = cross_protocol_trend(data)

    # ── New confidence calibration analyses ──
    selective_prediction_analysis(data)
    capability_gap_analysis(data)

    # Family-wise correction across all cross-protocol inferential tests
    family = []
    for d in [h2a, h2b, h3, h_trend]:
        family.extend(d.items())
    _hdr("CROSS-PROTOCOL HOLM-BONFERRONI")
    print(f"  {'Test':<54s}  {'raw p':>8s}  {'adj p':>8s}  Sig")
    for name, raw, a in _holm_bonferroni(family):
        if pd.isna(raw):
            continue
        sig = "***" if a < 0.001 else "**" if a < 0.01 else "*" if a < 0.05 else ""
        print(f"  {name:<54s}  {raw:8.4f}  {a:8.4f}  {sig}")

    _hdr("DONE")


if __name__ == "__main__":
    run()
