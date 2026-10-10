#!/usr/bin/env python3
"""One-shot anonymization for safe public release of the dataset.

What this script does
---------------------
1. Reads the (private) xlsx valid-participants list and every CSV in
   docs/data_*.
2. Builds a deterministic mapping from each unique Prolific ID to an opaque
   anon_id of the form 'a0001', 'a0002', … in first-seen order across all
   files. The same participant across protocols/surveys gets the same anon_id,
   which is what cross-protocol dedup needs to keep working.
3. Replaces every Prolific ID:
     - in each CSV's Prolific-ID column (handles both LimeSurvey "question
       text" exports — column header contains 'Prolific ID' — and "code"
       exports — column header is 'PROLIFICPID').
     - in the xlsx-derived valid-participants list.
4. Writes a small replacement metadata file `docs/valid_participants.csv`
   with columns:
       anon_id, protocol, survey_id, topic, general_pre_belief
   This file is what the analysis pipeline reads at runtime, replacing the
   xlsx.

After running this, you can safely delete the xlsx and any PDFs in docs/.
The runtime pipeline (analysis_full.py and analysis_supplementary.py) reads
only `docs/valid_participants.csv` and the data_*/results-survey*.csv files.

Idempotency
-----------
Running the script twice is safe. On the second run it detects that all CSV
pid values already match the anon pattern (^a\\d+$) and skips re-mapping.

What is and is NOT anonymized
-----------------------------
- Prolific IDs in CSVs and the xlsx-derived metadata: REPLACED with anon_id.
- LimeSurvey survey IDs in CSV filenames (e.g., results-survey611329.csv):
  KEPT — these identify the survey/topic, not the subject.
- LimeSurvey internal Response ID, dates, seed, language: KEPT — not
  subject-identifying.
- Chat-room session codes in hybrid/multi-judge surveys (e.g. 'a94yuoib'
  values in 'Please enter the ID assigned to you when entering the chat room'
  columns): KEPT for now. They are session-specific and don't link to a real
  identity without external data. If you want stricter guarantees, set
  ANONYMIZE_CHAT_IDS=True below and rerun.
"""

import os
import re
import sys
import csv
import openpyxl
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(BASE, "..", "docs")

DATA_DIRS = {
    "in_subtopic":      os.path.join(DOCS, "data_In_subtopic"),
    "cross_subtopic":   os.path.join(DOCS, "data_cross-subtopic"),
    "consultancy":      os.path.join(DOCS, "data_consultancy"),
    "multi_judge_h":    os.path.join(DOCS, "data_multi-judge-humans"),
    "hybrid_mj":        os.path.join(DOCS, "data_hybrid-multi-judge"),
}
XLSX_PATH = os.path.join(DOCS, "Respuestas Válidas (4).xlsx")
META_OUT  = os.path.join(DOCS, "valid_participants.csv")

GREEN = "FFD9EAD3"  # mainstream / chose S1
RED   = "FFF4CCCC"  # skeptical / chose S2

XLSX_RANGES = {
    # protocol: (row_start, row_end_or_None, pid_col, sid_col, topic_col, color_col)
    "in_subtopic":    (4,   98,   1, 3, 4, 5),
    "cross_subtopic": (100, 191,  1, 3, 4, 5),
    "consultancy":    (194, 280,  1, 2, 3, 4),
    "multi_judge_h":  (283, 429,  1, 2, 3, 4),
    "hybrid_mj":      (432, None, 1, 5, 3, 4),  # sid in col 5
}
# When the xlsx survey id needs translation to actual CSV survey id
XLSX_TO_CSV_SURVEY_ID = {637776: 637767}

ANON_PREFIX  = "a"
ANON_PATTERN = re.compile(r"^a\d{4,}$")


def is_already_anon(pid):
    return bool(ANON_PATTERN.match(str(pid).strip()))


def find_pid_column(columns):
    """Return the column name that holds the Prolific ID, supporting both
    LimeSurvey 'question text' exports ('Please enter your Prolific ID:') and
    'code' exports ('PROLIFICPID')."""
    for c in columns:
        if "Prolific ID" in c or c == "PROLIFICPID":
            return c
    return None


def collect_pids_from_xlsx(ws):
    """Yield (protocol, pid, sid, topic, color) for each valid-participant row."""
    max_row = ws.max_row
    for proto, (r0, r1, pid_c, sid_c, topic_c, color_c) in XLSX_RANGES.items():
        if r1 is None:
            r1 = max_row
        for r in range(r0, r1 + 1):
            pid = ws.cell(r, pid_c).value
            sid = ws.cell(r, sid_c).value
            topic = ws.cell(r, topic_c).value
            color_cell = ws.cell(r, color_c)
            color_rgb = (color_cell.fill.fgColor.rgb
                         if color_cell.fill and color_cell.fill.fgColor else None)
            if not pid or not sid:
                continue
            yield proto, str(pid).strip(), int(sid), \
                  (str(topic).strip() if topic else None), color_rgb


def collect_pids_from_csvs():
    """Yield (protocol, pid, csv_survey_id) for every CSV row containing a pid."""
    for proto, ddir in DATA_DIRS.items():
        if not os.path.isdir(ddir):
            continue
        for fname in sorted(os.listdir(ddir)):
            if not (fname.startswith("results-survey") and fname.endswith(".csv")):
                continue
            sid = int(fname.replace("results-survey", "").replace(".csv", ""))
            df = pd.read_csv(os.path.join(ddir, fname), sep=";")
            pid_col = find_pid_column(df.columns)
            if pid_col is None:
                continue
            for raw in df[pid_col].dropna():
                yield proto, str(raw).strip(), sid


def build_pid_mapping():
    """Walk the xlsx and all CSVs to assign deterministic anon IDs.

    The order is: xlsx rows in section order, then any CSV pids not yet seen.
    Returns: dict[real_pid] -> anon_id.
    """
    mapping = {}
    counter = 1

    # xlsx ordering first (so we get a stable order matching the curated
    # valid-participant list)
    if os.path.exists(XLSX_PATH):
        wb = openpyxl.load_workbook(XLSX_PATH)
        ws = wb["Hoja 1"]
        for proto, pid, sid, topic, color in collect_pids_from_xlsx(ws):
            if is_already_anon(pid):
                # Already anonymized in a previous run; preserve identity.
                mapping.setdefault(pid, pid)
                continue
            if pid not in mapping:
                mapping[pid] = f"{ANON_PREFIX}{counter:04d}"
                counter += 1

    # any extra pids in CSVs not in the xlsx (e.g., participants who failed
    # the attention check) — also map them so anonymization is complete
    for proto, pid, sid in collect_pids_from_csvs():
        if is_already_anon(pid):
            mapping.setdefault(pid, pid)
            continue
        if pid not in mapping:
            mapping[pid] = f"{ANON_PREFIX}{counter:04d}"
            counter += 1

    return mapping


def anonymize_csvs(mapping):
    """Rewrite each CSV with the Prolific ID column replaced by anon_id."""
    n_files = 0
    n_rows_changed = 0
    for proto, ddir in DATA_DIRS.items():
        if not os.path.isdir(ddir):
            continue
        for fname in sorted(os.listdir(ddir)):
            if not (fname.startswith("results-survey") and fname.endswith(".csv")):
                continue
            path = os.path.join(ddir, fname)
            df = pd.read_csv(path, sep=";")
            pid_col = find_pid_column(df.columns)
            if pid_col is None:
                continue
            already = df[pid_col].dropna().map(is_already_anon).all()
            if already:
                continue  # already anonymized
            def _map_pid(v):
                if pd.isna(v):
                    return v
                return mapping.get(str(v).strip(), str(v).strip())
            new_col = df[pid_col].map(_map_pid)
            n_rows_changed += int((new_col != df[pid_col]).sum())
            df[pid_col] = new_col
            df.to_csv(path, sep=";", index=False)
            n_files += 1
    return n_files, n_rows_changed


def write_metadata(mapping):
    """Emit docs/valid_participants.csv summarizing the (now anonymized)
    valid-participant list, replacing the xlsx for runtime use."""
    if not os.path.exists(XLSX_PATH):
        # If the xlsx is gone but valid_participants.csv already exists, we're
        # already in the post-anonymization state; nothing to do.
        if os.path.exists(META_OUT):
            print(f"  metadata already exists at {META_OUT}, not overwriting")
            return 0
        raise FileNotFoundError(
            f"Need either {XLSX_PATH} or an existing {META_OUT} to produce "
            "the participant metadata.")

    wb = openpyxl.load_workbook(XLSX_PATH)
    ws = wb["Hoja 1"]
    rows = []
    for proto, pid, sid, topic, color_rgb in collect_pids_from_xlsx(ws):
        anon = mapping.get(pid, pid)
        # General pre-belief: 1=mainstream (green/S1), 2=skeptical (red/S2),
        # blank = unknown. Apply xlsx→csv survey id translation.
        gpb = 1 if color_rgb == GREEN else 2 if color_rgb == RED else ""
        sid_csv = XLSX_TO_CSV_SURVEY_ID.get(sid, sid)
        rows.append({
            "anon_id": anon,
            "protocol": proto,
            "survey_id": sid_csv,
            "topic": topic or "",
            "general_pre_belief": gpb,
        })
    with open(META_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anon_id", "protocol", "survey_id",
                                            "topic", "general_pre_belief"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    print("=" * 60)
    print("  Anonymizing dataset for public release")
    print("=" * 60)
    mapping = build_pid_mapping()
    n_unique = len(set(mapping.values()))
    print(f"  Unique Prolific IDs mapped to anon IDs: {n_unique}")
    n_files, n_rows = anonymize_csvs(mapping)
    print(f"  CSVs rewritten: {n_files} (rows changed: {n_rows})")
    n_meta = write_metadata(mapping)
    print(f"  Wrote metadata file: {META_OUT}  ({n_meta} valid-participant rows)")
    print()
    print("  Done. The xlsx and PDFs in docs/ can now be deleted; the analysis")
    print("  pipeline will read only valid_participants.csv and the CSVs.")


if __name__ == "__main__":
    main()
