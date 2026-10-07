"""
Multi-Judge Voting Experiment
=====================================
The number of judges and persona distribution are determined dynamically from
the dataset CSV — they are NOT hardcoded. The CSV must have columns:
  general_topic, specific_topic, statement_1, statement_2,
  judge_name_1, general_belief_1, [judge_name_2, general_belief_2, ...]

Each debate runs 2 phases:
  Phase 1: Prior belief extraction (before debate)  — PROMPT_PRE_DEBATE
  Phase 2: Post-debate vote                         — PROMPT_POST_DEBATE

Saving strategy:
  - JSON and CSV are saved/updated after EACH phase (when all judges in that phase
    have responded). Incomplete phases appear as None in the CSV.
  - If the script is interrupted mid-phase (a judge is still pending), only the
    state at the END of the previous completed phase is persisted.

Error policy:
  - API failure → abort that voting, log to errors_summary.json, save nothing more.
  - Parsing failure → LLM fallback; if still fails, log and store None, continue.
  - errors_summary.json collects all debates with issues (API aborts + parsing errors).
"""

import anthropic
import json
import os
import pandas as pd
import time
import re
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from dotenv import load_dotenv

# =============================================================================
# API CONFIGURATION
# =============================================================================

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. "
        "Create a .env file with: ANTHROPIC_API_KEY=your-key-here"
    )

claude_client = anthropic.Anthropic(api_key=api_key)

JUDGE_MODEL = "claude-sonnet-4-20250514"
PARSING_FALLBACK_MODEL = "claude-haiku-4-5-20251001"
ENABLE_LLM_PARSING_FALLBACK = True

# =============================================================================
# PROMPTS  (verbatim from voting_nuevo/prompts_borrador.md)
# =============================================================================

SYSTEM_PROMPT_PERSONA = """You are {judge_name} a judge with a prior belief about {general_topic} as follows: {general_belief}.
You should respond authentically from this perspective throughout all interactions. You are not allowed to refuse a response."""

PROMPT_PRE_DEBATE = """The following statements are about {specific_topic}:

Statement 1: {statement_1}

Statement 2: {statement_2}

Which statement do you think is correct? Express your confidence on a scale from 0 to 100.

Format your response as:

Statement: [Statement 1 / Statement 2]
Confidence: [0-100]"""

PROMPT_POST_DEBATE = """You will act as an expert judge in a debate where two debaters are arguing over which of their two opposing statements is correct about a certain topic. Your task is to evaluate both the quality of the arguments presented and determine which statement you believe is correct.

You must take into account your prior belief as defined in the system prompt when making your judgment. Your evaluation should be based only on this prior belief and the arguments presented in the debate. Do not rely on your general knowledge or external information beyond these inputs. If the debate provides sufficiently strong reasons to revise your prior belief, you may update your judgment accordingly.

The following statements are about {specific_topic}:

Statement 1: {statement_1}

Statement 2: {statement_2}

{debater_1} is arguing for {assigned_statement_debater_1} and {debater_2} is arguing for {assigned_statement_debater_2}.

Here is the transcript of the debate:

<transcript>
{transcript}
</transcript>

You are part of a jury evaluating this debate. Provide your independent evaluation and cast your vote. A final decision will be determined after considering all judges' inputs.

Now, you should provide your evaluation. Think step by step and analyze the key arguments presented by both debaters, assessing their strengths and weaknesses.

First, explain your reasoning. Then, provide your answers in the following format:
1. Based on the arguments presented, which statement do you think is correct?
Statement: [Statement 1 / Statement 2]
2. How confident are you in your statement choice?
Confidence: [0-100]
"""

# =============================================================================
# DATASET LOADING
# =============================================================================

def load_persona_dataset(persona_path: str) -> Optional[pd.DataFrame]:
    """Load dataset with judge personas per topic row."""
    try:
        df = pd.read_csv(persona_path)
        print(f"[PERSONA] Loaded {len(df)} rows from {persona_path}")
        return df
    except Exception as e:
        print(f"[ERROR] Could not load persona dataset: {e}")
        return None


def detect_num_judges(persona_df: pd.DataFrame) -> int:
    """Detect how many judges are defined by counting judge_name_N columns."""
    i = 1
    while f"judge_name_{i}" in persona_df.columns:
        i += 1
    return i - 1


def get_judges_for_topic(topic: str, persona_df: pd.DataFrame,
                         num_judges: int) -> Optional[List[Dict]]:
    """
    Return list of judge configs for a given specific_topic.
    Number of judges is inferred from the dataset at runtime.
    Each dict: judge_id, judge_name, general_topic, general_belief.
    """
    matching = persona_df[persona_df["specific_topic"] == topic]
    if matching.empty:
        print(f"[WARNING] No personas found for topic: '{topic}'")
        return None

    row = matching.iloc[0]
    judges = []
    for i in range(1, num_judges + 1):
        judges.append({
            "judge_id": f"Judge_{i}",
            "judge_name": row[f"judge_name_{i}"],
            "general_topic": row["general_topic"],
            "general_belief": row[f"general_belief_{i}"],
        })
    return judges

# =============================================================================
# ERROR / PARSING LOGGING
# =============================================================================

def log_error_to_csv(voting_id: str, debate_id: str, topic: str,
                     judge_id: str, phase: str, error_message: str,
                     output_dir: str) -> None:
    """Append a parsing error entry to error_log.csv."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    error_log_path = Path(output_dir) / "error_log.csv"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "voting_id": voting_id,
        "debate_id": debate_id,
        "topic": topic,
        "judge_id": judge_id,
        "phase": phase,
        "error_message": error_message,
    }
    df = pd.DataFrame([entry])
    if error_log_path.exists():
        df.to_csv(error_log_path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(error_log_path, mode="w", header=True, index=False, encoding="utf-8")
    print(f"[ERROR LOGGED] {error_log_path}")


def log_llm_parsing_usage(voting_id: str, judge_id: str, phase: str,
                          original_response: str, llm_parsed_result: Dict,
                          output_dir: str) -> None:
    """Log LLM fallback parsing to CSV + individual JSON."""
    log_path = Path(output_dir) / "llm_parsing_log.csv"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "voting_id": voting_id,
        "judge_id": judge_id,
        "phase": phase,
        "success": llm_parsed_result is not None,
    }
    df = pd.DataFrame([entry])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(log_path, mode="w", header=True, index=False, encoding="utf-8")

    parsing_dir = Path(output_dir) / "llm_parsing_results"
    parsing_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "timestamp": datetime.now().isoformat(),
        "voting_id": voting_id,
        "judge_id": judge_id,
        "phase": phase,
        "original_response": original_response,
        "llm_parsed_result": llm_parsed_result,
        "model_used": PARSING_FALLBACK_MODEL,
    }
    json_path = parsing_dir / f"{voting_id}_{judge_id}_{phase}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"[LLM PARSING] Fallback used for {judge_id}/{phase} → {json_path}")


def _upsert_errors_summary(entry: Dict, output_dir: str) -> None:
    """
    Internal helper: upsert one entry into errors_summary.json,
    keyed by voting_id.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(output_dir) / "errors_summary.json"

    existing: List[Dict] = []
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing = [e for e in existing if e.get("voting_id") != entry["voting_id"]]
    existing.append(entry)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"[ERRORS SUMMARY] Updated: {summary_path}")


def log_abort_to_errors_summary(voting_id: str, debate_id: str, topic: str,
                                judge_id: str, phase: str, output_dir: str) -> None:
    """
    Record an API-abort event in errors_summary.json.
    Called immediately before returning None from execute_single_voting.
    """
    entry = {
        "voting_id": voting_id,
        "debate_id": debate_id,
        "topic": topic,
        "error_type": "api_abort",
        "num_errors": 1,
        "instances": [
            {
                "judge_id": judge_id,
                "phase": phase,
                "error_message": "API call failed after max retries — debate aborted",
                "timestamp": datetime.now().isoformat(),
            }
        ],
    }
    _upsert_errors_summary(entry, output_dir)


def update_errors_summary(voting: Dict, output_dir: str) -> None:
    """
    Record all parsing errors from a completed voting in errors_summary.json.
    Only called when voting["has_errors"] is True.
    """
    if not voting.get("has_errors"):
        return

    entry = {
        "voting_id": voting["voting_id"],
        "debate_id": voting["debate_id"],
        "topic": voting["topic"],
        "error_type": "parsing_errors",
        "num_errors": len(voting.get("errors", [])),
        "instances": [
            {
                "judge_id": e["judge_id"],
                "phase": e["phase"],
                "error_message": e["error"],
                "timestamp": voting["metadata"]["timestamp"],
            }
            for e in voting.get("errors", [])
        ],
    }
    _upsert_errors_summary(entry, output_dir)

# =============================================================================
# API CALL
# =============================================================================

def strip_markdown_bold(text: str) -> str:
    """Strip ** markers before regex parsing (Sonnet often wraps labels in bold)."""
    return text.replace("**", "")


def send_to_judge(user_prompt: str, system_prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call Claude API. Returns response text or None on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            response = claude_client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=10000,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            if attempt < max_retries:
                sleep_time = 2 * attempt
                print(f"[WARN] API error (attempt {attempt}/{max_retries}): {e}")
                print(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"[ERROR] Failed after {max_retries} attempts: {e}")
                return None
    return None

# =============================================================================
# LLM FALLBACK PARSING HELPERS
# =============================================================================

def _call_parsing_llm(prompt: str) -> Optional[str]:
    """Call Haiku to parse a response. Returns raw JSON string or None."""
    try:
        resp = claude_client.messages.create(
            model=PARSING_FALLBACK_MODEL,
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text
    except Exception as e:
        print(f"[LLM PARSING ERROR] {e}")
        return None


def _validate_statement(val) -> Optional[str]:
    return val if val in ["Statement 1", "Statement 2"] else None


def _validate_confidence(val) -> Optional[int]:
    if val is None:
        return None
    try:
        c = int(val)
        return c if 0 <= c <= 100 else None
    except (ValueError, TypeError):
        return None


def llm_parse_statement_confidence(response: str, phase_label: str) -> Optional[Dict]:
    """LLM fallback to extract statement + confidence from any phase response."""
    if not ENABLE_LLM_PARSING_FALLBACK:
        return None
    prompt = f"""Parse the following {phase_label} response to extract the statement choice and confidence level.

Response to parse:
<response>
{response}
</response>

Return ONLY a JSON object with this exact format (no other text):
{{
    "statement": "Statement 1" or "Statement 2",
    "confidence": <number between 0 and 100>
}}
Use null if a field cannot be determined."""
    text = _call_parsing_llm(prompt)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return {
            "statement": _validate_statement(parsed.get("statement")),
            "confidence": _validate_confidence(parsed.get("confidence")),
        }
    except Exception as e:
        print(f"[LLM PARSING ERROR] JSON parse failed: {e}")
        return None

# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def _normalize_statement(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    if val in ["Statement 1", "Statement 2"]:
        return val
    m = re.search(r"[12]", val)
    return f"Statement {m.group(0)}" if m else None


def _parse_statement_confidence(response: str, voting_id: str, judge_id: str,
                                 phase: str, output_dir: str) -> Dict:
    """
    Shared regex parser for any phase that asks only for Statement + Confidence.
    Falls back to LLM if regex fails.
    """
    result = {"statement": None, "confidence": None, "response_raw": response}
    clean = strip_markdown_bold(response)

    m = re.search(r"^Statement:\s*(Statement\s*[12])", clean, re.MULTILINE | re.IGNORECASE)
    if not m:
        m = re.search(r"Statement:\s*(Statement\s*[12])", clean, re.IGNORECASE)
    if m:
        result["statement"] = m.group(1).strip()

    m = re.search(r"^Confidence:\s*(\d+)", clean, re.MULTILINE | re.IGNORECASE)
    if not m:
        m = re.search(r"Confidence:\s*(\d+)", clean, re.IGNORECASE)
    if m:
        try:
            result["confidence"] = int(m.group(1))
        except ValueError:
            pass

    if result["statement"] is None or result["confidence"] is None:
        print(f"[WARNING] Regex parsing incomplete for {phase}, trying LLM fallback...")
        llm = llm_parse_statement_confidence(response, phase)
        if llm:
            if result["statement"] is None:
                result["statement"] = llm.get("statement")
            if result["confidence"] is None:
                result["confidence"] = llm.get("confidence")
            if voting_id and judge_id and output_dir:
                log_llm_parsing_usage(voting_id, judge_id, phase, response, llm, output_dir)

    result["statement"] = _normalize_statement(result["statement"])
    if result["confidence"] is not None and not (0 <= result["confidence"] <= 100):
        result["confidence"] = None

    return result


def parse_prior_belief(response: str, voting_id: str = None,
                       judge_id: str = None, output_dir: str = None) -> Dict:
    """Parse Phase 1 response: Statement + Confidence."""
    return _parse_statement_confidence(response, voting_id, judge_id, "prior_belief", output_dir)


def parse_vote(response: str, voting_id: str = None,
               judge_id: str = None, output_dir: str = None) -> Dict:
    """Parse Phase 2 response: Statement + Confidence."""
    return _parse_statement_confidence(response, voting_id, judge_id, "vote", output_dir)

# =============================================================================
# TRANSCRIPT FORMATTING
# =============================================================================

def format_debate_transcript(transcript: List[Dict]) -> str:
    """Format debate JSON transcript for judge prompts."""
    return "\n\n".join(f"{entry['debater']}: {entry['argument']}" for entry in transcript)

# =============================================================================
# DEBATE LOADING
# =============================================================================

def load_debate_json(json_path: Path) -> Optional[Dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load {json_path}: {e}")
        return None

# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_voting_json(voting: Dict, output_dir: str) -> Path:
    """Save (overwrite) voting as JSON. Called after each phase."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / f"{voting['voting_id']}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(voting, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Saved: {filepath}")
    return filepath


def _safe_phase(phase_list: List[Dict], idx: int) -> Dict:
    """Return entry at idx if the phase list is long enough, else empty dict."""
    return phase_list[idx] if idx < len(phase_list) else {}


def _build_csv_row(voting: Dict) -> Dict:
    """
    Build a flat CSV row from the current voting state.
    Phases not yet completed appear as None. This allows progressive saving:
    the row is written after each phase and overwritten when the next phase
    completes, with no schema change across saves.
    """
    cd = voting.get("collective_decision")
    n = voting["metadata"]["num_judges"]

    row = {
        "voting_id": voting["voting_id"],
        "debate_id": voting["debate_id"],
        "topic": voting["topic"],
        "general_topic": voting["general_topic"],
        "statement_1": voting["statement_1"],
        "statement_2": voting["statement_2"],
        "num_judges": n,
        "has_errors": voting["has_errors"],
        "num_errors": len(voting.get("errors", [])),
        "collective_final_statement": cd["final_statement"] if cd else None,
        "collective_unanimous": cd["unanimous"] if cd else None,
        "collective_vote_statement_1": cd["vote_count"].get("Statement 1", 0) if cd else None,
        "collective_vote_statement_2": cd["vote_count"].get("Statement 2", 0) if cd else None,
        "collective_ground_truth_alignment": (
            (cd["final_statement"] == "Statement 1")
            if cd and cd["final_statement"] else None
        ),
    }

    prior_list = voting["phases"]["prior_beliefs"]
    vote_list = voting["phases"]["votes"]

    for i, judge_info in enumerate(voting["judges"]):
        j = f"j{i + 1}"
        prior = _safe_phase(prior_list, i)
        vote = _safe_phase(vote_list, i)

        row[f"{j}_judge_name"] = judge_info["judge_name"]
        row[f"{j}_general_belief"] = judge_info["general_belief"]

        # Phase 1 — prior belief
        row[f"{j}_prior_statement"] = prior.get("statement")
        row[f"{j}_prior_confidence"] = prior.get("confidence")
        row[f"{j}_prior_ground_truth_alignment"] = prior.get("ground_truth_alignment")
        row[f"{j}_prior_response_raw"] = prior.get("response_raw")

        # Phase 2 — vote (post-debate)
        row[f"{j}_vote_statement"] = vote.get("statement")
        row[f"{j}_vote_confidence"] = vote.get("confidence")
        row[f"{j}_vote_ground_truth_alignment"] = vote.get("ground_truth_alignment")
        row[f"{j}_vote_response_raw"] = vote.get("response_raw")

        # Belief change flag (only available after Phase 2 metrics computation)
        row[f"{j}_belief_changed"] = prior.get("belief_changed")

        # Confidence delta
        p_conf = prior.get("confidence")
        v_conf = vote.get("confidence")
        row[f"{j}_confidence_change"] = (
            v_conf - p_conf if v_conf is not None and p_conf is not None else None
        )

    return row


def upsert_voting_to_csv(voting: Dict, output_dir: str, csv_filename: str) -> Path:
    """
    Write or update this voting's row in the CSV.
    Reads the existing file, removes the old row for this voting_id if present,
    then writes the updated row. This allows re-saving after each phase without
    creating duplicate rows.
    """
    row = _build_csv_row(voting)
    df_new = pd.DataFrame([row])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / csv_filename

    if filepath.exists():
        df_existing = pd.read_csv(filepath, dtype=str)
        df_existing = df_existing[
            df_existing["voting_id"] != voting["voting_id"]
        ]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filepath, index=False, encoding="utf-8")
        print(f"[CSV] Updated row in {filepath}")
    else:
        df_new.to_csv(filepath, index=False, encoding="utf-8")
        print(f"[CSV] Created {filepath}")

    return filepath


def save_progress(voting: Dict, output_dir: str, csv_filename: str) -> None:
    """
    Persist the current state of a voting after each phase completes.
    Overwrites the JSON and upserts the CSV row.
    """
    save_voting_json(voting, output_dir)
    upsert_voting_to_csv(voting, output_dir, csv_filename)

# =============================================================================
# MAIN VOTING EXECUTION
# =============================================================================

def execute_single_voting(debate: Dict, persona_df: pd.DataFrame,
                          num_judges: int, voting_number: int,
                          output_dir: str, csv_filename: str) -> Optional[Dict]:
    """
    Run 2-phase voting for one debate.
    - JSON and CSV are saved after each phase (all judges in that phase responded).
    - Returns None (and logs the abort) if any API call fails.
    - Parsing failures are logged but do not abort.
    """
    topic = debate["topic"]
    debate_id = debate["debate_id"]

    print(f"\n{'='*70}")
    print(f"VOTING #{voting_number}")
    print(f"Debate ID: {debate_id}")
    print(f"Topic:     {topic}")
    print(f"{'='*70}")

    judges = get_judges_for_topic(topic, persona_df, num_judges)
    if not judges:
        print(f"[SKIP] No personas found for topic: '{topic}'")
        return None

    n = len(judges)
    voting_id = f"voting_{voting_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    debate_transcript_formatted = format_debate_transcript(debate["transcript"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    voting = {
        "voting_id": voting_id,
        "debate_id": debate_id,
        "topic": topic,
        "general_topic": judges[0]["general_topic"],
        "statement_1": debate["statement_1"],
        "statement_2": debate["statement_2"],
        "judges": [
            {"judge_id": j["judge_id"], "judge_name": j["judge_name"],
             "general_belief": j["general_belief"]}
            for j in judges
        ],
        "phases": {
            "prior_beliefs": [],
            "votes": [],
        },
        "collective_decision": None,
        "has_errors": False,
        "errors": [],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_judges": n,
            "judge_model": JUDGE_MODEL,
        },
    }

    # ==========================================================================
    # PHASE 1: Prior belief extraction (before seeing the debate)
    # ==========================================================================

    print(f"\nPHASE 1: Prior belief extraction ({n} judges)...")

    for judge in judges:
        jid = judge["judge_id"]
        print(f"  {jid} ({judge['judge_name']})...")

        system_prompt = SYSTEM_PROMPT_PERSONA.format(
            judge_name=judge["judge_name"],
            general_topic=judge["general_topic"],
            general_belief=judge["general_belief"],
        )
        user_prompt = PROMPT_PRE_DEBATE.format(
            specific_topic=topic,
            statement_1=debate["statement_1"],
            statement_2=debate["statement_2"],
        )

        response = send_to_judge(user_prompt, system_prompt)
        if response is None:
            print(f"[ABORT] API failure for {jid} in Phase 1 — aborting debate")
            log_abort_to_errors_summary(voting_id, debate_id, topic,
                                        jid, "phase_1_prior_belief", output_dir)
            return None

        parsed = parse_prior_belief(response, voting_id, jid, output_dir)
        if parsed["statement"] is None or parsed["confidence"] is None:
            msg = (f"Parsing failed for {jid} Phase 1 — "
                   f"statement: {parsed['statement']}, confidence: {parsed['confidence']}")
            print(f"[WARNING] {msg}")
            log_error_to_csv(voting_id, debate_id, topic, jid, "prior_belief", msg, output_dir)
            voting["has_errors"] = True
            voting["errors"].append({"judge_id": jid, "phase": "prior_belief", "error": msg})

        voting["phases"]["prior_beliefs"].append(
            {"judge_id": jid, "judge_name": judge["judge_name"], **parsed}
        )
        print(f"    → {parsed['statement']} (confidence: {parsed['confidence']})")
        time.sleep(1)

    print(f"  [SAVE] Phase 1 complete — saving progress...")
    save_progress(voting, output_dir, csv_filename)

    # ==========================================================================
    # PHASE 2: Post-debate vote
    # ==========================================================================

    print(f"\nPHASE 2: Post-debate vote ({n} judges)...")

    for judge in judges:
        jid = judge["judge_id"]
        print(f"  {jid} ({judge['judge_name']})...")

        system_prompt = SYSTEM_PROMPT_PERSONA.format(
            judge_name=judge["judge_name"],
            general_topic=judge["general_topic"],
            general_belief=judge["general_belief"],
        )
        user_prompt = PROMPT_POST_DEBATE.format(
            specific_topic=topic,
            statement_1=debate["statement_1"],
            statement_2=debate["statement_2"],
            debater_1="Debater_1",
            debater_2="Debater_2",
            assigned_statement_debater_1="Statement 1",
            assigned_statement_debater_2="Statement 2",
            transcript=debate_transcript_formatted,
        )

        response = send_to_judge(user_prompt, system_prompt)
        if response is None:
            print(f"[ABORT] API failure for {jid} in Phase 2 — aborting debate")
            log_abort_to_errors_summary(voting_id, debate_id, topic,
                                        jid, "phase_2_vote", output_dir)
            return None

        parsed = parse_vote(response, voting_id, jid, output_dir)
        if parsed["statement"] is None or parsed["confidence"] is None:
            msg = (f"Parsing failed for {jid} Phase 2 (vote) — "
                   f"statement: {parsed['statement']}, confidence: {parsed['confidence']}")
            print(f"[WARNING] {msg}")
            log_error_to_csv(voting_id, debate_id, topic, jid, "vote", msg, output_dir)
            voting["has_errors"] = True
            voting["errors"].append({"judge_id": jid, "phase": "vote", "error": msg})

        stmt = parsed.get("statement")
        parsed["ground_truth_alignment"] = (stmt == "Statement 1") if stmt else None

        voting["phases"]["votes"].append(
            {"judge_id": jid, "judge_name": judge["judge_name"], **parsed}
        )
        print(f"    → Statement: {parsed['statement']}, Confidence: {parsed['confidence']}")
        time.sleep(1)

    # ==========================================================================
    # Compute derived metrics (before final save)
    # ==========================================================================

    # Collective decision: majority vote on post-debate statements
    final_statements = [v["statement"] for v in voting["phases"]["votes"] if v["statement"]]
    if final_statements:
        counts = Counter(final_statements)
        voting["collective_decision"] = {
            "final_statement": counts.most_common(1)[0][0],
            "vote_count": dict(counts),
            "unanimous": len(set(final_statements)) == 1,
        }

    # Ground truth alignment for prior beliefs
    for entry in voting["phases"]["prior_beliefs"]:
        stmt = entry.get("statement")
        entry["ground_truth_alignment"] = (stmt == "Statement 1") if stmt else None

    # Belief change flag stored on prior_beliefs entries for CSV access
    prior_list = voting["phases"]["prior_beliefs"]
    vote_list = voting["phases"]["votes"]

    for i in range(n):
        prior_stmt = prior_list[i]["statement"]
        vote_stmt = vote_list[i]["statement"]
        prior_list[i]["belief_changed"] = (
            (prior_stmt != vote_stmt) if prior_stmt and vote_stmt else None
        )

    print(f"  [SAVE] Phase 2 complete — saving final state...")
    save_progress(voting, output_dir, csv_filename)

    print(f"\n[SUCCESS] Voting completed")
    print(f"Collective decision: {voting['collective_decision']}")
    if voting["has_errors"]:
        print(f"[WARNING] Completed with {len(voting['errors'])} parsing error(s)")

    return voting

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_all_votings(debates_dir: str, persona_dataset_path: str,
                    output_dir: str, csv_filename: str,
                    n_votings: Optional[int] = None) -> List[Dict]:
    """
    Run votings for all (or n_votings) debates.
    Number of judges is detected from the dataset columns at startup.
    JSON and CSV are saved progressively after each phase inside
    execute_single_voting — no separate save needed here.
    errors_summary.json is updated after each voting with parsing errors.
    """
    persona_df = load_persona_dataset(persona_dataset_path)
    if persona_df is None:
        return []

    num_judges = detect_num_judges(persona_df)
    if num_judges < 1:
        print("[ERROR] Could not detect any judge columns (expected judge_name_1, ...)")
        return []
    print(f"[PERSONA] Detected {num_judges} judges per topic")

    debates_path = Path(debates_dir)
    if not debates_path.exists():
        print(f"[ERROR] Debates directory not found: {debates_dir}")
        return []

    debate_files = sorted(debates_path.glob("debate_*.json"))
    if not debate_files:
        print(f"[ERROR] No debate JSON files found in {debates_dir}")
        return []

    if n_votings is not None:
        debate_files = debate_files[:n_votings]

    # Timestamp-stamped CSV so runs don't overwrite each other
    csv_ts = f"{Path(csv_filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"\n{'#'*70}")
    print(f"VOTING EXPERIMENT — NEW VERSION")
    print(f"{'#'*70}")
    print(f"Debates to process: {len(debate_files)}")
    print(f"Judge model:        {JUDGE_MODEL}")
    print(f"Judges per debate:  {num_judges}")
    print(f"Output directory:   {output_dir}")
    print(f"CSV file:           {csv_ts}")
    print(f"LLM fallback:       {'ENABLED' if ENABLE_LLM_PARSING_FALLBACK else 'DISABLED'}")
    print(f"{'#'*70}\n")

    completed = []
    skipped = 0

    for idx, debate_file in enumerate(debate_files, start=1):
        try:
            debate = load_debate_json(debate_file)
            if not debate:
                print(f"[WARNING] Skipping {debate_file.name} — could not load")
                skipped += 1
                continue

            result = execute_single_voting(
                debate, persona_df, num_judges, idx, output_dir, csv_ts
            )

            if result is None:
                print(f"[SKIP] Voting {idx} aborted (see errors_summary.json)")
                skipped += 1
            else:
                update_errors_summary(result, output_dir)
                completed.append(result)
                print(f"[DONE] Voting {idx} complete\n")

        except Exception as e:
            print(f"[ERROR] Unexpected exception in voting {idx}: {e}")
            traceback.print_exc()
            skipped += 1

        if idx < len(debate_files):
            time.sleep(3)

    # Final summary
    print(f"\n{'='*70}")
    print("ALL VOTINGS COMPLETED")
    print(f"Completed: {len(completed)} / {len(debate_files)}   Skipped/Aborted: {skipped}")
    print(f"Output: {output_dir}/")
    print(f"{'='*70}")

    if completed:
        total_judge_evals = sum(v["metadata"]["num_judges"] for v in completed)

        belief_changes = sum(
            1 for v in completed
            for pb in v["phases"]["prior_beliefs"]
            if pb.get("belief_changed") is True
        )
        unanimous = sum(
            1 for v in completed
            if v["collective_decision"] and v["collective_decision"]["unanimous"]
        )
        correct_prior = sum(
            1 for v in completed
            for pb in v["phases"]["prior_beliefs"]
            if pb.get("ground_truth_alignment") is True
        )
        correct_vote = sum(
            1 for v in completed
            for vote in v["phases"]["votes"]
            if vote.get("ground_truth_alignment") is True
        )
        error_votings = sum(1 for v in completed if v["has_errors"])

        print(f"\nSUMMARY STATISTICS:")
        print(f"Total judge evaluations:           {total_judge_evals}")
        print(f"Belief changes (prior → vote):     {belief_changes}/{total_judge_evals} ({belief_changes/total_judge_evals*100:.1f}%)")
        print(f"Unanimous collective decisions:    {unanimous}/{len(completed)} ({unanimous/len(completed)*100:.1f}%)")
        print(f"Ground truth alignment (prior):    {correct_prior}/{total_judge_evals} ({correct_prior/total_judge_evals*100:.1f}%)")
        print(f"Ground truth alignment (vote):     {correct_vote}/{total_judge_evals} ({correct_vote/total_judge_evals*100:.1f}%)")
        if error_votings:
            print(f"Votings with parsing errors:       {error_votings}/{len(completed)}")

    return completed

# =============================================================================
# RUN CONFIGURATIONS
# =============================================================================

# Base project directory (debate_2026/) relative to this script's location
_BASE_DIR = Path(__file__).parent.parent
_DATASET_1M2S = str(Path(__file__).parent / "dataset_3_judges _1_mainstream_2_skeptical.csv")
_DATASET_2M1S = str(Path(__file__).parent / "dataset_3_judges_2_mainstream_1_skeptical.csv")

RUNS = [
    {
        "debates_dir": str(_BASE_DIR / "debates"),
        "persona_dataset_path": _DATASET_1M2S,
        "output_dir": str(_BASE_DIR / "votaciones_nuevo" / "orden_original"),
        "csv_filename": "votaciones_orden_original.csv",
    },
    {
        "debates_dir": str(_BASE_DIR / "inverted_debates"),
        "persona_dataset_path": _DATASET_1M2S,
        "output_dir": str(_BASE_DIR / "votaciones_nuevo" / "orden_invertido"),
        "csv_filename": "votaciones_orden_invertido.csv",
    },
    {
        "debates_dir": str(_BASE_DIR / "debates"),
        "persona_dataset_path": _DATASET_2M1S,
        "output_dir": str(_BASE_DIR / "votaciones_nuevo" / "2mainstream_1skeptical" / "orden_original"),
        "csv_filename": "votaciones_2mainstream_1skeptical_orden_original.csv",
    },
    {
        "debates_dir": str(_BASE_DIR / "inverted_debates"),
        "persona_dataset_path": _DATASET_2M1S,
        "output_dir": str(_BASE_DIR / "votaciones_nuevo" / "2mainstream_1skeptical" / "orden_invertido"),
        "csv_filename": "votaciones_2mainstream_1skeptical_orden_invertido.csv",
    },
]

if __name__ == "__main__":

    N_VOTINGS = None  # Set to None to process all debates

    all_results = []

    for run_idx, run_config in enumerate(RUNS, start=1):
        print(f"\n{'#'*70}")
        print(f"RUN {run_idx}/{len(RUNS)}")
        print(f"Debates: {run_config['debates_dir']}")
        print(f"Output:  {run_config['output_dir']}")
        print(f"{'#'*70}\n")

        results = run_all_votings(
            debates_dir=run_config["debates_dir"],
            persona_dataset_path=run_config["persona_dataset_path"],
            output_dir=run_config["output_dir"],
            csv_filename=run_config["csv_filename"],
            n_votings=N_VOTINGS,
        )

        if results:
            all_results.extend(results)

    print(f"\n{'#'*70}")
    print(f"ALL RUNS COMPLETE — Total votings saved: {len(all_results)}")
    print(f"{'#'*70}")
