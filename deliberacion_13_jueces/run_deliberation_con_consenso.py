"""
Multi-Judge Deliberation Experiment — Con Consenso
=====================================
Same 4-phase structure as run_deliberation_sin_consenso.py but with prompts
that explicitly ask judges to reach consensus by the end of Round 3.

Key differences from sin_consenso:
- Prompts instruct judges to agree on the same statement by Round 3.
- Round 3 only asks for a Statement (no confidence or reasoning).
- Valid statements: "Statement 1", "Statement 2", or "No agreement reached".
- CSV drops r3_confidence, r3_final_reasoning, and confidence_change_r1_to_r3.
- collective_decision includes a no_agreement_count field.

The number of judges and persona distribution are determined dynamically from
the dataset CSV — they are NOT hardcoded. The CSV must have columns:
  general_topic, specific_topic, statement_1, statement_2,
  judge_name_1, general_belief_1, [judge_name_2, general_belief_2, ...]

Each debate runs 4 phases:
  Phase 1: Prior belief extraction (before debate)       — PROMPT_PRE_DEBATE
  Phase 2: Post-debate evaluation + Round 1 deliberation — PROMPT_POST_DEBATE
  Phase 3: Round 2 deliberation (response to others)     — PROMPT_DELIBERATION_ROUND_2
  Phase 4: Round 3 final deliberation                    — PROMPT_DELIBERATION_ROUND_3

Saving strategy:
  - JSON and CSV are saved/updated after EACH phase (when all judges in that phase
    have responded). Incomplete phases appear as None in the CSV.
  - If the script is interrupted mid-phase (a judge is still pending), only the
    state at the END of the previous completed phase is persisted.

Error policy:
  - API failure → abort that debate, log to errors_summary.json, save nothing more.
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
from concurrent.futures import ThreadPoolExecutor
import threading

from dotenv import load_dotenv
from sklearn.utils import shuffle

# Guards concurrent CSV appends from log_error_to_csv / log_llm_parsing_usage
# when judges within a phase run in parallel.
_csv_lock = threading.Lock()

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

N_JUDGES = 13  # Set to 3 or 13

# =============================================================================
# PROMPTS
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

After providing your individual evaluation, you will deliberate with {number_other_judges} other judges over 3 rounds to reach a collective decision. At the end of the third round of deliberation, you should all agree on the same statement.

In this first round, you should provide your initial argument to share with the other judges.

Now, you should provide your evaluation. Think step by step and analyze the key arguments presented by both debaters, assessing their strengths and weaknesses.

First, explain your reasoning. Then, provide your answers in the following format:

2. Based on the arguments presented, which statement do you think is correct?
Statement: [Statement 1 / Statement 2]

3. How confident are you in your statement choice?
Confidence: [0-100]

4. What are the key points you want to raise for deliberation with the other judges?
Key points: [Your main arguments or concerns to discuss]
"""

PROMPT_DELIBERATION_ROUND_2 = """You are acting as an expert judge in a debate where two debaters are arguing over which of their two opposing statements is correct about a certain topic. Your task is to evaluate both the quality of the arguments presented and determine which statement you believe is correct.

You must take into account your prior belief as defined in the system prompt when making your judgment. Your evaluation should be based only on this prior belief and the arguments presented in the debate and the deliberation with other judges. Do not rely on your general knowledge or external information beyond these inputs. If the debate and deliberation provides sufficiently strong reasons to revise your prior belief, you may update your judgment accordingly.

The following statements are about {specific_topic}:

Statement 1: {statement_1}

Statement 2: {statement_2}

{debater_1} is arguing for {assigned_statement_debater_1} and {debater_2} is arguing for {assigned_statement_debater_2}.

Here is the transcript of the debate:

<transcript>
{transcript}
</transcript>

At this stage, you are deliberating with {number_other_judges} other judges over 3 rounds to reach a collective decision. At the end of the third round of deliberation, you should all agree on the same statement. This is round 2 of 3.
Here are the arguments from all judges in the previous round of deliberation:
<deliberation_transcript>
{deliberation_transcript}
</deliberation_transcript>

Consider the other judges' perspectives and arguments carefully. Engage constructively with the other judges' points. Identify areas of agreement and disagreement, and explain your reasoning.

Provide your response in the following format:
1. Response to other judges' arguments:
[Address specific points raised by other judges. Do you agree or disagree? Why?]
2. What remains unresolved or what key points should be addressed in the final round?
Outstanding issues: [Points that need further discussion]
"""

PROMPT_DELIBERATION_ROUND_3 = """You are acting as an expert judge in a debate where two debaters are arguing over which of their two opposing statements is correct about a certain topic. Your task is to evaluate both the quality of the arguments presented and determine which statement you believe is correct.

You must take into account your prior belief as defined in the system prompt when making your judgment. Your evaluation should be based only on this prior belief and the arguments presented in the debate and the deliberation with other judges. Do not rely on your general knowledge or external information beyond these inputs. If the debate and deliberation provides sufficiently strong reasons to revise your prior belief, you may update your judgment accordingly.

The following statements are about {specific_topic}:

Statement 1: {statement_1}

Statement 2: {statement_2}

{debater_1} is arguing for {assigned_statement_debater_1} and {debater_2} is arguing for {assigned_statement_debater_2}.

Here is the transcript of the debate:

<transcript>
{transcript}
</transcript>

At this stage, you are deliberating with {number_other_judges} other judges over 3 rounds to reach a collective decision. At the end of the third round of deliberation, you should all agree on the same statement. This is the third and final round of deliberation.
Here is the full transcript of the deliberation across both previous rounds:
<deliberation_transcript>
{deliberation_transcript}
</deliberation_transcript>

After considering all arguments from the debate and from your fellow judges, you should now agree on the same statement, considering
- The original arguments from the debate
- The perspectives raised by all judges during deliberation
Provide your final response in the following format:
Based on the deliberation, what was the agreement between all judges?
Statement: [Statement 1 / Statement 2 / No agreement reached]
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


def get_majority_composition(n_judges: int, majority_type: str) -> tuple:
    """
    Return (n_mainstream, n_skeptical) for the requested majority type.
    For n=3: mainstream→(2,1), skeptical→(1,2).
    For n=13: mainstream→(7,6), skeptical→(6,7).
    """
    if majority_type == "mainstream":
        n_mainstream = n_judges // 2 + 1
        n_skeptical = n_judges - n_mainstream
    else:
        n_skeptical = n_judges // 2 + 1
        n_mainstream = n_judges - n_skeptical
    return n_mainstream, n_skeptical


def extract_canonical_debate_number(debate_id: str) -> int:
    """
    Extract the canonical debate number from debate IDs like:
    debate_001_20260325_201124 or debate_001_20260325_201124_inverted.
    """
    match = re.match(r"^debate_(\d+)_", debate_id)
    if not match:
        raise ValueError(f"Could not extract debate number from debate_id: {debate_id}")
    return int(match.group(1))


def get_judges_for_topic(topic: str, persona_df: pd.DataFrame,
                         n_judges: int, majority_type: str,
                         shuffle_seed: Optional[int] = None) -> Optional[List[Dict]]:
    """
    Build judge list for a topic using the new dataset format (general_belief_mainstream /
    general_belief_skeptical columns). Mainstream/skeptical slots are assigned to judge
    positions in a seeded random order to prevent position bias, while honouring the
    majority composition required by majority_type. The seed is the canonical debate
    number so original and inverted versions of the same debate share judge order.
    """
    matching = persona_df[persona_df["specific_topic"] == topic]
    if matching.empty:
        print(f"[WARNING] No personas found for topic: '{topic}'")
        return None

    row = matching.iloc[0]
    n_mainstream, n_skeptical = get_majority_composition(n_judges, majority_type)

    assignments = (
        [("mainstream", row["general_belief_mainstream"])] * n_mainstream +
        [("skeptical", row["general_belief_skeptical"])] * n_skeptical
    )
    assignments = shuffle(assignments, random_state=shuffle_seed)

    judges = []
    for i, (belief_type, belief_text) in enumerate(assignments, 1):
        judges.append({
            "judge_id": f"Judge_{i}",
            "judge_name": f"Judge {i}",
            "general_topic": row["general_topic"],
            "general_belief": belief_text,
            "belief_type": belief_type,
        })
    return judges

# =============================================================================
# ERROR / PARSING LOGGING
# =============================================================================

def log_error_to_csv(deliberation_id: str, debate_id: str, topic: str,
                     judge_id: str, phase: str, error_message: str,
                     output_dir: str) -> None:
    """Append a parsing error entry to error_log.csv."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    error_log_path = Path(output_dir) / "error_log.csv"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "deliberation_id": deliberation_id,
        "debate_id": debate_id,
        "topic": topic,
        "judge_id": judge_id,
        "phase": phase,
        "error_message": error_message,
    }
    df = pd.DataFrame([entry])
    with _csv_lock:
        if error_log_path.exists():
            df.to_csv(error_log_path, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df.to_csv(error_log_path, mode="w", header=True, index=False, encoding="utf-8")
    print(f"[ERROR LOGGED] {error_log_path}")


def log_llm_parsing_usage(deliberation_id: str, judge_id: str, phase: str,
                          original_response: str, llm_parsed_result: Dict,
                          output_dir: str) -> None:
    """Log LLM fallback parsing to CSV + individual JSON."""
    log_path = Path(output_dir) / "llm_parsing_log.csv"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "deliberation_id": deliberation_id,
        "judge_id": judge_id,
        "phase": phase,
        "success": llm_parsed_result is not None,
    }
    df = pd.DataFrame([entry])
    with _csv_lock:
        if log_path.exists():
            df.to_csv(log_path, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df.to_csv(log_path, mode="w", header=True, index=False, encoding="utf-8")

    parsing_dir = Path(output_dir) / "llm_parsing_results"
    parsing_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "timestamp": datetime.now().isoformat(),
        "deliberation_id": deliberation_id,
        "judge_id": judge_id,
        "phase": phase,
        "original_response": original_response,
        "llm_parsed_result": llm_parsed_result,
        "model_used": PARSING_FALLBACK_MODEL,
    }
    json_path = parsing_dir / f"{deliberation_id}_{judge_id}_{phase}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"[LLM PARSING] Fallback used for {judge_id}/{phase} → {json_path}")


def _upsert_errors_summary(entry: Dict, output_dir: str) -> None:
    """
    Internal helper: upsert one entry into errors_summary.json,
    keyed by deliberation_id.
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

    # Remove any previous entry for this deliberation_id (upsert)
    existing = [e for e in existing if e.get("deliberation_id") != entry["deliberation_id"]]
    existing.append(entry)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"[ERRORS SUMMARY] Updated: {summary_path}")


def log_abort_to_errors_summary(deliberation_id: str, debate_id: str, topic: str,
                                 judge_id: str, phase: str, output_dir: str) -> None:
    """
    Record an API-abort event in errors_summary.json.
    Called immediately before returning None from execute_single_deliberation.
    """
    entry = {
        "deliberation_id": deliberation_id,
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


def update_errors_summary(deliberation: Dict, output_dir: str) -> None:
    """
    Record all parsing errors from a completed deliberation in errors_summary.json.
    Only called when deliberation["has_errors"] is True.
    """
    if not deliberation.get("has_errors"):
        return

    entry = {
        "deliberation_id": deliberation["deliberation_id"],
        "debate_id": deliberation["debate_id"],
        "topic": deliberation["topic"],
        "error_type": "parsing_errors",
        "num_errors": len(deliberation.get("errors", [])),
        "instances": [
            {
                "judge_id": e["judge_id"],
                "phase": e["phase"],
                "error_message": e["error"],
                "timestamp": deliberation["metadata"]["timestamp"],
            }
            for e in deliberation.get("errors", [])
        ],
    }
    _upsert_errors_summary(entry, output_dir)

# =============================================================================
# API CALL
# =============================================================================

def strip_markdown_bold(text: str) -> str:
    """Strip ** markers before regex parsing (Sonnet often wraps labels in bold)."""
    return text.replace("**", "")


def send_to_judge(user_prompt: str, system_prompt: str, max_retries: int = 3,
                  cache: bool = False) -> Optional[str]:
    """Call Claude API. Returns response text or None on failure.

    When cache=True the user_prompt is sent as a cacheable content block, so the
    13 judges within a phase share the prefix (debate transcript + deliberation
    transcript). Cache TTL is 5 min — well above the per-debate runtime.
    """
    if cache:
        user_content = [{"type": "text", "text": user_prompt,
                         "cache_control": {"type": "ephemeral"}}]
    else:
        user_content = user_prompt

    for attempt in range(1, max_retries + 1):
        try:
            response = claude_client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=10000,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
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
    return val if val in ["Statement 1", "Statement 2", "No agreement reached"] else None


def _validate_confidence(val) -> Optional[int]:
    if val is None:
        return None
    try:
        c = int(val)
        return c if 0 <= c <= 100 else None
    except (ValueError, TypeError):
        return None


def llm_parse_prior_belief(response: str) -> Optional[Dict]:
    if not ENABLE_LLM_PARSING_FALLBACK:
        return None
    prompt = f"""Parse the following response to extract the statement choice and confidence level.

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


def llm_parse_round_1(response: str) -> Optional[Dict]:
    if not ENABLE_LLM_PARSING_FALLBACK:
        return None
    prompt = f"""Parse the following judge Round 1 response to extract statement, confidence, and key points.

Response to parse:
<response>
{response}
</response>

Return ONLY a JSON object with this exact format (no other text):
{{
    "statement": "Statement 1" or "Statement 2",
    "confidence": <number between 0 and 100>,
    "key_points": "text of key points"
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
            "key_points": parsed.get("key_points"),
        }
    except Exception as e:
        print(f"[LLM PARSING ERROR] JSON parse failed: {e}")
        return None


def llm_parse_round_2(response: str) -> Optional[Dict]:
    if not ENABLE_LLM_PARSING_FALLBACK:
        return None
    prompt = f"""Parse the following judge Round 2 deliberation response to extract the response to other judges and outstanding issues.

Response to parse:
<response>
{response}
</response>

Return ONLY a JSON object with this exact format (no other text):
{{
    "response_to_others": "text of response to other judges",
    "outstanding_issues": "text of outstanding issues"
}}
Use null if a field cannot be determined."""
    text = _call_parsing_llm(prompt)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return {
            "response_to_others": parsed.get("response_to_others"),
            "outstanding_issues": parsed.get("outstanding_issues"),
        }
    except Exception as e:
        print(f"[LLM PARSING ERROR] JSON parse failed: {e}")
        return None


def llm_parse_round_3(response: str) -> Optional[Dict]:
    if not ENABLE_LLM_PARSING_FALLBACK:
        return None
    prompt = f"""Parse the following judge final deliberation response to extract the agreed statement.

Response to parse:
<response>
{response}
</response>

Return ONLY a JSON object with this exact format (no other text):
{{
    "statement": "Statement 1" or "Statement 2" or "No agreement reached"
}}
Use null if the field cannot be determined."""
    text = _call_parsing_llm(prompt)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return {
            "statement": _validate_statement(parsed.get("statement")),
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
    if val in ["Statement 1", "Statement 2", "No agreement reached"]:
        return val
    if "no agreement" in val.lower():
        return "No agreement reached"
    m = re.search(r"[12]", val)
    return f"Statement {m.group(0)}" if m else None


def parse_prior_belief(response: str, deliberation_id: str = None,
                       judge_id: str = None, output_dir: str = None) -> Dict:
    """Parse Phase 1 response: Statement + Confidence."""
    result = {"statement": None, "confidence": None, "response_raw": response}
    clean = strip_markdown_bold(response)

    m = re.search(r"Statement:\s*(Statement\s*[12])", clean, re.IGNORECASE)
    if m:
        result["statement"] = m.group(1).strip()

    m = re.search(r"Confidence:\s*(\d+)", clean, re.IGNORECASE)
    if m:
        try:
            result["confidence"] = int(m.group(1))
        except ValueError:
            pass

    if result["statement"] is None or result["confidence"] is None:
        print("[WARNING] Regex parsing incomplete for prior belief, trying LLM fallback...")
        llm = llm_parse_prior_belief(response)
        if llm:
            if result["statement"] is None:
                result["statement"] = llm.get("statement")
            if result["confidence"] is None:
                result["confidence"] = llm.get("confidence")
            if deliberation_id and judge_id and output_dir:
                log_llm_parsing_usage(deliberation_id, judge_id, "prior_belief",
                                      response, llm, output_dir)

    result["statement"] = _normalize_statement(result["statement"])
    if result["confidence"] is not None and not (0 <= result["confidence"] <= 100):
        result["confidence"] = None

    return result


def parse_round_1(response: str, deliberation_id: str = None,
                  judge_id: str = None, output_dir: str = None) -> Dict:
    """Parse Phase 2 (post-debate + Round 1): Statement + Confidence + Key points."""
    result = {"statement": None, "confidence": None, "key_points": None, "response_raw": response}
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

    m = re.search(r"Key points:\s*(.+?)(?=\n\d+\.|$)", clean, re.IGNORECASE | re.DOTALL)
    if m:
        result["key_points"] = m.group(1).strip()

    if result["statement"] is None or result["confidence"] is None:
        print("[WARNING] Regex parsing incomplete for Round 1, trying LLM fallback...")
        llm = llm_parse_round_1(response)
        if llm:
            if result["statement"] is None:
                result["statement"] = llm.get("statement")
            if result["confidence"] is None:
                result["confidence"] = llm.get("confidence")
            if result["key_points"] is None:
                result["key_points"] = llm.get("key_points")
            if deliberation_id and judge_id and output_dir:
                log_llm_parsing_usage(deliberation_id, judge_id, "round_1",
                                      response, llm, output_dir)

    result["statement"] = _normalize_statement(result["statement"])
    if result["confidence"] is not None and not (0 <= result["confidence"] <= 100):
        result["confidence"] = None

    return result


def parse_round_2(response: str, deliberation_id: str = None,
                  judge_id: str = None, output_dir: str = None) -> Dict:
    """Parse Phase 3 (Round 2): response_to_others + outstanding_issues.
    This round does not ask for Statement/Confidence — only engagement with other judges."""
    result = {"response_to_others": None, "outstanding_issues": None, "response_raw": response}
    clean = strip_markdown_bold(response)

    m = re.search(r"Response to other judges' arguments:\s*(.+?)(?=\n\d+\.|$)",
                  clean, re.IGNORECASE | re.DOTALL)
    if m:
        result["response_to_others"] = m.group(1).strip()

    m = re.search(r"Outstanding issues:\s*(.+?)(?=\n\d+\.|$)",
                  clean, re.IGNORECASE | re.DOTALL)
    if m:
        result["outstanding_issues"] = m.group(1).strip()

    if result["response_to_others"] is None or result["outstanding_issues"] is None:
        print("[WARNING] Regex parsing incomplete for Round 2, trying LLM fallback...")
        llm = llm_parse_round_2(response)
        if llm:
            if result["response_to_others"] is None:
                result["response_to_others"] = llm.get("response_to_others")
            if result["outstanding_issues"] is None:
                result["outstanding_issues"] = llm.get("outstanding_issues")
            if deliberation_id and judge_id and output_dir:
                log_llm_parsing_usage(deliberation_id, judge_id, "round_2",
                                      response, llm, output_dir)

    return result


def parse_round_3(response: str, deliberation_id: str = None,
                  judge_id: str = None, output_dir: str = None) -> Dict:
    """Parse Phase 4 (Round 3 final): Statement only.
    Round 3 in the con_consenso variant asks only for the agreed statement —
    no confidence or reasoning."""
    result = {"statement": None, "response_raw": response}
    clean = strip_markdown_bold(response)

    m = re.search(
        r"^Statement:\s*(Statement\s*[12]|No agreement reached)",
        clean, re.MULTILINE | re.IGNORECASE
    )
    if not m:
        m = re.search(
            r"Statement:\s*(Statement\s*[12]|No agreement reached)",
            clean, re.IGNORECASE
        )
    if m:
        result["statement"] = m.group(1).strip()

    if result["statement"] is None:
        print("[WARNING] Regex parsing incomplete for Round 3, trying LLM fallback...")
        llm = llm_parse_round_3(response)
        if llm:
            result["statement"] = llm.get("statement")
            if deliberation_id and judge_id and output_dir:
                log_llm_parsing_usage(deliberation_id, judge_id, "round_3",
                                      response, llm, output_dir)

    result["statement"] = _normalize_statement(result["statement"])
    return result

# =============================================================================
# TRANSCRIPT FORMATTING
# =============================================================================

def format_debate_transcript(transcript: List[Dict]) -> str:
    """Format debate JSON transcript for judge prompts."""
    return "\n\n".join(f"{entry['debater']}: {entry['argument']}" for entry in transcript)


def format_deliberation_transcript(responses: List[Dict]) -> str:
    """Format one round's responses into a deliberation transcript string."""
    parts = [f"{r['judge_name']}:\n{r['response_raw']}" for r in responses]
    return "\n\n---\n\n".join(parts)


def format_round1_key_points_transcript(round_1_responses: List[Dict]) -> str:
    """Format Round 1 for sharing with other judges: only key_points, no statement or confidence."""
    parts = []
    for r in round_1_responses:
        key_points = r.get("key_points") or "(no key points provided)"
        parts.append(f"{r['judge_name']}:\n{key_points}")
    return "\n\n---\n\n".join(parts)

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

def save_deliberation_json(deliberation: Dict, output_dir: str) -> Path:
    """Save (overwrite) deliberation as JSON. Called after each phase."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / f"{deliberation['deliberation_id']}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(deliberation, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Saved: {filepath}")
    return filepath


def _safe_phase(phase_list: List[Dict], idx: int) -> Dict:
    """Return entry at idx if the phase list is long enough, else empty dict."""
    return phase_list[idx] if idx < len(phase_list) else {}


def _build_csv_row(deliberation: Dict) -> Dict:
    """
    Build a flat CSV row from the current deliberation state.
    Phases not yet completed appear as None. This allows progressive saving:
    the row is written after each phase and overwritten when the next phase
    completes, with no schema change across saves.

    Round 3 (con_consenso) has no confidence or reasoning — those columns are absent.
    """
    cd = deliberation.get("collective_decision")
    n = deliberation["metadata"]["num_judges"]

    row = {
        "deliberation_id": deliberation["deliberation_id"],
        "debate_id": deliberation["debate_id"],
        "topic": deliberation["topic"],
        "general_topic": deliberation["general_topic"],
        "statement_1": deliberation["statement_1"],
        "statement_2": deliberation["statement_2"],
        "num_judges": n,
        "majority_type": deliberation.get("majority_type"),
        "has_errors": deliberation["has_errors"],
        "num_errors": len(deliberation.get("errors", [])),
        "collective_final_statement": cd["final_statement"] if cd else None,
        "collective_unanimous": cd["unanimous"] if cd else None,
        "collective_vote_statement_1": cd["vote_count"].get("Statement 1", 0) if cd else None,
        "collective_vote_statement_2": cd["vote_count"].get("Statement 2", 0) if cd else None,
        "collective_no_agreement_count": cd["vote_count"].get("No agreement reached", 0) if cd else None,
        "collective_ground_truth_alignment": (
            (cd["final_statement"] == "Statement 1")
            if cd and cd["final_statement"] and cd["final_statement"] != "No agreement reached"
            else None
        ),
    }

    prior_list = deliberation["phases"]["prior_beliefs"]
    r1_list = deliberation["phases"]["round_1"]
    r2_list = deliberation["phases"]["round_2"]
    r3_list = deliberation["phases"]["round_3"]

    for i, judge_info in enumerate(deliberation["judges"]):
        j = f"j{i + 1}"
        prior = _safe_phase(prior_list, i)
        r1 = _safe_phase(r1_list, i)
        r2 = _safe_phase(r2_list, i)
        r3 = _safe_phase(r3_list, i)

        row[f"{j}_judge_name"] = judge_info["judge_name"]
        row[f"{j}_belief_type"] = judge_info.get("belief_type")
        row[f"{j}_general_belief"] = judge_info["general_belief"]

        # Phase 1 — prior belief
        row[f"{j}_prior_statement"] = prior.get("statement")
        row[f"{j}_prior_confidence"] = prior.get("confidence")
        row[f"{j}_prior_ground_truth_alignment"] = prior.get("ground_truth_alignment")
        row[f"{j}_prior_response_raw"] = prior.get("response_raw")

        # Phase 2 — Round 1 (post-debate + initial deliberation)
        row[f"{j}_r1_statement"] = r1.get("statement")
        row[f"{j}_r1_confidence"] = r1.get("confidence")
        row[f"{j}_r1_ground_truth_alignment"] = r1.get("ground_truth_alignment")
        row[f"{j}_r1_key_points"] = r1.get("key_points")
        row[f"{j}_r1_response_raw"] = r1.get("response_raw")

        # Phase 3 — Round 2 (no statement/confidence, only qualitative engagement)
        row[f"{j}_r2_response_to_others"] = r2.get("response_to_others")
        row[f"{j}_r2_outstanding_issues"] = r2.get("outstanding_issues")
        row[f"{j}_r2_response_raw"] = r2.get("response_raw")

        # Phase 4 — Round 3 (final: statement only, no confidence or reasoning)
        row[f"{j}_r3_statement"] = r3.get("statement")
        row[f"{j}_r3_ground_truth_alignment"] = r3.get("ground_truth_alignment")
        row[f"{j}_r3_no_agreement"] = (r3.get("statement") == "No agreement reached") if r3.get("statement") else None
        row[f"{j}_r3_response_raw"] = r3.get("response_raw")

        # Belief change flags (only available after Phase 4 metrics computation)
        row[f"{j}_belief_changed_prior_to_r1"] = prior.get("belief_changed_prior_to_r1")
        row[f"{j}_belief_changed_prior_to_r3"] = prior.get("belief_changed_prior_to_r3")
        row[f"{j}_belief_changed_r1_to_r3"] = prior.get("belief_changed_r1_to_r3")

        # Confidence delta prior → r1 (r3 has no confidence in con_consenso)
        p_conf = prior.get("confidence")
        r1_conf = r1.get("confidence")
        row[f"{j}_confidence_change_prior_to_r1"] = (
            r1_conf - p_conf if r1_conf is not None and p_conf is not None else None
        )

    return row


def upsert_deliberation_to_csv(deliberation: Dict, output_dir: str, csv_filename: str) -> Path:
    """
    Write or update this deliberation's row in the CSV.
    Reads the existing file, removes the old row for this deliberation_id if present,
    then writes the updated row. This allows re-saving after each phase without
    creating duplicate rows.
    """
    row = _build_csv_row(deliberation)
    df_new = pd.DataFrame([row])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / csv_filename

    if filepath.exists():
        df_existing = pd.read_csv(filepath, dtype=str)
        # Remove previous row for this deliberation (upsert)
        df_existing = df_existing[
            df_existing["deliberation_id"] != deliberation["deliberation_id"]
        ]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filepath, index=False, encoding="utf-8")
        print(f"[CSV] Updated row in {filepath}")
    else:
        df_new.to_csv(filepath, index=False, encoding="utf-8")
        print(f"[CSV] Created {filepath}")

    return filepath


def save_progress(deliberation: Dict, output_dir: str, csv_filename: str) -> None:
    """
    Persist the current state of a deliberation after each phase completes.
    Overwrites the JSON and upserts the CSV row.
    """
    save_deliberation_json(deliberation, output_dir)
    upsert_deliberation_to_csv(deliberation, output_dir, csv_filename)

# =============================================================================
# MAIN DELIBERATION EXECUTION
# =============================================================================

def execute_single_deliberation(debate: Dict, persona_df: pd.DataFrame,
                                num_judges: int, majority_type: str,
                                deliberation_number: int,
                                output_dir: str, csv_filename: str) -> Optional[Dict]:
    """
    Run 4-phase deliberation for one debate.
    - JSON and CSV are saved after each phase (all judges in that phase responded).
    - Returns None (and logs the abort) if any API call fails.
    - Parsing failures are logged but do not abort.
    """
    topic = debate["topic"]
    debate_id = debate["debate_id"]
    canonical_debate_number = extract_canonical_debate_number(debate_id)

    print(f"\n{'='*70}")
    print(f"DELIBERATION #{deliberation_number}")
    print(f"Debate ID: {debate_id}")
    print(f"Judge shuffle seed: {canonical_debate_number}")
    print(f"Topic:     {topic}")
    print(f"{'='*70}")

    judges = get_judges_for_topic(
        topic, persona_df, num_judges, majority_type,
        shuffle_seed=canonical_debate_number,
    )
    if not judges:
        print(f"[SKIP] No personas found for topic: '{topic}'")
        return None

    n = len(judges)
    deliberation_id = f"delib_{deliberation_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    debate_transcript_formatted = format_debate_transcript(debate["transcript"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    deliberation = {
        "deliberation_id": deliberation_id,
        "debate_id": debate_id,
        "topic": topic,
        "general_topic": judges[0]["general_topic"],
        "statement_1": debate["statement_1"],
        "statement_2": debate["statement_2"],
        "majority_type": majority_type,
        "judges": [
            {"judge_id": j["judge_id"], "judge_name": j["judge_name"],
             "general_belief": j["general_belief"], "belief_type": j["belief_type"]}
            for j in judges
        ],
        "phases": {
            "prior_beliefs": [],
            "round_1": [],
            "round_2": [],
            "round_3": [],
        },
        "collective_decision": None,
        "has_errors": False,
        "errors": [],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_judges": n,
            "judge_model": JUDGE_MODEL,
            "canonical_debate_number": canonical_debate_number,
            "judge_shuffle_seed": canonical_debate_number,
            "deliberation_transcript_order_policy": "round_1_normal_round_2_mirrored",
        },
    }

    # ==========================================================================
    # PHASE 1: Prior belief extraction (before seeing the debate)
    # ==========================================================================

    print(f"\nPHASE 1: Prior belief extraction ({n} judges in parallel)...")

    def _phase1_call(judge):
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
            return judge, None, None
        parsed = parse_prior_belief(response, deliberation_id, judge["judge_id"], output_dir)
        return judge, response, parsed

    with ThreadPoolExecutor(max_workers=n) as ex:
        phase1_results = list(ex.map(_phase1_call, judges))

    for judge, response, parsed in phase1_results:
        jid = judge["judge_id"]
        if response is None or parsed is None:
            print(f"[ABORT] API failure for {jid} in Phase 1 — aborting debate")
            log_abort_to_errors_summary(deliberation_id, debate_id, topic,
                                        jid, "phase_1_prior_belief", output_dir)
            return None
        if parsed["statement"] is None or parsed["confidence"] is None:
            msg = (f"Parsing failed for {jid} Phase 1 — "
                   f"statement: {parsed['statement']}, confidence: {parsed['confidence']}")
            print(f"[WARNING] {msg}")
            log_error_to_csv(deliberation_id, debate_id, topic, jid, "prior_belief", msg, output_dir)
            deliberation["has_errors"] = True
            deliberation["errors"].append({"judge_id": jid, "phase": "prior_belief", "error": msg})

        deliberation["phases"]["prior_beliefs"].append(
            {"judge_id": jid, "judge_name": judge["judge_name"], **parsed}
        )
        print(f"  {jid} → {parsed['statement']} (confidence: {parsed['confidence']})")

    print(f"  [SAVE] Phase 1 complete — saving progress...")
    save_progress(deliberation, output_dir, csv_filename)

    # ==========================================================================
    # PHASE 2: Post-debate evaluation (Round 1 of deliberation)
    # ==========================================================================

    print(f"\nPHASE 2: Post-debate evaluation + Round 1 deliberation ({n} judges, 1 warms cache + {n - 1} parallel)...")

    def _phase2_call(judge):
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
            number_other_judges=n - 1,
        )
        response = send_to_judge(user_prompt, system_prompt, cache=True)
        if response is None:
            return judge, None, None
        parsed = parse_round_1(response, deliberation_id, judge["judge_id"], output_dir)
        return judge, response, parsed

    # Warm cache with first judge sequentially, then run the rest in parallel.
    phase2_results = [_phase2_call(judges[0])]
    if n > 1:
        with ThreadPoolExecutor(max_workers=n - 1) as ex:
            phase2_results.extend(ex.map(_phase2_call, judges[1:]))

    for judge, response, parsed in phase2_results:
        jid = judge["judge_id"]
        if response is None or parsed is None:
            print(f"[ABORT] API failure for {jid} in Phase 2 — aborting debate")
            log_abort_to_errors_summary(deliberation_id, debate_id, topic,
                                        jid, "phase_2_round_1", output_dir)
            return None
        if parsed["statement"] is None or parsed["confidence"] is None:
            msg = (f"Parsing failed for {jid} Phase 2 (Round 1) — "
                   f"statement: {parsed['statement']}, confidence: {parsed['confidence']}")
            print(f"[WARNING] {msg}")
            log_error_to_csv(deliberation_id, debate_id, topic, jid, "round_1", msg, output_dir)
            deliberation["has_errors"] = True
            deliberation["errors"].append({"judge_id": jid, "phase": "round_1", "error": msg})

        stmt = parsed.get("statement")
        parsed["ground_truth_alignment"] = (stmt == "Statement 1") if stmt else None

        deliberation["phases"]["round_1"].append(
            {"judge_id": jid, "judge_name": judge["judge_name"], **parsed}
        )
        print(f"  {jid} → Statement: {parsed['statement']}, Confidence: {parsed['confidence']}")

    print(f"  [SAVE] Phase 2 complete — saving progress...")
    save_progress(deliberation, output_dir, csv_filename)

    deliberation_transcript_r1 = format_round1_key_points_transcript(deliberation["phases"]["round_1"])

    # ==========================================================================
    # PHASE 3: Deliberation Round 2 (response to other judges)
    # ==========================================================================

    print(f"\nPHASE 3: Deliberation Round 2 — response to other judges ({n} judges, 1 warms cache + {n - 1} parallel)...")

    def _phase3_call(judge):
        system_prompt = SYSTEM_PROMPT_PERSONA.format(
            judge_name=judge["judge_name"],
            general_topic=judge["general_topic"],
            general_belief=judge["general_belief"],
        )
        user_prompt = PROMPT_DELIBERATION_ROUND_2.format(
            specific_topic=topic,
            statement_1=debate["statement_1"],
            statement_2=debate["statement_2"],
            debater_1="Debater_1",
            debater_2="Debater_2",
            assigned_statement_debater_1="Statement 1",
            assigned_statement_debater_2="Statement 2",
            transcript=debate_transcript_formatted,
            number_other_judges=n - 1,
            deliberation_transcript=deliberation_transcript_r1,
        )
        response = send_to_judge(user_prompt, system_prompt, cache=True)
        if response is None:
            return judge, None, None
        parsed = parse_round_2(response, deliberation_id, judge["judge_id"], output_dir)
        return judge, response, parsed

    phase3_results = [_phase3_call(judges[0])]
    if n > 1:
        with ThreadPoolExecutor(max_workers=n - 1) as ex:
            phase3_results.extend(ex.map(_phase3_call, judges[1:]))

    for judge, response, parsed in phase3_results:
        jid = judge["judge_id"]
        if response is None or parsed is None:
            print(f"[ABORT] API failure for {jid} in Phase 3 — aborting debate")
            log_abort_to_errors_summary(deliberation_id, debate_id, topic,
                                        jid, "phase_3_round_2", output_dir)
            return None
        if parsed["response_to_others"] is None and parsed["outstanding_issues"] is None:
            msg = f"Parsing fully failed for {jid} Phase 3 (Round 2)"
            print(f"[WARNING] {msg}")
            log_error_to_csv(deliberation_id, debate_id, topic, jid, "round_2", msg, output_dir)
            deliberation["has_errors"] = True
            deliberation["errors"].append({"judge_id": jid, "phase": "round_2", "error": msg})

        deliberation["phases"]["round_2"].append(
            {"judge_id": jid, "judge_name": judge["judge_name"], **parsed}
        )
        print(f"  {jid} → outstanding_issues: {'parsed' if parsed['outstanding_issues'] else 'missing'}")

    print(f"  [SAVE] Phase 3 complete — saving progress...")
    save_progress(deliberation, output_dir, csv_filename)

    deliberation_transcript_r2_mirrored = format_deliberation_transcript(
        list(reversed(deliberation["phases"]["round_2"]))
    )
    deliberation_transcript_r1_r2 = (
        "=== ROUND 1 ===\n\n" + deliberation_transcript_r1 +
        "\n\n=== ROUND 2 ===\n\n" + deliberation_transcript_r2_mirrored
    )

    # ==========================================================================
    # PHASE 4: Deliberation Round 3 — final evaluation (consensus)
    # ==========================================================================

    print(f"\nPHASE 4: Deliberation Round 3 — final consensus ({n} judges, 1 warms cache + {n - 1} parallel)...")

    def _phase4_call(judge):
        system_prompt = SYSTEM_PROMPT_PERSONA.format(
            judge_name=judge["judge_name"],
            general_topic=judge["general_topic"],
            general_belief=judge["general_belief"],
        )
        user_prompt = PROMPT_DELIBERATION_ROUND_3.format(
            specific_topic=topic,
            statement_1=debate["statement_1"],
            statement_2=debate["statement_2"],
            debater_1="Debater_1",
            debater_2="Debater_2",
            assigned_statement_debater_1="Statement 1",
            assigned_statement_debater_2="Statement 2",
            transcript=debate_transcript_formatted,
            number_other_judges=n - 1,
            deliberation_transcript=deliberation_transcript_r1_r2,
        )
        response = send_to_judge(user_prompt, system_prompt, cache=True)
        if response is None:
            return judge, None, None
        parsed = parse_round_3(response, deliberation_id, judge["judge_id"], output_dir)
        return judge, response, parsed

    phase4_results = [_phase4_call(judges[0])]
    if n > 1:
        with ThreadPoolExecutor(max_workers=n - 1) as ex:
            phase4_results.extend(ex.map(_phase4_call, judges[1:]))

    for judge, response, parsed in phase4_results:
        jid = judge["judge_id"]
        if response is None or parsed is None:
            print(f"[ABORT] API failure for {jid} in Phase 4 — aborting debate")
            log_abort_to_errors_summary(deliberation_id, debate_id, topic,
                                        jid, "phase_4_round_3", output_dir)
            return None
        if parsed["statement"] is None:
            msg = f"Parsing failed for {jid} Phase 4 (Round 3) — statement: {parsed['statement']}"
            print(f"[WARNING] {msg}")
            log_error_to_csv(deliberation_id, debate_id, topic, jid, "round_3", msg, output_dir)
            deliberation["has_errors"] = True
            deliberation["errors"].append({"judge_id": jid, "phase": "round_3", "error": msg})

        stmt = parsed.get("statement")
        parsed["ground_truth_alignment"] = (
            (stmt == "Statement 1") if stmt and stmt != "No agreement reached" else None
        )

        deliberation["phases"]["round_3"].append(
            {"judge_id": jid, "judge_name": judge["judge_name"], **parsed}
        )
        print(f"  {jid} → Statement: {parsed['statement']}")

    # ==========================================================================
    # Compute derived metrics (before final save)
    # ==========================================================================

    # Collective decision: majority vote on Round 3 statements (includes "No agreement reached")
    final_statements = [r["statement"] for r in deliberation["phases"]["round_3"] if r["statement"]]
    if final_statements:
        counts = Counter(final_statements)
        deliberation["collective_decision"] = {
            "final_statement": counts.most_common(1)[0][0],
            "vote_count": dict(counts),
            "unanimous": len(set(final_statements)) == 1,
        }

    # Ground truth alignment for prior beliefs (round_1 and round_3 already set inline)
    for entry in deliberation["phases"]["prior_beliefs"]:
        stmt = entry.get("statement")
        entry["ground_truth_alignment"] = (stmt == "Statement 1") if stmt else None

    # Belief change flags stored on prior_beliefs entries for CSV access
    prior_list = deliberation["phases"]["prior_beliefs"]
    r1_list = deliberation["phases"]["round_1"]
    r3_list = deliberation["phases"]["round_3"]

    for i in range(n):
        prior_stmt = prior_list[i]["statement"]
        r1_stmt = r1_list[i]["statement"]
        r3_stmt = r3_list[i]["statement"]
        prior_list[i]["belief_changed_prior_to_r1"] = (
            (prior_stmt != r1_stmt) if prior_stmt and r1_stmt else None
        )
        # Belief change is undefined when r3 is "No agreement reached"
        prior_list[i]["belief_changed_prior_to_r3"] = (
            (prior_stmt != r3_stmt)
            if prior_stmt and r3_stmt and r3_stmt != "No agreement reached"
            else None
        )
        prior_list[i]["belief_changed_r1_to_r3"] = (
            (r1_stmt != r3_stmt)
            if r1_stmt and r3_stmt and r3_stmt != "No agreement reached"
            else None
        )

    print(f"  [SAVE] Phase 4 complete — saving final state...")
    save_progress(deliberation, output_dir, csv_filename)

    print(f"\n[SUCCESS] Deliberation completed")
    print(f"Collective decision: {deliberation['collective_decision']}")
    if deliberation["has_errors"]:
        print(f"[WARNING] Completed with {len(deliberation['errors'])} parsing error(s)")

    return deliberation

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_all_deliberations(debates_dir: str, persona_dataset_path: str,
                          output_dir: str, csv_filename: str,
                          majority_type: str,
                          n_deliberations: Optional[int] = None) -> List[Dict]:
    """
    Run deliberations for all (or n_deliberations) debates.
    Number of judges comes from the global N_JUDGES constant; majority_type ('mainstream'
    or 'skeptical') determines the composition of beliefs per debate.
    JSON and CSV are saved progressively after each phase inside
    execute_single_deliberation — no separate save needed here.
    errors_summary.json is updated after each deliberation with parsing errors.
    """
    persona_df = load_persona_dataset(persona_dataset_path)
    if persona_df is None:
        return []

    num_judges = N_JUDGES
    n_mainstream, n_skeptical = get_majority_composition(num_judges, majority_type)
    print(f"[PERSONA] {num_judges} judges per topic — majority: {majority_type} "
          f"({n_mainstream} mainstream, {n_skeptical} skeptical)")

    debates_path = Path(debates_dir)
    if not debates_path.exists():
        print(f"[ERROR] Debates directory not found: {debates_dir}")
        return []

    debate_files = sorted(debates_path.glob("debate_*.json"))
    if not debate_files:
        print(f"[ERROR] No debate JSON files found in {debates_dir}")
        return []

    if n_deliberations is not None:
        debate_files = debate_files[:n_deliberations]

    # Timestamp-stamped CSV so runs don't overwrite each other
    csv_ts = f"{Path(csv_filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"\n{'#'*70}")
    print(f"DELIBERATION EXPERIMENT — CON CONSENSO")
    print(f"{'#'*70}")
    print(f"Debates to process: {len(debate_files)}")
    print(f"Judge model:        {JUDGE_MODEL}")
    print(f"Judges per debate:  {num_judges} ({majority_type} majority)")
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

            result = execute_single_deliberation(
                debate, persona_df, num_judges, majority_type, idx, output_dir, csv_ts
            )

            if result is None:
                print(f"[SKIP] Deliberation {idx} aborted (see errors_summary.json)")
                skipped += 1
            else:
                # Update errors_summary.json if there were parsing errors
                update_errors_summary(result, output_dir)
                completed.append(result)
                print(f"[DONE] Deliberation {idx} complete\n")

        except Exception as e:
            print(f"[ERROR] Unexpected exception in deliberation {idx}: {e}")
            traceback.print_exc()
            skipped += 1


    # Final summary
    print(f"\n{'='*70}")
    print("ALL DELIBERATIONS COMPLETED")
    print(f"Completed: {len(completed)} / {len(debate_files)}   Skipped/Aborted: {skipped}")
    print(f"Output: {output_dir}/")
    print(f"{'='*70}")

    if completed:
        total_judge_evals = sum(d["metadata"]["num_judges"] for d in completed)

        belief_changes = sum(
            1 for d in completed
            for pb in d["phases"]["prior_beliefs"]
            if pb.get("belief_changed_prior_to_r3") is True
        )
        unanimous = sum(
            1 for d in completed
            if d["collective_decision"] and d["collective_decision"]["unanimous"]
        )
        no_agreement = sum(
            1 for d in completed
            if d["collective_decision"]
            and d["collective_decision"]["final_statement"] == "No agreement reached"
        )
        correct_prior = sum(
            1 for d in completed
            for pb in d["phases"]["prior_beliefs"]
            if pb.get("ground_truth_alignment") is True
        )
        correct_final = sum(
            1 for d in completed
            for r3 in d["phases"]["round_3"]
            if r3.get("ground_truth_alignment") is True
        )
        error_delibs = sum(1 for d in completed if d["has_errors"])

        print(f"\nSUMMARY STATISTICS:")
        print(f"Total judge evaluations:           {total_judge_evals}")
        print(f"Belief changes (prior → final):    {belief_changes}/{total_judge_evals} ({belief_changes/total_judge_evals*100:.1f}%)")
        print(f"Unanimous collective decisions:    {unanimous}/{len(completed)} ({unanimous/len(completed)*100:.1f}%)")
        print(f"No agreement reached:              {no_agreement}/{len(completed)} ({no_agreement/len(completed)*100:.1f}%)")
        print(f"Ground truth alignment (prior):    {correct_prior}/{total_judge_evals} ({correct_prior/total_judge_evals*100:.1f}%)")
        print(f"Ground truth alignment (final):    {correct_final}/{total_judge_evals} ({correct_final/total_judge_evals*100:.1f}%)")
        if error_delibs:
            print(f"Deliberations with parsing errors: {error_delibs}/{len(completed)}")

    return completed

# =============================================================================
# RUN CONFIGURATIONS
# =============================================================================

# Base directory for this deliberation batch, relative to this script's location
_BASE_DIR = Path(__file__).parent
_DATASET_PATH = str(Path(__file__).parent / "dataset_deliberation.csv")

RUNS = [
    {
        "debates_dir": str(_BASE_DIR / "debates"),
        "output_dir": str(_BASE_DIR / "deliberaciones_con_consenso" / f"{N_JUDGES}_jueces_mayoria_mainstream" / "orden_original"),
        "csv_filename": f"deliberaciones_{N_JUDGES}j_mainstream_original.csv",
        "majority_type": "mainstream",
    },
    {
        "debates_dir": str(_BASE_DIR / "inverted_debates"),
        "output_dir": str(_BASE_DIR / "deliberaciones_con_consenso" / f"{N_JUDGES}_jueces_mayoria_mainstream" / "orden_invertido"),
        "csv_filename": f"deliberaciones_{N_JUDGES}j_mainstream_invertido.csv",
        "majority_type": "mainstream",
    },
    {
        "debates_dir": str(_BASE_DIR / "debates"),
        "output_dir": str(_BASE_DIR / "deliberaciones_con_consenso" / f"{N_JUDGES}_jueces_mayoria_skeptical" / "orden_original"),
        "csv_filename": f"deliberaciones_{N_JUDGES}j_skeptical_original.csv",
        "majority_type": "skeptical",
    },
    {
        "debates_dir": str(_BASE_DIR / "inverted_debates"),
        "output_dir": str(_BASE_DIR / "deliberaciones_con_consenso" / f"{N_JUDGES}_jueces_mayoria_skeptical" / "orden_invertido"),
        "csv_filename": f"deliberaciones_{N_JUDGES}j_skeptical_invertido.csv",
        "majority_type": "skeptical",
    },
]

if __name__ == "__main__":

    N_DELIBERATIONS = 25  # Set to None to process all debates

    all_results = []

    for run_idx, run_config in enumerate(RUNS, start=1):
        print(f"\n{'#'*70}")
        print(f"RUN {run_idx}/{len(RUNS)}")
        print(f"Debates: {run_config['debates_dir']}")
        print(f"Output:  {run_config['output_dir']}")
        print(f"{'#'*70}\n")

        results = run_all_deliberations(
            debates_dir=run_config["debates_dir"],
            persona_dataset_path=_DATASET_PATH,
            output_dir=run_config["output_dir"],
            csv_filename=run_config["csv_filename"],
            majority_type=run_config["majority_type"],
            n_deliberations=N_DELIBERATIONS,
        )

        if results:
            all_results.extend(results)

    print(f"\n{'#'*70}")
    print(f"ALL RUNS COMPLETE — Total deliberations saved: {len(all_results)}")
    print(f"{'#'*70}")
