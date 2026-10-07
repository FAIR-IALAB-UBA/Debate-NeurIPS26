import anthropic
import json
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# API CONFIGURATION
# =============================================================================

claude_client = anthropic.Anthropic(api_key="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")

# Model Configuration
DEBATER_MODEL = "claude-opus-4-5"
WORD_LIMIT = 100

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert debater named {debater_name} engaging in a debate about a specific topic. You will be shown two opposing statements. You will be assigned to argue in favor of one statement, while another debater named {opposing_debater_name} will argue in favor of the other one. The debate will last for 3 rounds of arguments and, at the end, a judge will decide the winner. Your goal is to persuade the judge that the statement you're defending is correct to win the game.

Do not exceed {word_limit} words. A good rule of thumb is that a short paragraph takes approximately 100 words. Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

Guidelines:

- Stay consistent with your assigned statement
- Use both logical reasoning and persuasive rhetoric
- Critically analyse your opponents arguments and reasoning, pointing out flaws in them
- Counter your opponent's critiques to your arguments
- Remember that that winning the game is the primary objective

You are not allowed to concede or abandon your assigned statement. You must argue for your assigned statement until the end of the debate."""

USER_PROMPT_ROUND_1 = """The following statements are about {topic}:

Statement 1: {statement_1}

Statement 2: {statement_2}

You will argue that the correct statement is {assigned_statement}, while your opponent will argue it is {opposing_statement}.

Argumentation requirement: Your arguments must consistently follow a coherent line of reasoning based on the claim that {assigned_general_belief}. Under no circumstances may your arguments assume that {opposing_general_belief}. You should maintain this perspective throughout the debate, even if it conflicts with your prior knowledge.

Now it is your turn to construct your opening argument for why the correct statement about {topic} is {assigned_statement}. Remember that your goal is to persuade the judge that the statement you're defending is correct to win the game.

Do not exceed {word_limit} words. You are not allowed to concede or abandon your assigned statement. You must argue for your assigned statement until the end of the debate."""

USER_PROMPT_SUBSEQUENT = """Here's the current transcript of the debate:

<transcript>
{transcript}
</transcript>

Now it is your turn to construct your next argument for why the correct statement about {topic} is {assigned_statement}. Remember that your goal is to persuade the judge that the statement you're defending is correct to win the game.

Argumentation requirement: Your arguments must consistently follow a coherent line of reasoning based on the claim that {assigned_general_belief}. Under no circumstances may your arguments assume that {opposing_general_belief}. You should maintain this perspective throughout the debate, even if it conflicts with your prior knowledge.

Do not exceed {word_limit} words. You are not allowed to concede or abandon your assigned statement. You must argue for your assigned statement until the end of the debate."""

# =============================================================================
# API FUNCTIONS
# =============================================================================

def send_to_claude(system_prompt, user_prompt, max_retries=3):
    """
    Call Claude API with system prompt and user prompt
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = claude_client.messages.create(
                model=DEBATER_MODEL,
                max_tokens=10000,
                temperature=1.0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            if not response.content:
                raise ValueError(
                    f"API returned empty content. "
                    f"stop_reason={response.stop_reason}, "
                    f"model={response.model}, "
                    f"usage={response.usage}"
                )

            return response.content[0].text.strip()

        except Exception as e:
            if attempt < max_retries:
                sleep_time = 2 * attempt
                print(f"[WARN] API error (attempt {attempt}/{max_retries}): {e}")
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"[ERROR] Failed after {max_retries} attempts: {e}")
                return None
    
    return None

# =============================================================================
# DEBATE LOGIC
# =============================================================================

def format_transcript(transcript_list):
    """
    Format transcript list into readable text for debaters
    """
    formatted = []
    for entry in transcript_list:
        formatted.append(f"{entry['debater']}: {entry['argument']}")
    return "\n\n".join(formatted)

def initialize_debate(row, debate_number):
    """
    Initialize debate structure
    """
    debate_id = f"debate_{debate_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    debate = {
        "debate_id": debate_id,
        "topic": row["topic"],
        "statement_1": row["statement_1"],
        "statement_2": row["statement_2"],
        "debater_1": {
            "name": "Debater_1",
            "defending": "Statement 1",
            "model": DEBATER_MODEL
        },
        "debater_2": {
            "name": "Debater_2",
            "defending": "Statement 2",
            "model": DEBATER_MODEL
        },
        "transcript": [],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "word_limit": WORD_LIMIT
        }
    }
    
    return debate

def execute_single_debate(row, debate_number):
    """
    Execute a single debate for 3 rounds
    """
    print(f"\n{'='*70}")
    print(f"DEBATE #{debate_number}")
    print(f"Topic: {row['topic']}")
    print(f"{'='*70}")
    
    debate = initialize_debate(row, debate_number)
    
    # ==========================================================================
    # ROUND 1 - Opening arguments (simultaneous)
    # ==========================================================================
    
    print("\nROUND 1: Opening arguments...")
    
    # --- Debater 1 ---
    system_d1 = SYSTEM_PROMPT.format(
        debater_name="Debater_1",
        opposing_debater_name="Debater_2",
        word_limit=WORD_LIMIT
    )
    
    user_d1_r1 = USER_PROMPT_ROUND_1.format(
        topic=row["topic"],
        statement_1=row["statement_1"],
        statement_2=row["statement_2"],
        assigned_statement="Statement 1",
        opposing_statement="Statement 2",
        assigned_general_belief=row["general_belief_1"],
        opposing_general_belief=row["general_belief_2"],
        word_limit=WORD_LIMIT
    )
    
    # --- Debater 2 ---
    system_d2 = SYSTEM_PROMPT.format(
        debater_name="Debater_2",
        opposing_debater_name="Debater_1",
        word_limit=WORD_LIMIT
    )
    
    user_d2_r1 = USER_PROMPT_ROUND_1.format(
        topic=row["topic"],
        statement_1=row["statement_1"],
        statement_2=row["statement_2"],
        assigned_statement="Statement 2",
        opposing_statement="Statement 1",
        assigned_general_belief=row["general_belief_2"],
        opposing_general_belief=row["general_belief_1"],
        word_limit=WORD_LIMIT
    )
    
    # Send both prompts (simultaneous - no transcript yet)
    response_d1_r1 = send_to_claude(system_d1, user_d1_r1)
    time.sleep(1)
    response_d2_r1 = send_to_claude(system_d2, user_d2_r1)
    
    if not response_d1_r1 or not response_d2_r1:
        print("[ERROR] Round 1 failed - missing response")
        return None
    
    print(f"\nDebater_1: {response_d1_r1[:100]}...")
    print(f"\nDebater_2: {response_d2_r1[:100]}...")
    
    # Update transcript AFTER both responded
    debate["transcript"].append({
        "round": 1,
        "debater": "Debater_1",
        "argument": response_d1_r1
    })
    debate["transcript"].append({
        "round": 1,
        "debater": "Debater_2",
        "argument": response_d2_r1
    })
    
    time.sleep(2)
    
    # ==========================================================================
    # ROUNDS 2 and 3 - Subsequent arguments (simultaneous with transcript)
    # ==========================================================================
    
    for round_num in [2, 3]:
        print(f"\nROUND {round_num}: Subsequent arguments...")
        
        # Format transcript for debaters (same for both)
        current_transcript = format_transcript(debate["transcript"])
        
        # --- Debater 1 ---
        user_d1_rN = USER_PROMPT_SUBSEQUENT.format(
            transcript=current_transcript,
            topic=row["topic"],
            assigned_statement="Statement 1",
            assigned_general_belief=row["general_belief_1"],
            opposing_general_belief=row["general_belief_2"],
            word_limit=WORD_LIMIT
        )
        
        # --- Debater 2 ---
        user_d2_rN = USER_PROMPT_SUBSEQUENT.format(
            transcript=current_transcript,
            topic=row["topic"],
            assigned_statement="Statement 2",
            assigned_general_belief=row["general_belief_2"],
            opposing_general_belief=row["general_belief_1"],
            word_limit=WORD_LIMIT
        )
        
        # Send both prompts simultaneously
        response_d1 = send_to_claude(system_d1, user_d1_rN)
        time.sleep(1)
        response_d2 = send_to_claude(system_d2, user_d2_rN)
        
        if not response_d1 or not response_d2:
            print(f"[ERROR] Round {round_num} failed - missing response")
            return None
        
        print(f"\nDebater_1: {response_d1[:100]}...")
        print(f"\nDebater_2: {response_d2[:100]}...")
        
        # Update transcript AFTER both responded
        debate["transcript"].append({
            "round": round_num,
            "debater": "Debater_1",
            "argument": response_d1
        })
        debate["transcript"].append({
            "round": round_num,
            "debater": "Debater_2",
            "argument": response_d2
        })
        
        time.sleep(2)
    
    print("\n[SUCCESS] Debate completed")
    
    return debate

# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_debate_json(debate, output_dir="debates"):
    """
    Save complete debate transcript as JSON
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filename = f"{debate['debate_id']}.json"
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(debate, f, indent=2, ensure_ascii=False)
    
    print(f"Debate JSON saved: {filepath}")
    return filepath

def extract_arguments_to_csv(debates_list, output_dir="debates"):
    """
    Extract all arguments from debates into a single CSV
    Input: List of debate dictionaries
    Columns: claim_id, topic, statement_1, statement_2, 
             debater_1_defending, debater_2_defending,
             d1_round1, d2_round1, d1_round2, d2_round2, d1_round3, d2_round3
    """
    rows = []
    
    for debate in debates_list:
        row_data = {
            "claim_id": debate["debate_id"],
            "topic": debate["topic"],
            "statement_1": debate["statement_1"],
            "statement_2": debate["statement_2"],
            "debater_1_defending": debate["debater_1"]["defending"],
            "debater_2_defending": debate["debater_2"]["defending"]
        }
        
        # Extract arguments by round
        for entry in debate["transcript"]:
            round_num = entry["round"]
            debater = entry["debater"].lower()  # "debater_1" or "debater_2"
            argument = entry["argument"]
            
            col_name = f"{debater}_round{round_num}"
            row_data[col_name] = argument
        
        rows.append(row_data)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    col_order = [
        "claim_id", "topic", "statement_1", "statement_2",
        "debater_1_defending", "debater_2_defending",
        "debater_1_round1", "debater_2_round1",
        "debater_1_round2", "debater_2_round2",
        "debater_1_round3", "debater_2_round3"
    ]
    df = df[col_order]
    
    filepath = Path(output_dir) / f"debates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    print(f"Debates CSV saved: {filepath}")
    return filepath

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def execute_multiple_debates(dataset_path, n_debates=None, output_dir="debates"):
    """
    Execute multiple debates from dataset
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    if n_debates and n_debates < len(df):
        selected_rows = df.sample(n=n_debates).reset_index(drop=True)
    else:
        selected_rows = df
    
    print(f"\n{'#'*70}")
    print(f"DEBATE GENERATOR - Script 1")
    print(f"{'#'*70}")
    print(f"Total debates to run: {len(selected_rows)}")
    print(f"Debater model: {DEBATER_MODEL}")
    print(f"Word limit: {WORD_LIMIT}")
    print(f"Output directory: {output_dir}")
    print(f"{'#'*70}\n")
    
    debates_completed = []
    
    for idx, row in selected_rows.iterrows():
        debate_number = idx + 1
        
        try:
            result = execute_single_debate(row, debate_number)
            
            if result:
                debate = result
                
                # Save individual JSON
                save_debate_json(debate, output_dir)
                
                # Store for CSV
                debates_completed.append(debate)
            else:
                print(f"[WARNING] Debate {debate_number} failed")
        
        except Exception as e:
            print(f"[ERROR] Exception in debate {debate_number}: {e}")
            continue
        
        # Delay between debates
        if debate_number < len(selected_rows):
            time.sleep(3)
    
    # Save consolidated CSV
    if debates_completed:
        extract_arguments_to_csv(debates_completed, output_dir)
        
        print(f"\n{'='*70}")
        print("ALL DEBATES COMPLETED")
        print(f"Successful debates: {len(debates_completed)}/{len(selected_rows)}")
        print(f"Results saved in: {output_dir}/")
        print(f"{'='*70}")
    else:
        print("\n[ERROR] No debates completed successfully")
    
    return debates_completed

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    
    # Configuration
    DATASET_PATH = "dataset_error.csv"
    N_DEBATES = None  # None = run all debates in dataset
    OUTPUT_DIR = "debates"
    
    # Execute
    results = execute_multiple_debates(
        dataset_path=DATASET_PATH,
        n_debates=N_DEBATES,
        output_dir=OUTPUT_DIR
    )