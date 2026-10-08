"""
Cross-Topic Results Merge
=========================
Joins the output of run_judge_crosstopic.py with the judge's prior belief on
subtopic N+1 and computes the cross-topic metrics.

The prior on N+1 is the pre-debate answer of the same judge (same model,
persona and temperature) in the one-judge-with-persona experiment
(debate_experiment_one_judge_con_persona/results/sonnet_temp_0). The join key
is (topic, statement_1, statement_2) of N+1.

Added columns:
  pre_statement_on_next, pre_confidence_on_next, pre_on_next_ground_truth_alignment
  cross_topic_belief_changed     post_statement (on N+1) != prior statement on N+1
  cross_topic_confidence_delta   post_confidence - prior confidence on N+1

Output: results/evaluations_crosstopic_<config>.csv
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = BASE_DIR.parent / "debate_experiment_one_judge_con_persona" / "results" / "sonnet_temp_0"
RESULTS_DIR = BASE_DIR / "results"

CONFIGS = [
    "mainstream_original",
    "mainstream_inverted",
    "skeptical_original",
    "skeptical_inverted",
]


def latest_crosstopic_csv(config: str) -> Path:
    """Most recent output of run_judge_crosstopic.py for a configuration."""
    output_dir = BASE_DIR / f"evaluations_crosstopic_{config}"
    csvs = sorted(output_dir.glob("judge_evaluations_crosstopic_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No cross-topic results found in {output_dir}")
    return csvs[-1]


def merge_config(config: str) -> pd.DataFrame:
    crosstopic = pd.read_csv(latest_crosstopic_csv(config))
    baseline = pd.read_csv(BASELINE_DIR / f"evaluations_{config}_sonnet_temp_0.csv")

    prior_on_next = baseline[[
        "topic", "statement_1", "statement_2",
        "pre_statement", "pre_confidence", "pre_statement_ground_truth_alignment",
    ]].rename(columns={
        "topic": "next_specific_topic",
        "statement_1": "next_statement_1",
        "statement_2": "next_statement_2",
        "pre_statement": "pre_statement_on_next",
        "pre_confidence": "pre_confidence_on_next",
        "pre_statement_ground_truth_alignment": "pre_on_next_ground_truth_alignment",
    })

    merged = crosstopic.merge(
        prior_on_next,
        on=["next_specific_topic", "next_statement_1", "next_statement_2"],
        how="left",
        validate="many_to_one",
    )

    unmatched = merged["pre_statement_on_next"].isna().sum()
    if unmatched:
        raise ValueError(f"{config}: {unmatched} rows without a prior on N+1")

    merged["cross_topic_belief_changed"] = merged["post_statement"] != merged["pre_statement_on_next"]
    merged["cross_topic_confidence_delta"] = merged["post_confidence"] - merged["pre_confidence_on_next"]
    return merged


if __name__ == "__main__":

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"{'config':<22}{'n':>4}{'belief changed':>17}{'conf. delta':>13}{'prior correct':>15}{'post correct':>14}")
    for config in CONFIGS:
        df = merge_config(config)
        df.to_csv(RESULTS_DIR / f"evaluations_crosstopic_{config}.csv", index=False)

        n = len(df)
        changed = int(df["cross_topic_belief_changed"].sum())
        print(f"{config:<22}{n:>4}{f'{changed} ({changed / n:.1%})':>17}"
              f"{df['cross_topic_confidence_delta'].mean():>+13.2f}"
              f"{df['pre_on_next_ground_truth_alignment'].mean():>15.1%}"
              f"{df['post_statement_ground_truth_alignment'].mean():>14.1%}")
