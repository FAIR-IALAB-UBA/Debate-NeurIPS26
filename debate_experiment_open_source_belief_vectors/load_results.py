"""
Load the judge evaluations in results/ as one DataFrame per model.

results/<model>/evaluations_{mainstream,skeptical}_{original,inverted}_<model>.csv are
the outputs of run_judge_open_source_<model>.ipynb, split by judge group and debate
ordering like the other experiment folders of this repo. Concatenated, they are the
notebook's evaluations_<MODEL>_logodds.csv export: 2340 rows per model.
"""
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
MODEL_DIRS = {
    "Qwen2.5-7B-Instruct":   "qwen2.5_7b",
    "Llama-3.1-8B-Instruct": "llama_3.1_8b",
    "gemma-3-4b-it":         "gemma_3_4b",
}


def load_evaluations(model):
    """All rows of one model, plus the short column names the analysis scripts use."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, MODEL_DIRS[model], "evaluations_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["pre_correct"] = df["pre_statement_ground_truth_alignment"]
    df["post_correct"] = df["post_statement_ground_truth_alignment"]
    df["pre_conf"] = df["pre_confidence"]
    df["post_conf"] = df["post_confidence"]
    return df
