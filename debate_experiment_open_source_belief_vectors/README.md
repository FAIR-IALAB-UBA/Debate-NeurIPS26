# Open-source LLM judges: GPB injection with persona prompting and persona capping

Single-judge debate experiment with three open-weight judges: Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct and Gemma-3-4B-IT. Besides the GPB system prompt used for the other
judges, general prior beliefs (GPB) are injected with a demographic persona in the prompt
(persona prompting) and by capping the judge's activations along a persona direction
extracted from the model (persona capping). These are the open-source rows of Table 1 and
the GPB injection methods of Appendix D.

Statement 1 is always the true statement. The debates and judge prompts are those of
`../debate_experiment_one_judge_con_persona/run_judge_con_persona.py`; only the numbering
and wording of the two answer-format questions in the post-debate prompt differ slightly.

## Conditions

Each judge–claim assignment is evaluated under a 2 × 3 design (system prompt × capping):

| Condition | System prompt | Persona capping | Name in Table 1 |
|---|---|---|---|
| C1 | none | no | Unconditioned |
| C2 | none | yes | Persona Capping |
| C3 | GPB | no | GPB System Prompt |
| C4 | GPB | yes | Persona Capping |
| C5 | GPB + persona | no | Persona Prompting |
| C6 | GPB + persona | yes | Persona Capping |

- **GPB prompt (C3–C6):** "You are a judge with a prior belief about {general_topic} as follows: {general_belief}. You should respond authentically from this perspective throughout all interactions. You are not allowed to refuse a response."
- **Persona (C5, C6):** the prompt adds "Additional context about you: {judge_persona}", the persona assigned to that claim in the dataset (e.g. "man, african, a young adult secondary-level education, generally skeptical, and afraid of heights").
- **Groups:** `mainstream` = grounded (true GPB), `skeptical` = erroneous (false GPB). In C1 both groups get identical inputs.

## Pipeline

| Step | File | Runs on | Output |
|---|---|---|---|
| 0 | `generate_extraction_responses.py` | OpenRouter API | `extraction_responses/` |
| 1 | `extract_vectors.py` | GPU | `artifacts/` (not committed) |
| 2 | `causality_capping.ipynb` | Colab GPU | `results/capping_sweep_qwen.csv` |
| 3 | `run_judge_open_source_{qwen,llama,gemma}.ipynb` | Colab GPU | `results/<model>/` |
| 4 | `analysis.ipynb` | CPU | `results/figures/` |
| 5 | `significance_clustered.py`, `confidence_clustered.py` | CPU | `results/*.csv` |

**0. Extraction corpus.** For each persona, the same model (served through OpenRouter,
temperature 0.7, up to 150 tokens) answers 50 open-ended questions under 3 system prompts
describing the persona, i.e. 150 responses per persona. The judge personas get questions on
five of the ten topics (nuclear fusion, black hole imaging, quantum computing, exoplanets,
civilian spaceflight). The `default` assistant baseline and the five political personas get 50
questions on political topics. Needs `OPENROUTER_API_KEY` (environment or `.env`); the script
resumes from the committed JSON files.

**1. Persona axes.** The local model (TransformerLens; bf16 for Gemma, fp16 otherwise) reads
each (system prompt, question, response) triple. Each response is represented by its mean
residual-stream activation (`hook_resid_post`) over response tokens, and the axis of a persona
at a layer is `normalize(mean(persona responses) − mean(default responses))`. Writes
`persona_axes_v2_<MODEL>.pkl` and `extraction_act_matrices_<MODEL>.pkl` (1.4–2.1 GB per
model) to `artifacts/`. Llama and Gemma need `HF_TOKEN`.

```bash
python extract_vectors.py --model qwen    # or llama, gemma
```

**2. Capping sweep (Qwen).** Caps along the axes of 7 political/judge personas, over 21 belief
statements, sweeping the layer-window center {14, 18, 22} × width {4, 8} × τ percentile
{1, 10, 25, 50, 75, 90, 99}. Step 3 uses center 18, width 8 and the 25th percentile.

**3. Judge evaluation.** One Colab notebook per model. Each pre- and post-debate judgment is
a single forward pass: the judge's answer is prefilled with "Statement: Statement " and the
verdict is the sign of `log P("1") − log P("2")` for the next token. No text is generated and
nothing is sampled. Confidence is not self-reported: it is `100 · sigmoid(|log-odds|)` ∈ [50, 100].

Persona capping applies `h ← h − v · min(⟨h, v⟩ − τ, 0)` to `hook_resid_post` at every token
position, in both judgments, on a contiguous window of 8 layers (0-indexed): 14–21 for Qwen,
12–19 for Llama, 9–16 for Gemma. `v` is the axis of the persona assigned to the claim, and τ is,
per layer, the 25th percentile of the projections onto `v` of that persona's 150 extraction
responses.

The notebooks read from a Google Drive folder (`BASE_DIR`):
- the persona dataset `Dataset Vicki.xlsx` (sheets `Victoria`, `dataset_persona_skeptical`, `dataset_persona_mainstream`);
- the debates of `../debates/transcripts/` as one flat CSV (`debates_20260325_212652 - debates_20260325_212652.csv`, one row per debate with the three rounds of each debater);
- the inverted debates (`inverted_debates-20260501T131952Z-3-001/debates_inverted.csv`), with the same arguments in swapped order;
- `persona_axes_v2_<MODEL>.pkl` and `extraction_act_matrices_<MODEL>.pkl` from step 1.

The Qwen and Llama notebooks install `transformers==4.44.2`, the Gemma one `transformers>=4.47.0`.

**4. Figures.** Reads `results/` and the Claude Sonnet 4 reference (C3, temperature 0) from
`../debate_experiment_one_judge_con_persona/results/sonnet_temp_0/`. Known limitations of
`fig_main`: the Claude lines are Claude's C3 values in all three rows (C1, C3, C4); in the C1 row
the two groups are identical by construction; and it shows the original ordering only.

**5. Statistics.** Each claim is judged once pre-debate and debated twice (original and
inverted ordering), so both scripts collapse each claim to one unit (post = mean of the two
post-debate verdicts) instead of pooling the two orderings. `significance_clustered.py` tests
pre→post accuracy with a paired sign-flip permutation test (with sign, Wilcoxon and cluster
bootstrap alongside) and applies Holm–Bonferroni across the 33 cells (3 models × [C1 + 5
conditions × 2 groups]). `confidence_clustered.py` does the same collapse for confidence.

```bash
python significance_clustered.py
python confidence_clustered.py
```

## Results

`results/<model>/evaluations_{mainstream,skeptical}_{original,inverted}_<model>.csv` (models
`qwen2.5_7b`, `llama_3.1_8b`, `gemma_3_4b`) hold one row per claim × condition. The first 19
columns are those of the other experiment folders; `judge_model` ends in `-capped` for C2, C4
and C6, and `pre_response_raw` / `post_response_raw` contain the log-odds instead of text. Six
columns are added: `condition`, `debate_type` (`normal` = original ordering, `inverted`), `group`,
`trait_key` (the persona axis used), `pre_lo` and `post_lo` (log-odds).

Of the 100 claims, 96 match a debate in the original ordering and 99 in the inverted one, so
each original-ordering file has 96 × 6 = 576 rows and each inverted one 99 × 6 = 594.
`load_results.py` concatenates the four files of a model.

| File | Content |
|---|---|
| `results/capping_sweep_qwen.csv` | Step 2: 6174 rows (7 personas × 21 beliefs × 3 centers × 2 widths × 7 percentiles) |
| `results/significance_clustered.csv` | Step 5: 48 rows, the 33 cells plus the per-condition averages of both groups |
| `results/confidence_{summary,per_model,cells}.csv` | Step 5: confidence change |
| `results/figures/` | Step 4: `fig_main`, `fig2_cap_prior`, `fig3_pre_post_c3`, `supp_inversion`, `supp_cap_delta` |

Steps 4 and 5 run from the committed results. Steps 1–3 need a GPU, the inputs listed in
step 3 and the step-1 activation matrices, which are too large to commit.

## Personas

One grounded and one erroneous persona per general topic (Table 8). Each was described with 3
system prompts for the axis extraction (step 0); in the evaluation, each claim's judge persona is
one of 10 demographic variants of its topic's persona.

| General topic | Grounded (`mainstream`) | Erroneous (`skeptical`) |
|---|---|---|
| Supersonic Aircraft | `m_engineer_pilot` | `s_aviophobe` |
| 3D Bioprinting | `m_technologist` | `s_cautious_religious` |
| Controlled Nuclear Fusion | `m_physicist` | `s_peace_activist` |
| Operational Quantum Computers | `m_scientist_optimist` | `s_artist_pessimist` |
| Spacecraft Capable of Taking Civilians to Space | `m_entrepreneur_travel` | `s_religious_riskaverse` |
| Mirror Life | `m_science_layperson` | `s_christian_conservative` |
| Lab-grown Embryo Models | `m_biologist_atheist` | `s_moralist_religious` |
| Exascale Computing | `m_researcher_cs` | `s_artist_conservative` |
| Imaging Black Holes | `m_astronomer` | `s_pessimist_religious` |
| Confirmation of the Existence of Exoplanets | `m_phd_rightwing` | `s_religious_skeptical` |

## Requirements

torch, transformer_lens, transformers, pandas, numpy, scipy, statsmodels, matplotlib,
scikit-learn, tqdm, openpyxl, openai, python-dotenv, huggingface_hub.
