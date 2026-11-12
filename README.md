# Barthel index from nursing dictations: small fine-tuned model vs. frontier API

A small demo project: an LLM reads a German nursing dictation and fills in the **Barthel index**
(10 activities of daily living, 0–100 points). It answers one question:

> When does a small, fine-tuned open model match a rented frontier model on quality,
> and what does it gain in cost, speed and privacy?

All data is synthetic. No real patient data is used.

## Summary

Results on 40 held-out test notes (400 item decisions; details and confidence intervals in section 4):

| Model | Accuracy | Hallucination | USD / 1,000 notes |
|---|---|---|---|
| always `not_assessed` (floor) | 65.8 % | 0 % | – |
| Claude Haiku 5.5, zero-shot | 98.2 % | 0.0 % | 0.39 |
| Claude Sonnet 5.5, zero-shot | 98.2 % | 0.8 % | 5.37 |
| Gemma 3 4B, untrained | 54.0 % | 43.3 % | ~1.0 (assumed GPU price) |
| Gemma 3 4B + LoRA, 180 notes | 88.0 % | 5.3 % | ~0.95 (assumed GPU price) |

- **Frontier models are close to the ceiling on this task.** Claude scores 97-98 % in every setting. Haiku
  matches Sonnet at about 1/14 of the cost, and 6-shot examples did not help (differences are within noise).
  What is left is mostly contested labels, scoring-rule boundaries and inference-versus-abstain judgement calls.
- **A small untrained model is unusable.** Gemma 3 4B scores below a rule that always abstains: it fills in
  almost every item instead of answering `not_assessed` when the note does not say.
- **Fine-tuning on 180 notes closes most, not all, of the gap.** LoRA lifts Gemma by 34 points (hallucination
  43 % to 5 %) but stays about 10 points behind Claude. The largest error type is wrong score levels
  (off by one level), not hallucination.
- **Answer to the question:** on this task a small fine-tuned model gets within about 10 points of a frontier
  API but does not match it, and it gains nothing on cost or speed against Haiku (about $0.95 vs $0.39 per
  1,000 notes, 6.8 s vs 3.6 s per note). Its case is **self-hosting and privacy**: open weights that can run
  without sending patient text to a third party.
- **How far to trust this:** synthetic notes, only 40 test notes (wide intervals), labels reviewed by a model
  rather than a nurse, one run per setting (run-to-run variance not measured), and an unexplained drop from
  95 % on validation to 88 % on test (different writer and a different share of unmentioned items).
- **With production data the next steps would be:** have nurses label real dictations (to measure the label
  ceiling and replace the contested labels), train on a wider mix of patient types and absent-item rates,
  and test constrained decoding and a short prompt for the small model.

---

## 1. Task

**Input:** a free-text nursing dictation (German), e.g.

> Frühdienst, Zi. 4. Pat. muss beim Frühstück komplett angereicht werden, isst nichts selbst. RR 128/76 …
> Kann nicht gehen, auch nicht mit Hilfe, Rollstuhl selbst fahren nicht möglich. …

**Output:** one value per Barthel item (official ICD-10-GM item names):

```json
{"essen": 0, "aufsetzen_umsetzen": "not_assessed", "aufstehen_gehen": 0, "...": "..."}
```

- A score if the dictation supports it.
- `not_assessed` if the dictation does not mention the item. **The system must not guess.**

Scoring rules follow the Barthel index in ICD-10-GM ([BfArM](https://klassifikationen.bfarm.de/icd-10-gm/kode-suche/htmlgm2024/zusatz-06-anh-barthelindex.htm), based on the Hamburg manual),
condensed in `data/generate_synthetic_data.py` (`BARTHEL`, `GENERAL_RULES`).

## 2. Data

| File | Size | Written by | Notes |
|---|---|---|---|
| `data/barthel_train_source/train.json` | 200 notes | Claude Opus 5.5 | labels sampled first (seed 1), notes written per plan; see `DATA_CARD.md`. 180 for training, 20 for validation |
| `data/test/test.json` | 40 notes | ChatGPT, free text | written per plan; rewritten after review to avoid sentence templates and copied rubric wording |

### Label semantics (important)
Each item has three layers:

| Field | Meaning | Used for |
|---|---|---|
| `true_score` | the simulated patient's actual state (what the nurse knows from observation) | analysis only |
| `mention` | how the note covers the item: `explicit` / `inferred` / `absent` | evaluation breakdown |
| `target` | what the text supports: the score, or `not_assessed` if absent | **training and evaluation** |

Real paired data (dictation + completed Barthel form) only has `true_score`. Training on it directly
would teach the model to guess. Here `target` exists by construction; with real data it would require
annotating which items each dictation supports.

## 3. Approach

**Shared setup for every model:** the same prompt, the same output format (JSON), the same test set.


**Models compared**

| Setup | Training | Runs |
|---|---|---|
| Claude (frontier API), zero-shot and few-shot | none | API |
| OpenAI (frontier API), zero-shot and few-shot | none | API |
| Small open-weight model (Gemma 3 4B), untrained | none | free cloud T4 notebook, 4-bit; self-hostable |
| Same small model + LoRA fine-tune | 180 notes (20 held out for validation) | trained and evaluated on a free cloud T4 notebook; self-hostable |

"Self-hostable" means open weights that a hospital could run on its own infrastructure, so no patient text goes
to a third-party API. Everything here ran on a free cloud T4 notebook (Colab or Kaggle); the model was **not** run on a laptop,
so no laptop latency or cost is claimed. A hosted notebook is acceptable only because the data is synthetic.

The test set is never used for tuning. The prompt, output schema and few-shot examples (`barthel/prompt.py`) were
frozen before the first test run. If a later change is needed, both versions are reported; the first is not
replaced. Few-shot examples come from the 180 training notes only.

**Home advantage (caveat of synthetic data).** Each frontier family wrote part of the data:
Claude wrote the training notes (used for few-shot examples and fine-tuning), ChatGPT wrote the test notes.
A model may do better on text in its own family's style. Mitigation: evaluate a different model
within each family than the one that wrote the data, and report both families side by side.
The effect cannot be removed completely and is stated with the results.

## 4. Evaluation

Each test note has 10 items, so the test set has **400 item decisions**. Each decision has two parts:
*does the note support a score?* (detection) and *if so, which score?* (scoring).

**Metrics**

| Metric | Definition | Why |
|---|---|---|
| Overall accuracy | share of all 400 decisions exactly right (score or `not_assessed`) | single headline number |
| Detection precision | of the items the model scored, share the note actually mentions | low precision = guessing |
| Detection recall | of the items the note mentions, share the model scored | low recall = over-cautious |
| Hallucination rate | share of `absent` items the model still scored | safety metric, easiest to explain |
| Score accuracy | share of mentioned items with the exact right score, split by `explicit` / `inferred` | correct use of the rules |
| Cost and speed | € per 1,000 notes, seconds per note, runs on a laptop yes/no | the practical trade-off |

**Floor baselines:** always `not_assessed`; always the most frequent score per item. Any model must beat these clearly.

**Confidence intervals.** Items within one note are not independent (same writer, same patient), so
resampling is done on whole notes: draw 40 notes with replacement, recompute each metric, repeat
1,000 times, and report the 2.5th–97.5th percentile as the 95 % confidence interval.
Differences smaller than the interval width are not treated as findings.

**Results**

*All models on the 40 test notes (400 decisions).* Percentages; 95 % note-level bootstrap CI in
brackets. Generated by `barthel/evaluation.py` into `runs/results_test.md`. One run per row. The Gemma rows use plain
(unconstrained) greedy generation and a free-tier T4 GPU; their cost uses an assumed $0.50 / GPU hour.

| Model | Accuracy | Precision | Recall | Hallucination ↓ | Score acc. explicit / inferred | USD / 1,000 notes | s / note |
|---|---|---|---|---|---|---|---|
| always `not_assessed` | 65.8 [63-68] | n/a | 0.0 [0-0] | 0.0 [0-0] | 0.0 / 0.0 | – | – |
| majority score per item | 13.2 [10-17] | 34.2 [32-37] | 100.0 [100-100] | 100.0 [100-100] | 34.4 / 46.8 | – | – |
| Haiku 5.5, zero-shot | 98.2 [97-99] | 100.0 [100-100] | 96.4 [93-99] | 0.0 [0-0] | 97.8 / 89.4 | 0.39 | 3.6 |
| Haiku 5.5, 6-shot | 97.2 [96-99] | 98.5 [96-100] | 96.4 [94-99] | 0.8 [0-2] | 97.8 / 85.1 | 0.35 | 3.8 |
| Sonnet 5.5, zero-shot | 98.2 [97-100] | 98.5 [96-100] | 97.1 [94-99] | 0.8 [0-2] | 98.9 / 91.5 | 5.37 | 3.8 |
| Sonnet 5.5, 6-shot | 97.5 [96-99] | 100.0 [100-100] | 93.4 [90-97] | 0.0 [0-0] | 96.7 / 85.1 | 4.67 | 3.0 |
| Gemma 3 4B, untrained | 54.0 [48-59] | 50.0 [46-54] | 83.2 [76-90] | 43.3 [37-51] | 50.0 / 46.8 | 1.01 | 7.3 |
| Gemma 3 4B + LoRA (180 notes) | 88.0 [85-91] | 90.5 [86-95] | 97.1 [93-100] | 5.3 [3-8] | 82.2 / 61.7 | 0.95 | 6.8 |
| OpenAI | *not run* | | | | | | |

On the 20 validation notes (same writer and absent share as training): untrained 58.0 % accuracy,
68.5 % hallucination; LoRA 95.0 % accuracy, 1.4 % hallucination.

Notes on these numbers:
- **Costs are in USD** from measured token usage, with prompt caching on (the rules and JSON schema are a
  fixed ~2,750-token prefix, read at 10 % of the input price). Output tokens (mostly the model's thinking)
  are about 85–90 % of the remaining cost. Opus 5.5 was not evaluated because it wrote the training notes.
- **All four Claude runs are within noise of each other.** 7–11 of 400 decisions are wrong per run and the
  intervals overlap. Haiku matches Sonnet at about 1/14 of the cost. 6-shot is not better than zero-shot
  (6 decisions fixed, 13 broken across both models, not significant); run-to-run variance and the choice
  of the 6 examples (one draw, seed 0) have not been measured yet.
- **Error analysis:** `runs/error_analysis.txt` lists the 59 decisions that at least one of the four Claude
  runs or Gemma + LoRA got wrong (18 of them involve a Claude run), with every run's answer, the note text
  and the gold evidence; regenerate with `uv run python -m barthel.errors --notes`. Only 2 are wrong in all four
  Claude runs. Most Claude errors are on `inferred` items; several are contested labels or scoring-rule
  boundary cases (e.g. pouring soup: gold 10, models say 5; "blieb im Bett" as a 0 for transfers) and some
  "hallucinations" are clinically plausible inferences (cannot stand, so stairs = 0). The test labels were
  deliberately **not** corrected after seeing these errors; contested ones are reported, not fixed.

**Small model (Gemma 3 4B) findings**
- **Untrained, it is worse than always abstaining** (54.0 % vs 65.8 %). It scores almost every item
  (43 % hallucination); the "answer `not_assessed` if the note does not say" instruction is what a 4B model
  does not follow from the prompt alone.
- **LoRA on 180 notes closes most of the gap:** +34 points accuracy, hallucination 43 % to 5 %, recall as
  high as Claude's. It does not match Claude (88.0 % vs 97-98 %): 48 wrong decisions vs 7-11.
- **Where the 48 errors are:** 30 wrong scores (13 on explicit items, 17 on inferred; 16 too high, 14 too
  low, all but one off by a single level; 10 on `essen`, the contested preparation boundary), 14
  hallucinations, 4 missed items. Wrong levels, not hallucination, are the largest part of the gap.
  Nine of the 14 hallucinations predict 0 for items on notes of dependent-sounding patients. One untested
  explanation is that the training patients come from a few archetypes where many items are dependent.
- **Validation vs test:** 95.0 % on `val` but 88.0 % on test. The two differences, a 35 % vs 66 % absent
  share and a Claude vs ChatGPT writer, cannot be separated with these runs.
- **Cost and speed are not an advantage:** about $0.95 per 1,000 notes and 6.8 s per note on a T4 (long
  prompt, 4-bit, assumed GPU price) against $0.39 and 3.6 s for Haiku through the API. The case for the small
  model is self-hosting and privacy, not price.
- Fine-tuning settings were fixed before any test run: LoRA r=16, alpha 16, lr 2e-4, 3 epochs, effective batch
  16, loss on the answer only; `val` was used for monitoring only. Single run per setting; run-to-run
  variance is unmeasured.

## 5. Status

- [x] Barthel rules (ICD-10-GM) and label sampler (`data/generate_synthetic_data.py`)
- [x] Training set: 200 notes, validated, data card
- [x] First (template-based) test set reviewed and replaced by a free-text rewrite
- [ ] Check rewritten test set (format, sentence reuse, leaks)
- [x] Evaluation script + floor baselines + bootstrap CIs (`barthel/evaluation.py`, tested)
- [x] Shared prompt (rules in the prompt for all models; the `get_barthel_rule` tool was dropped from the main run)
- [x] Claude runs (Haiku 5.5, Sonnet 5.5; zero-shot and 6-shot) and first error analysis
- [ ] Noise check: repeat identical runs, 6-shot with other example draws
- [ ] OpenAI run (optional)
- [x] Untrained small model (Gemma 3 4B)
- [x] LoRA fine-tuning on 180 notes (`notebooks/finetune_gemma3_nursing_notes_to_barthel_index.ipynb`)
- [ ] Final write-up (error categories, small-vs-frontier conclusion)

## 6. Limitations

- **Label quality is checked automatically, not by a human.** Notes whose text does not clearly express
  their label count as model errors, so no model can reach 100 %; the true ceiling is unknown.
- **Home advantage:** see section 3.
- Synthetic notes are cleaner than real dictations (no speech-recognition errors, more regular phrasing).
- Labels assume the dictation is clinically correct.
- Some score levels are rare in the training data (see `DATA_CARD.md`).
- 40 test notes give wide confidence intervals; only large differences are meaningful.

## 7. Possible extensions (not built)

- **Validation with retry:** a validator checks the output (valid scores, evidence present) and the model corrects itself.
- **Patient state across notes:** routine notes update only the items they mention; others keep their last value.
- **Follow-up questions:** for items missing from an assessment dictation, the agent asks the nurse;
  the answer is simulated from `true_score`, and the gain from one round of questions is measured.
- **Naive vs. correct training:** fine-tune on `note → true_score` vs. `note → target` and compare hallucination rates.
- **Learning curve:** fine-tune on 30 / 100 / 180 notes to see how much data the small model needs.
- **Routing:** split one dictation into report categories (vital signs, catheter care, …) and the Barthel form.
- Human blind labelling of the test set to measure label quality; German speech-recognition noise; real, expert-annotated dictations.

## 8. Repository layout

```
barthel/
  prompt.py          shared prompt, JSON schema, few-shot selection, data splits (train 180 / val 20 / test 40)
  evaluation.py      metrics, floor baselines, note-level bootstrap CIs, markdown results table
  run_api.py         runs a Claude model over a split (prompt caching, cost + latency) -> runs/
  errors.py          error analysis across runs -> runs/error_analysis.txt
notebooks/
  finetune_gemma3_nursing_notes_to_barthel_index.ipynb   Gemma 3 4B untrained vs. LoRA (see note below)
tests/               pytest for evaluation.py and prompt.py (metrics, parse failures, bootstrap, splits,
                     regression on the test set)
runs/                predictions (*.json), cost/latency (*.meta.json), results_test.md, error_analysis.txt
data/
  generate_synthetic_data.py   Barthel rules (BARTHEL, GENERAL_RULES), archetypes, label sampler, note generator
  barthel_train_source/        train.json (200 notes), plans.json, batches/, make_plans.py, validate.py,
                               merge.py, DATA_CARD.md (how the training data was made)
  test/                        test.json (40 notes), README_barthel_test_40.md
```

Run from the project root with `uv run`. Tests: `uv run pytest`. Claude runs:
`uv run python -m barthel.run_api --model claude-haiku-5-5 --split test --shots 0`.

Reproduce the results table:
`uv run python -m barthel.evaluation --gold data/test/test.json --baselines --train data/barthel_train_source/train.json --preds "<label>=runs/<run>.json" ... --md runs/results_test.md`

**Notebook note:** the notebook is saved exactly as it was run on Colab, before the code was moved into the
`barthel/` package. It imports `prompt` and `eval` from files uploaded next to it (flat layout), i.e. the
former `prompt.py` and `eval.py`. To rerun it on the current code, upload the `barthel/` folder and change the
two imports to `barthel.prompt` and `barthel.evaluation`; the model and training settings are unaffected.
