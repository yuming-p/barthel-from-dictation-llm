# Barthel index from nursing dictations: small fine-tuned model vs. frontier API

A small demo project inspired by [voize case study](https://www.voize.ai/de/loesungen/krankenhaus#case-study): an LLM reads a German nursing dictation and fills in the **Barthel index**
(10 activities of daily living, 0–100 points). It answers one question:

> When does a small, fine-tuned open model match a rented frontier model on quality,
> and what does it gain in cost, speed and privacy?

All data is synthetic. No real patient data is used.

## Summary

Results on 40 held-out test notes (400 item decisions; details and confidence intervals in section 4):

| Model | Accuracy | Precision | Recall | Hallucination | USD / 1,000 notes |
|---|---|---|---|---|---|
| always `not_assessed` (floor) | 65.8 % | n/a | 0.0 % | 0.0 % | – |
| Claude Haiku 5.5, zero-shot | 98.2 % | 100.0 % | 96.4 % | 0.0 % | 0.39 |
| Claude Sonnet 5.5, zero-shot | 98.2 % | 98.5 % | 97.1 % | 0.8 % | 5.37 |
| Gemma 3 4B, untrained | 54.0 % | 50.0 % | 83.2 % | 43.3 % | ~1.0 (inference only, assumed GPU price) |
| Gemma 3 4B + LoRA, 180 notes | 88.0 % | 90.5 % | 97.1 % | 5.3 % | ~0.95 (inference only, assumed GPU price; plus ~$0.27 one-off training) |

Precision = of the items the model scored, the share the note actually supports (low = guessing). Recall = of
the items the note supports, the share the model scored (low = over-cautious). Accuracy alone is inflated by
abstention: 66 % of the decisions are `not_assessed`, so a model that always abstains already scores 65.8 %.

- **Fine-tuning on 180 notes closes most, not all, of the gap.** Frontier models are close to the ceiling on this task. LoRA lifts Gemma by 34 points (hallucination
  43 % to 5 %) but stays about 10 points behind Claude. The largest error type is wrong score levels
  (off by one level), not hallucination.
- **Answer to the question:** on this task a small fine-tuned model gets within about 10 points of a frontier
  API but does not match it, and it does not gain on cost or speed against Haiku. (cost of gemma3 for this demo is 0$ since it was ran on colab with free tier). Its case is **self-hosting and privacy**: open weights that can run
  without sending patient text to a third party. The two also fail in opposite ways. **Claude is cautious:** when
  the evidence is indirect it often abstains (hallucination 0-0.8 %, but recall on inferred items only 87-94 %).
  **The fine-tuned Gemma commits:** it scores nearly everything that looks related (5.3 % hallucination on
  unmentioned items, 98 % recall on inferred items), and about a third of its scores on inferred items are the
  wrong level (63 % precision). 
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
| `data/barthel_train_source/train.json` | 200 notes | Claude Opus 5.5 | labels sampled first (seed 1), notes written per plan; see [`data/barthel_train_source/DATA_CARD.md`](data/barthel_train_source/DATA_CARD.md). 180 for training, 20 for validation |
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


Everything here ran on a free cloud T4 notebook (Colab or Kaggle); the model was **not** run on a laptop,
so no laptop latency or cost is claimed. A hosted notebook is acceptable only because the data is synthetic.

The test set is never used for tuning. The prompt, output schema and few-shot examples (`barthel/prompt.py`) were
frozen before the first test run. If a later change is needed, both versions are reported; the first is not
replaced. Few-shot examples come from the 180 training notes only.

**Home advantage (caveat of synthetic data).** 
Claude Opus 5.5 wrote the training notes (used for few-shot examples and fine-tuning), ChatGPT 6 wrote the test notes. Claude Haiku and Sonnet are used for execution of the task.
A model may do better on text in its own family's style. 
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
(unconstrained) greedy generation and a free-tier T4 GPU. Their cost is **inference only**, from measured
seconds per note at an assumed $0.50 / GPU hour. It excludes model loading and idle time and the one-off
LoRA training (32 min, about $0.27 at the same price). Notes were generated one at a time (batch size 1), so
it is an upper bound; batched serving would be cheaper per note but was not measured.

| Model | Accuracy | Precision | Recall | Hallucination ↓ | Precision expl. / infer. | Recall expl. / infer. | USD / 1,000 notes | s / note |
|---|---|---|---|---|---|---|---|---|
| always `not_assessed` | 65.8 [63-68] | n/a | 0.0 [0-0] | 0.0 [0-0] | n/a / n/a | 0.0 / 0.0 | – | – |
| majority score per item | 13.2 [10-17] | 34.2 [32-37] | 100.0 [100-100] | 100.0 [100-100] | 34.4 / 46.8 | 100.0 / 100.0 | – | – |
| Haiku 5.5, zero-shot | 98.2 [97-99] | 100.0 [100-100] | 96.4 [93-99] | 0.0 [0-0] | 100.0 / 95.5 | 97.8 / 93.6 | 0.39 | 3.6 |
| Haiku 5.5, 6-shot | 97.2 [96-99] | 98.5 [96-100] | 96.4 [94-99] | 0.8 [0-2] | 100.0 / 90.9 | 97.8 / 93.6 | 0.35 | 3.8 |
| Sonnet 5.5, zero-shot | 98.2 [97-100] | 98.5 [96-100] | 97.1 [94-99] | 0.8 [0-2] | 100.0 / 97.7 | 98.9 / 93.6 | 5.37 | 3.8 |
| Sonnet 5.5, 6-shot | 97.5 [96-99] | 100.0 [100-100] | 93.4 [90-97] | 0.0 [0-0] | 100.0 / 97.6 | 96.7 / 87.2 | 4.67 | 3.0 |
| Gemma 3 4B, untrained | 54.0 [48-59] | 50.0 [46-54] | 83.2 [76-90] | 43.3 [37-51] | 62.5 / 52.4 | 80.0 / 89.4 | 1.01 | 7.3 |
| Gemma 3 4B + LoRA (180 notes) | 88.0 [85-91] | 90.5 [86-95] | 97.1 [93-100] | 5.3 [3-8] | 85.1 / 63.0 | 96.7 / 97.9 | 0.95 | 6.8 |
| OpenAI | *not run* | | | | | | |

The two "expl. / infer." columns are restricted to the items the gold labels mark as explicit or
inferred: *recall* = share of those items the model scored at all; *precision* = of the items it scored, the
share with exactly the right level (so precision × recall = score accuracy for that type). The headline
Precision / Recall columns are about detection only: is the item supported by the note at all.


Notes on these numbers:
- **Costs are in USD** from measured token usage, with prompt caching on (the rules and JSON schema are a
  fixed ~2,750-token prefix, read at 10 % of the input price). Output tokens (mostly the model's thinking)
  are about 85–90 % of the remaining cost. Opus 5.5 was not evaluated because it wrote the training notes.
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

- **Fine-tuning** settings were fixed before any test run: LoRA r=16, alpha 16, lr 2e-4, 3 epochs, effective batch
  16, loss on the answer only; `val` was used for monitoring only. Single run per setting; run-to-run
  variance is unmeasured.


## 5. Limitations

- **Label quality is checked automatically, not by a human.** Notes whose text does not clearly express
  their label count as model errors, so no model can reach 100 %; the true ceiling is unknown.
- **Home advantage:** see section 3.
- Synthetic notes are cleaner than real dictations (no speech-recognition errors, more regular phrasing).
- Labels assume the dictation is clinically correct.
- Some score levels are rare in the training data (see [`data/barthel_train_source/DATA_CARD.md`](data/barthel_train_source/DATA_CARD.md)).
- **Small samples, everywhere.**
  - *Test:* 40 notes = 400 decisions, of which only 137 have a documented score (90 explicit, 47 inferred).
    Confidence intervals are wide (e.g. Gemma + LoRA accuracy 85-91 %), so differences of a few points,
    such as between the Claude runs, are not findings. Per-item results rest on only 9-24 documented cases
    each (see [`data/test/README_barthel_test_40.md`](data/test/README_barthel_test_40.md)), so none are reported.
  - *Training and validation:* 180 notes to train and 20 to validate. The 20 validation notes are 200
    decisions, so one decision moves the score by 0.5 points and validation numbers are very noisy. How much
    more data would help the small model is unknown (no learning curve was run).
  - *Repeats:* one run per setting, one draw of the few-shot examples; run-to-run variance is unmeasured.


## 6. Repository layout

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
