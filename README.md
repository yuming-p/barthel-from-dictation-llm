# Barthel index from nursing dictations: small fine-tuned model vs. frontier API

A small demo project: an LLM reads a German nursing dictation and fills in the **Barthel index**
(10 activities of daily living, 0–100 points). It answers one question:

> When does a small, fine-tuned open model match a rented frontier model on quality,
> and what does it gain in cost, speed and privacy?

All data is synthetic. No real patient data is used.

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
| Small open-weight model (Gemma 3 4B), untrained | none | Colab T4, 4-bit; self-hostable |
| Same small model + LoRA fine-tune | 180 notes (20 held out for validation) | trained and evaluated on a Colab T4; self-hostable |

"Self-hostable" means open weights that a hospital could run on its own infrastructure, so no patient text goes
to a third-party API. Everything here ran on a Colab T4 (Google's cloud); the model was **not** run on a laptop,
so no laptop latency or cost is claimed. Colab is acceptable only because the data is synthetic.

The test set is never used for tuning. The prompt, output schema and few-shot examples (`prompt.py`) were
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

*Frontier Claude on the 40 test notes (400 decisions).* Percentages; 95 % note-level bootstrap CI in
brackets. Generated by `eval.py` into `runs/results_test.md`. One run per row.

| Model | Accuracy | Precision | Recall | Hallucination ↓ | Score acc. explicit / inferred | USD / 1,000 notes | s / note |
|---|---|---|---|---|---|---|---|
| always `not_assessed` | 65.8 [63-68] | n/a | 0.0 [0-0] | 0.0 [0-0] | 0.0 / 0.0 | – | – |
| majority score per item | 13.2 [10-17] | 34.2 [32-37] | 100.0 [100-100] | 100.0 [100-100] | 34.4 / 46.8 | – | – |
| Haiku 5.5, zero-shot | 98.2 [97-99] | 100.0 [100-100] | 96.4 [93-99] | 0.0 [0-0] | 97.8 / 89.4 | 0.39 | 3.6 |
| Haiku 5.5, 6-shot | 97.2 [96-99] | 98.5 [96-100] | 96.4 [94-99] | 0.8 [0-2] | 97.8 / 85.1 | 0.35 | 3.8 |
| Sonnet 5.5, zero-shot | 98.2 [97-100] | 98.5 [96-100] | 97.1 [94-99] | 0.8 [0-2] | 98.9 / 91.5 | 5.37 | 3.8 |
| Sonnet 5.5, 6-shot | 97.5 [96-99] | 100.0 [100-100] | 93.4 [90-97] | 0.0 [0-0] | 96.7 / 85.1 | 4.67 | 3.0 |
| OpenAI | *not run* | | | | | | |
| Small open model, untrained | *not run* | | | | | – | – |
| Small open model + LoRA (180) | *not run* | | | | | – | – |

Notes on these numbers:
- **Costs are in USD** from measured token usage, with prompt caching on (the rules and JSON schema are a
  fixed ~2,750-token prefix, read at 10 % of the input price). Output tokens (mostly the model's thinking)
  are about 85–90 % of the remaining cost. Opus 5.5 was not evaluated because it wrote the training notes.
- **All four Claude runs are within noise of each other.** 7–11 of 400 decisions are wrong per run and the
  intervals overlap. Haiku matches Sonnet at about 1/14 of the cost. 6-shot is not better than zero-shot
  (6 decisions fixed, 13 broken across both models, not significant); run-to-run variance and the choice
  of the 6 examples (one draw, seed 0) have not been measured yet.
- **Error analysis:** `runs/error_analysis.txt` lists all 18 decisions that at least one run got wrong, with
  every run's answer, the note text and the gold evidence (regenerate with `uv run python errors.py --notes`).
  Only 2 are wrong in all four runs. Most errors are on `inferred` items; several are contested labels or
  scoring-rule boundary cases (e.g. pouring soup: gold 10, models say 5; "blieb im Bett" as a 0 for transfers)
  and some "hallucinations" are clinically plausible inferences (cannot stand, so stairs = 0). The test labels
  were deliberately **not** corrected after seeing these errors; contested ones are reported, not fixed.

## 5. Status

- [x] Barthel rules (ICD-10-GM) and label sampler (`data/generate_synthetic_data.py`)
- [x] Training set: 200 notes, validated, data card
- [x] First (template-based) test set reviewed and replaced by a free-text rewrite
- [ ] Check rewritten test set (format, sentence reuse, leaks)
- [x] Evaluation script + floor baselines + bootstrap CIs (`eval.py`, tested)
- [x] Shared prompt (rules in the prompt for all models; the `get_barthel_rule` tool was dropped from the main run)
- [x] Claude runs (Haiku 5.5, Sonnet 5.5; zero-shot and 6-shot) and first error analysis
- [ ] Noise check: repeat identical runs, 6-shot with other example draws
- [ ] OpenAI run (optional)
- [ ] Untrained small model
- [ ] LoRA fine-tuning on 180 notes
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
prompt.py            shared prompt, JSON schema, few-shot selection, data splits (train 180 / val 20 / test 40)
run_api.py           runs a Claude model over a split (prompt caching, cost + latency) -> runs/
eval.py              metrics, floor baselines, note-level bootstrap CIs, markdown results table
errors.py            error analysis across runs -> runs/error_analysis.txt
tests/test_eval.py   pytest for eval.py (metrics, parse failures, bootstrap, regression on the test set)
runs/                predictions (*.json), cost/latency (*.meta.json), results_test.md, error_analysis.txt
data/
  generate_synthetic_data.py   Barthel rules (BARTHEL, GENERAL_RULES), archetypes, label sampler, note generator
  barthel_train_source/        train.json (200 notes), plans.json, batches/, make_plans.py, validate.py,
                               merge.py, DATA_CARD.md (how the training data was made)
  test/                        test.json (40 notes), README_barthel_test_40.md
```

Reproduce the results table:
`uv run python eval.py --gold data/test/test.json --baselines --train data/barthel_train_source/train.json --preds "<label>=runs/<run>.json" ... --md runs/results_test.md`
