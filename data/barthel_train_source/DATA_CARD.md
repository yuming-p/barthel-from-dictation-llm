# Synthetic Barthel dataset: training split (v1)

## Purpose
Train and evaluate extraction of Barthel index item scores from German nursing dictations.
Fully synthetic: no real patient data.

## Contents
`train.json`: 200 examples. Each example has:
- `note`: German nursing dictation (hospital ward, morning round)
- `items`: one entry per Barthel item (official ICD-10-GM item names):
  - `target`: what a correct extractor should output: the score, or `"not_assessed"` if the note does not mention the item
  - `true_score`: the hidden ground-truth score (also defined for absent items; use for analysis only, **never as a training target**)
  - `mention`: `explicit` / `inferred` / `absent`
  - `evidence`: verbatim span from the note supporting the score (`null` if absent)
- metadata: `archetype`, `style`, `distractors`, `label_source`, `writer`

## Label semantics: `true_score` vs `target`
The dataset mirrors real clinical data, which has two layers:

| Field | Real-world analogue | Use |
|---|---|---|
| `note` | the nurse's dictation, usually incomplete | model input |
| `true_score` | the Barthel form the nurse fills in from everything she observed over 24–48 h, i.e. the patient's actual state | **analysis only**; never a training or evaluation target |
| `mention` / `target` | which items the text actually supports, and the score it justifies (`not_assessed` otherwise) | **training and evaluation target** |

Real paired data (dictation + completed Barthel form) only has the first two layers. A model trained
naively on `note → true_score` learns to fill in plausible scores for items nobody mentioned, i.e. to guess.
Here the third layer exists by construction; with real data it would need annotation (does the
dictation support this item?) before training.

`true_score` is kept to (1) keep each note internally consistent while writing it, (2) analyse
hallucinated scores (plausible inference from context vs. random invention), and (3) simulate the
nurse's answer if the agent asks a follow-up question about a missing item.

Possible experiment: train on `note → true_score` vs `note → target` and compare hallucination rates on absent items.

## Source files (`barthel_train_source.zip`)
| File | What it is |
|---|---|
| `generate.py` | Barthel level definitions, archetypes, label sampler (`sample_example`), API-based generator |
| `make_plans.py` | Samples the 200 label plans with seed 1 → writes `plans.json` |
| `plans.json` | The 200 plans: per item `true_score`, `mention`, `target`, level description; plus archetype, style, distractors. **Labels live here.** |
| `batches/batch_00.jsonl` … `batch_07.jsonl` | The notes, 25 per file, in id order (000–024, 025–049, …). Each line: `id`, `note`, `evidence` (item → verbatim span). No labels. |
| `validate.py` | Checks every batch line against its plan (evidence keys, verbatim spans, leak heuristics) |
| `merge.py` | Joins `plans.json` + `batches/` by id → `train.json`, prints statistics |

Pipeline: `make_plans.py` → `plans.json` → notes written per plan into `batches/` → `validate.py` → `merge.py` → `train.json`.
Labels and notes are kept separate on purpose: the labels are reproducible from the seed; the notes are not.

## Generation
1. **Labels first.** Plans were sampled with `sample_example()` from `generate.py`, `random.Random(1)`:
   5 patient archetypes, item scores sampled within the archetype, mention type per item
   (~45 % explicit / 20 % inferred / 35 % absent), 4 dictation styles, 2–4 unrelated observations.
2. **Notes.** Each note was written to its plan by Claude Opus 5.5 in a chat session (not via API,
   so the text itself is not exactly reproducible; the labels are).
3. **Scoring rules.** Score levels follow the Barthel index in ICD-10-GM (BfArM, Hamburg manual),
   condensed in `generate.py` (`BARTHEL` dict).

## Quality control
`validate.py`, run on all 200 notes:
- **Hard checks (0 errors):** every non-absent item has an evidence span; every span is a verbatim
  substring of the note; no evidence for absent items.
- **Leak heuristics (keyword search for absent items outside evidence spans):** 6 warnings, all
  checked by hand and false positives (e.g. "Bett-Stuhl" = chair, "Antibiotikum läuft").
- 22 notes are under 45 words (telegraphic style); kept on purpose.
- Not yet done: a full manual read-through of all notes for medical plausibility and score correctness.

## Statistics
- Words per note: min 26, median 58, max 131
- Mentions: 927 explicit, 375 inferred, 698 absent (2,000 item decisions)
- Archetypes: bedridden after stroke 44, wheelchair user with catheter 43, post-hip-surgery with rollator 41,
  mostly independent with mild confusion 37, frail/deconditioned 35

## Known limitations
- **Imbalanced levels:** a few score levels are rare, e.g. `baden_duschen = 5` (8 examples),
  `treppensteigen = 10` (9), `aufstehen_gehen = 15` (10). Per-level metrics on these will be noisy.
- **Single writer:** all notes come from one model. Phrasing is more regular than real dictations
  (recurring patterns like "Hilfe vor Ort nötig", "Treppe nicht möglich"). Write the **test set with a
  different model and prompts** and partly by hand to avoid rewarding style memorisation.
- **Home advantage:** do not use Claude Opus as the "frontier baseline" on data it wrote, or report it separately.
- **Clean text:** real input would come from speech recognition, with transcription errors. These notes have none.
- **Simplified rules:** level descriptions are condensed from the official manual; some edge cases
  (e.g. supervision → second-highest level) are represented only in a few notes.
- **Some archetype score combinations are unusual** (e.g. major transfer help but walks 50 m with a rollator),
  because scores are sampled per item within an archetype rather than from a joint model.