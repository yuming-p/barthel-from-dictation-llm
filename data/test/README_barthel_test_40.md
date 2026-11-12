# Synthetic Barthel Test Dataset

## Overview

`test.json` contains 40 synthetic German nursing dictations (`test_0001`–`test_0040`)
for evaluating structured extraction of the ten Barthel index items. The file is a
JSON array in the same format as `train.json`.

## Provenance

- **Author of the notes and labels:** OpenAI GPT-6 (ChatGPT), written in a chat session.
  No external model API was called.
- **Method:** each note and its labels were written by hand in ChatGPT, following the
  Barthel item definitions, score levels, patient archetypes and annotation principles
  in `generate_synthetic_data.py`.
- **Revision history:** a first version was assembled from a small set of template
  sentences (63 unique sentences across 380) and partly copied the wording of the
  scoring rules. It was replaced by this free-text version.
- **Quality review:** all 40 notes were reviewed against their labels by
  Claude Opus 5.5. Seven notes were corrected; each change is logged in that
  note's `qc_edits` field (see Quality control).

ChatGPT wrote the test notes and Claude wrote the training notes on purpose, so that
the fine-tuned model is not tested on the same writer's style it was trained on.

## Fields

Each example contains:

- `note`: the German nursing dictation
- `items`: one entry per Barthel item (`essen`, `aufsetzen_umsetzen`, `sich_waschen`,
  `toilettenbenutzung`, `baden_duschen`, `aufstehen_gehen`, `treppensteigen`,
  `an_auskleiden`, `stuhlkontinenz`, `harnkontinenz`), each with:
  - `target`: the score the note supports, or `not_assessed` (**the evaluation label**)
  - `true_score`: same as `target` for documented items; `null` for items the note
    does not mention (unlike `train.json`, no hidden patient state was generated)
  - `mention`: `explicit` / `inferred` / `absent`
  - `inference_type`: `direct_evidence` / `activity_based_evidence` / `insufficient_evidence`
  - `evidence`: verbatim span from the note supporting the score, or `null`
- `style`, `writer`, `label_source`, `qc_edits`: metadata

## Annotation principles

- A score is assigned only when the note provides sufficient supporting evidence.
- An item that is not documented is marked `not_assessed`; missing information
  must not be treated as evidence of independence or dependence.
- Evidence excerpts are verbatim substrings of the note.
- Activity descriptions support only the conclusion warranted by the observed
  activity. For example, transferring to a chair does not by itself establish
  independent walking.
- The wording is varied across notes: length, names, gender, vital signs and
  dictation style (structured handover, spoken with filler words, telegraphic
  with ward abbreviations).

## Statistics

- 40 notes, 46–90 words (median 59); 247 sentences, 242 unique
- 20 female and 16 male patients (the rest without a clear reference); 29 notes with
  ward abbreviations, 8 with filler words
- 400 item decisions: 90 explicit, 47 inferred, **263 absent (66 %)**

Documented items per Barthel item (scores in brackets):

| Item | Documented | Score distribution |
|---|---|---|
| essen | 24 | 0: 5, 5: 9, 10: 10 |
| aufsetzen_umsetzen | 22 | 0: 2, 5: 7, 10: 8, 15: 5 |
| sich_waschen | 14 | 0: 11, 5: 3 |
| toilettenbenutzung | 13 | 0: 4, 5: 7, 10: 2 |
| an_auskleiden | 13 | 0: 1, 5: 8, 10: 4 |
| aufstehen_gehen | 12 | 0: 1, 5: 5, 10: 4, 15: 2 |
| stuhlkontinenz | 11 | 0: 5, 5: 3, 10: 3 |
| baden_duschen | 10 | 0: 6, 5: 4 |
| treppensteigen | 9 | 0: 2, 5: 5, 10: 2 |
| harnkontinenz | 9 | 0: 1, 5: 6, 10: 2 |

**Compared with the training set** (35 % absent), far more items are undocumented.
The test set therefore weighs abstention more heavily: a model that tends to fill
in unmentioned items will be penalised more here than on training-like data.
With 9–24 documented cases per item, per-item results are noisy; report overall
metrics with confidence intervals rather than per-item scores.

## Quality control

Automated checks: every evidence span is a verbatim substring of its note, absent
items have no evidence and target `not_assessed`, all scores are valid levels.

Manual review corrected seven notes (`qc_edits`):

| Note | Issue | Fix |
|---|---|---|
| test_0004 | pronoun switched from er to sie for the same patient; evidence not verbatim | pronoun and evidence corrected |
| test_0005 | evidence not verbatim; dressing labelled 5 but only the upper body was described | evidence corrected; lower-body help added to the note |
| test_0007 | dressing described but labelled absent | labelled 5 (explicit) |
| test_0015 | "stayed in the room at a distance" implies supervision (10), label was 15; sink-hygiene evidence described washing back and legs (bathing) | wording adjusted to match labels |
| test_0017 | toilet use scored 0 only because urine drains via catheter | note made explicit (bedpan, no WC) |
| test_0019 | transfer described ("Mobi im Bett") but labelled absent | labelled 0 (inferred) |
| test_0032 | "stayed in bed all day" contradicted a transfer to the chair | changed to "mostly in bed" |

Known boundary cases, left unchanged and useful for error analysis:
- Opening a package or lid before eating is scored 5 (help with preparation);
  pouring soup or coffee is scored 10.
- `test_0006`: sink hygiene done "with prompting" is scored 0, following the rule
  that needing supervision or prompting gives the next-lower level.

## Limitations

- Synthetic prototype and evaluation data: not real patient data and not a
  clinically validated benchmark.
- Labels were prepared against condensed Barthel definitions and reviewed by a
  model, not by a nursing professional. A qualified reviewer should check labels
  and evidence before any clinical use.
- Both the notes and the labels come from one model family; a GPT model evaluated
  on this set may benefit from familiarity with the writing style.
- Notes are clean text, without speech-recognition errors.
- 40 notes give wide confidence intervals; only large differences between models
  are meaningful.
- Passing structural checks does not establish clinical correctness or
  suitability for production use.