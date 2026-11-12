"""Shared prompt, output schema and data splits. Every model (API or local) uses these."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "data"))
import generate_synthetic_data as _gen  # noqa: E402  (BARTHEL level texts, GENERAL_RULES)

from eval import ITEMS, LEVELS, NA  # noqa: E402

TRAIN_PATH = ROOT / "data" / "barthel_train_source" / "train.json"
TEST_PATH = ROOT / "data" / "test" / "test.json"
N_VAL = 20


def load_split(name: str) -> list[dict]:
    """'train' = first 180 train notes, 'val' = last 20 (by id order), 'test' = the 40 test notes."""
    if name == "test":
        return json.loads(TEST_PATH.read_text(encoding="utf-8"))
    notes = sorted(json.loads(TRAIN_PATH.read_text(encoding="utf-8")), key=lambda n: n["note_id"])
    return notes[:-N_VAL] if name == "train" else notes[-N_VAL:]


SYSTEM = """Du bist eine Assistenz für die Pflegedokumentation. Aus einem deutschen Pflege-Diktat füllst du \
den Barthel-Index (ICD-10-GM) aus.

Für jede der 10 Aktivitäten gibst du genau einen Wert zurück:
- die Punktzahl, wenn das Diktat sie belegt, oder
- "not_assessed", wenn das Diktat die Aktivität nicht erwähnt oder nicht genug aussagt.

WICHTIG: Rate nicht. Fehlende Information ist kein Beleg für Selbstständigkeit oder Hilfsbedürftigkeit. \
Eine beobachtete Tätigkeit belegt nur das, was sie tatsächlich zeigt (z. B. Transfer ins Bett sagt nichts \
über Gehen oder Treppensteigen). Vitalwerte und andere Beobachtungen sind für die Bewertung irrelevant.

Allgemeine Regeln:
{general}

Bewertungsstufen je Aktivität:
{levels}

Antworte nur mit dem JSON-Objekt (alle 10 Schlüssel, Werte als Strings: "0", "5", "10", "15" oder "not_assessed")."""


def _levels_text() -> str:
    out = []
    for item in ITEMS:
        out.append(f"\n{item}:")
        for score, text in sorted(_gen.BARTHEL[item].items(), reverse=True):
            out.append(f"  {score}: {text}")
    return "\n".join(out)


def system_prompt() -> str:
    return SYSTEM.format(general="\n".join(f"- {r}" for r in _gen.GENERAL_RULES), levels=_levels_text())


def output_schema() -> dict:
    props = {i: {"type": "string", "enum": sorted(str(s) for s in LEVELS[i]) + [NA]} for i in ITEMS}
    return {"type": "object", "properties": props, "required": ITEMS, "additionalProperties": False}


def target_json(note: dict) -> str:
    return json.dumps({i: str(note["items"][i]["target"]) for i in ITEMS}, ensure_ascii=False)


def few_shot_messages(k: int, seed: int = 0) -> list[dict]:
    """k training notes as prior user/assistant turns. Stratified over archetypes; train split only."""
    if k == 0:
        return []
    rng = random.Random(seed)
    by_arch: dict[str, list[dict]] = {}
    for n in load_split("train"):
        by_arch.setdefault(n["archetype"], []).append(n)
    archs = sorted(by_arch)
    picks = [rng.choice(by_arch[archs[j % len(archs)]]) for j in range(k)]
    rng.shuffle(picks)
    msgs = []
    for n in picks:
        msgs += [{"role": "user", "content": n["note"]}, {"role": "assistant", "content": target_json(n)}]
    return msgs


def parse_output(text: str) -> dict | None:
    """Model text -> {item: int | 'not_assessed'}; None if it is not a valid JSON object of the right shape.
    Individual bad values are passed through unchanged so eval.py counts them as parse failures."""
    if not isinstance(text, str):
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        # Lenient only about packaging: models without enforced output often wrap the JSON in a
        # markdown fence or add a sentence around it. Take the outermost {...} block; content is not repaired.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            raw = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    out = {}
    for i, v in raw.items():
        if isinstance(v, str) and v.lstrip("-").isdigit():
            v = int(v)
        out[i] = v
    return out
