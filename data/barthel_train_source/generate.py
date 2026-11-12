"""
Generate synthetic German nursing dictations with Barthel index labels.

Only the generated notes are German (that's what the model will see in production);
code, comments, labels and logs are English.

Pipeline per example:
  1. sample a patient archetype -> Barthel item scores (labels come FIRST)
  2. plan how each item appears in the note: explicit / inferred / absent
  3. ask an LLM to write a German nurse dictation that follows the plan
  4. (optional) verify with a second LLM call; regenerate notes that break the plan
  5. save label + note to JSONL, log every prompt/response

Score levels: Barthel index as defined in ICD-10-GM (BfArM), based on the
"Hamburger Einstufungsmanual":
https://klassifikationen.bfarm.de/icd-10-gm/kode-suche/htmlgm2024/zusatz-06-anh-barthelindex.htm
The level descriptions below are condensed -- check them against the source before use.

Works with any OpenAI-compatible endpoint (OpenAI, OpenRouter, Gemini compat, local Ollama...).

Usage:
  python generate.py --n 3 --dry-run        # print prompts, no API calls
  python generate.py --n 300 --split train --model <writer-model> --seed 1 --verify-model <checker-model>
  python generate.py --n 40  --split test  --model <other-model>  --seed 2 --verify-model <checker-model> \
      --base-url <other-provider-url>
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# 1. Barthel items: allowed scores + level descriptions.
#    Item keys use the official German item names (English in comments).
#    Descriptions stay German because they are fed into the German note writer.
# --------------------------------------------------------------------------
BARTHEL = {
    "essen": {  # feeding
        10: "isst komplett selbstständig, oder versorgt PEG-Sonde selbstständig",
        5: "braucht Hilfe bei mundgerechter Vorbereitung (z.B. Brot schneiden), isst dann "
           "selbstständig; oder braucht Hilfe bei PEG-Ernährung",
        0: "isst nicht selbstständig und erhält keine Sondenernährung",
    },
    "aufsetzen_umsetzen": {  # sitting up and transfer (bed <-> chair/wheelchair)
        15: "setzt sich komplett selbstständig aus dem Liegen auf und wechselt in Stuhl/Rollstuhl und zurück",
        10: "braucht dabei Aufsicht oder geringe Hilfe durch einen Laien",
        5: "braucht erhebliche Hilfe (geschulter Laie oder Pflegekraft)",
        0: "wird faktisch nicht aus dem Bett transferiert",
    },
    "sich_waschen": {  # personal hygiene at the sink
        5: "wäscht sich am Waschbecken selbstständig, inkl. Zähneputzen, Rasieren, Kämmen",
        0: "braucht Hilfe bei Gesichtswäsche, Zähneputzen, Rasieren oder Kämmen",
    },
    "toilettenbenutzung": {  # toilet use
        10: "benutzt Toilette oder Toilettenstuhl komplett selbstständig, inkl. Spülung und Reinigung",
        5: "braucht vor Ort Hilfe oder Aufsicht bei der Toilettenbenutzung",
        0: "benutzt faktisch weder Toilette noch Toilettenstuhl",
    },
    "baden_duschen": {  # bathing / showering
        5: "badet oder duscht selbstständig, inkl. Ein-/Ausstieg, Waschen und Abtrocknen",
        0: "braucht Hilfe beim Baden oder Duschen",
    },
    "aufstehen_gehen": {  # rising and walking
        15: "steht selbstständig auf und geht mindestens 50 m ohne Gehwagen (Stock/Unterarmstützen erlaubt)",
        10: "steht selbstständig auf und geht mindestens 50 m mit Gehwagen/Rollator",
        5: "steht mit Laienhilfe oder Gehwagen auf und bewältigt Strecken im Wohnbereich; "
           "oder bewegt sich im Wohnbereich komplett selbstständig mit dem Rollstuhl",
        0: "erfüllt keines davon (kann weder gehen noch Rollstuhl selbst fahren)",
    },
    "treppensteigen": {  # stairs
        10: "steigt ohne Aufsicht oder Hilfe mindestens ein Stockwerk hinauf und hinunter (Hilfsmittel erlaubt)",
        5: "steigt ein Stockwerk mit Aufsicht oder Laienhilfe",
        0: "kann keine Treppe steigen",
    },
    "an_auskleiden": {  # dressing
        10: "zieht Alltagskleidung, Schuhe und ggf. Hilfsmittel selbstständig in angemessener Zeit an und aus",
        5: "kleidet mindestens den Oberkörper selbstständig an und aus, wenn Sachen bereitliegen",
        0: "braucht beim An- und Auskleiden weitgehend Hilfe",
    },
    "stuhlkontinenz": {  # bowel continence
        10: "stuhlkontinent, ggf. Abführmaßnahmen oder Stomaversorgung selbstständig",
        5: "im Durchschnitt höchstens einmal pro Woche stuhlinkontinent, oder braucht Hilfe "
           "bei Abführmaßnahmen/Stomaversorgung",
        0: "im Durchschnitt mehr als einmal pro Woche stuhlinkontinent",
    },
    "harnkontinenz": {  # bladder continence
        10: "harnkontinent, oder versorgt Dauerkatheter/Hilfsmittel komplett selbstständig",
        5: "kompensiert überwiegend selbstständig (höchstens einmal täglich Einnässen), "
           "oder braucht Hilfe bei der Versorgung des Kathetersystems",
        0: "im Durchschnitt mehr als einmal täglich harninkontinent",
    },
}
ITEMS = list(BARTHEL)

# General rules from the manual -- reuse these in the extraction prompt later.
GENERAL_RULES = [
    "If the person needs supervision or prompting (for safety or lack of drive), "
    "use the second-highest level of that item.",
    "If the condition varies (e.g. day to day), use the lower level.",
]

# --------------------------------------------------------------------------
# 2. Archetypes: allowed scores per item, so patients are internally coherent
#    (no one who can't transfer but climbs stairs alone).
# --------------------------------------------------------------------------
ARCHETYPES = {
    "post_hip_surgery_rollator": dict(
        essen=[10], aufsetzen_umsetzen=[10, 15], sich_waschen=[5], toilettenbenutzung=[5, 10],
        baden_duschen=[0], aufstehen_gehen=[10], treppensteigen=[0, 5], an_auskleiden=[5],
        stuhlkontinenz=[10], harnkontinenz=[10, 5],
    ),
    "bedridden_after_stroke": dict(
        essen=[0, 5], aufsetzen_umsetzen=[0, 5], sich_waschen=[0], toilettenbenutzung=[0],
        baden_duschen=[0], aufstehen_gehen=[0], treppensteigen=[0], an_auskleiden=[0],
        stuhlkontinenz=[0, 5], harnkontinenz=[0, 5],  # 5 = indwelling catheter managed by staff
    ),
    "mostly_independent_mild_confusion": dict(
        essen=[10, 5], aufsetzen_umsetzen=[15, 10], sich_waschen=[0, 5], toilettenbenutzung=[10, 5],
        baden_duschen=[0, 5], aufstehen_gehen=[15, 10], treppensteigen=[10, 5], an_auskleiden=[5, 10],
        stuhlkontinenz=[10], harnkontinenz=[5, 10],
    ),
    "wheelchair_user_catheter": dict(
        essen=[10], aufsetzen_umsetzen=[10, 5], sich_waschen=[5], toilettenbenutzung=[5, 0],
        baden_duschen=[0], aufstehen_gehen=[5], treppensteigen=[0], an_auskleiden=[5, 10],
        stuhlkontinenz=[10, 5], harnkontinenz=[5, 10],
    ),
    "frail_elderly_deconditioned": dict(
        essen=[5, 10], aufsetzen_umsetzen=[5, 10], sich_waschen=[0, 5], toilettenbenutzung=[5],
        baden_duschen=[0], aufstehen_gehen=[10, 5, 0], treppensteigen=[0], an_auskleiden=[5, 0],
        stuhlkontinenz=[10, 5], harnkontinenz=[5, 10, 0],
    ),
}

# Share of items per mention type. Absent items get target "not_assessed".
MENTION_WEIGHTS = {"explicit": 0.45, "inferred": 0.20, "absent": 0.35}

# Unrelated observations mixed into the note (German, since they appear in the text).
# Must NOT carry Barthel information: e.g. "Katheterpflege" would leak bladder,
# "Lagerung" (repositioning) would leak mobility -> absent items would stop being absent.
DISTRACTORS = [
    "Blutdruck und Puls", "Temperatur", "Schmerzskala", "Verbandswechsel",
    "Medikamentengabe", "Trinkmenge", "Stimmung / Schlaf in der Nacht",
    "Besuch von Angehörigen", "Blutzucker", "Sauerstoffsättigung",
]

STYLES = [
    "short, bullet-like fragments",
    "spoken, with filler words and small self-corrections",
    "structured shift handover",
    "rushed, with typical German nursing abbreviations (e.g. RR, Pat., Mobi, DK)",
]


def sample_example(rng: random.Random, idx: int, split: str) -> dict:
    archetype = rng.choice(list(ARCHETYPES))
    items = {}
    for item in ITEMS:
        score = rng.choice(ARCHETYPES[archetype][item])
        mention = rng.choices(list(MENTION_WEIGHTS), weights=MENTION_WEIGHTS.values())[0]
        items[item] = {
            "true_score": score,
            "mention": mention,
            "target": score if mention != "absent" else "not_assessed",
            "level_description": BARTHEL[item][score],
            "evidence": None,  # filled by the verifier if used
        }
    return {
        "note_id": f"{split}_{idx:04d}",
        "split": split,
        "archetype": archetype,
        "items": items,
        "distractors": rng.sample(DISTRACTORS, k=rng.randint(2, 4)),
        "style": rng.choice(STYLES),
    }


# --------------------------------------------------------------------------
# 3. Prompts (instructions in English, output in German)
# --------------------------------------------------------------------------
WRITER_SYSTEM = (
    "You write realistic, spoken-style nursing dictations from a German hospital ward. "
    "Write in natural German as a German nurse would speak, not textbook German. "
    "Never mention scores, scales, point values or the word 'Barthel'. "
    "Output only the dictation text."
)


def writer_prompt(ex: dict) -> str:
    lines = []
    for item, d in ex["items"].items():
        if d["mention"] == "explicit":
            lines.append(f"- {item}: describe clearly and directly -> {d['level_description']}")
        elif d["mention"] == "inferred":
            lines.append(f"- {item}: only HINT at this indirectly; a careful reader could infer "
                         f"it but it is never stated outright -> {d['level_description']}")
        else:
            lines.append(f"- {item}: do NOT mention or hint at this at all")
    # The archetype name is deliberately NOT given to the writer: "wheelchair user" or
    # "bedridden" would leak walking/transfer even when those items should be absent.
    return (
        f"Style: {ex['style']}. Length: 60-150 words. First person, as the nurse dictating "
        f"after the morning round. Language: German.\n\n"
        f"Daily-living information to include (level descriptions are in German):\n"
        + "\n".join(lines) + "\n\n"
        f"Also mention these unrelated observations: {', '.join(ex['distractors'])}.\n"
        f"Mix the order naturally; do not go item by item."
    )


VERIFIER_SYSTEM = (
    "You audit German nursing notes. For each listed activity, decide from the note ALONE "
    "whether it is stated explicitly, only implied, or not mentioned. Quote the supporting "
    "text verbatim (in German). Answer with JSON only."
)


def verifier_prompt(note: str) -> str:
    schema = {item: {"mention": "explicit|inferred|absent", "evidence": "quote or null"}
              for item in ITEMS}
    return (
        f"Activities (German Barthel item names): {', '.join(ITEMS)}\n\n"
        f"Note:\n\"\"\"\n{note}\n\"\"\"\n\n"
        f"Return JSON exactly in this shape:\n{json.dumps(schema, indent=1)}"
    )


# --------------------------------------------------------------------------
# 4. LLM call (OpenAI-compatible) with retries + logging
# --------------------------------------------------------------------------
def make_client(base_url: str | None):
    from openai import OpenAI  # pip install openai
    return OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "ollama"))


SLEEP_BETWEEN_CALLS = 0.0   # set via --sleep; use ~5-15 s on free tiers (low requests/minute)
NO_SYSTEM_ROLE = False      # set via --no-system-role; some models (e.g. Gemma via Gemini API) reject system messages


def call_llm(client, model, system, user, temperature, log_f, json_mode=False, retries=6):
    for attempt in range(retries):
        try:
            if NO_SYSTEM_ROLE:
                messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            else:
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": user}]
            kwargs = dict(model=model, temperature=temperature, messages=messages)
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content.strip()
            log_f.write(json.dumps({
                "time": datetime.now(timezone.utc).isoformat(), "model": model,
                "temperature": temperature, "system": system, "user": user,
                "response": text,
            }, ensure_ascii=False) + "\n")
            time.sleep(SLEEP_BETWEEN_CALLS)
            return text
        except Exception as e:  # noqa: BLE001
            # long backoff: free-tier rate-limit errors (HTTP 429) need tens of seconds
            wait = min(90, 5 * 2 ** attempt)
            print(f"  API error ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")


def check_plan(ex: dict, verdict: dict) -> list[str]:
    """Return list of items where the note broke the mention plan."""
    problems = []
    for item, d in ex["items"].items():
        got = verdict.get(item, {}).get("mention", "absent")
        planned = d["mention"]
        # absent must stay absent; explicit/inferred must be detectable at all.
        # explicit-vs-inferred confusion is tolerated (it's a soft boundary).
        if (planned == "absent") != (got == "absent"):
            problems.append(f"{item}: planned {planned}, verifier saw {got}")
        else:
            d["evidence"] = verdict.get(item, {}).get("evidence")
    return problems


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="REPLACE_WITH_WRITER_MODEL")
    ap.add_argument("--verify-model", default=None, help="enable verification step")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-regens", type=int, default=2)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to wait after each API call (free tiers: 5-15)")
    ap.add_argument("--no-system-role", action="store_true",
                    help="merge system prompt into user message (for models that reject system role)")
    args = ap.parse_args()

    global SLEEP_BETWEEN_CALLS, NO_SYSTEM_ROLE
    SLEEP_BETWEEN_CALLS = args.sleep
    NO_SYSTEM_ROLE = args.no_system_role

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)
    (out_dir / "generation_logs").mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "processed" / f"{args.split}.jsonl"
    log_path = out_dir / "generation_logs" / f"{args.split}_{args.seed}.jsonl"
    qc_path = out_dir / "qc_log.csv"

    # Save the run config next to the data -> this IS your documentation.
    config = vars(args) | {"started": datetime.now(timezone.utc).isoformat(),
                           "mention_weights": MENTION_WEIGHTS,
                           "archetypes": list(ARCHETYPES)}
    (out_dir / "generation_logs" / f"{args.split}_{args.seed}_config.json").write_text(
        json.dumps(config, indent=2))

    if args.dry_run:
        for i in range(args.n):
            ex = sample_example(rng, i, args.split)
            print("=" * 70, f"\n{ex['note_id']}  ({ex['archetype']})\n")
            print(writer_prompt(ex))
        return

    client = make_client(args.base_url)
    n_ok = n_discarded = 0
    with open(out_path, "a") as out_f, open(log_path, "a") as log_f, open(qc_path, "a") as qc_f:
        for i in range(args.n):
            ex = sample_example(rng, i, args.split)
            for attempt in range(args.max_regens + 1):
                note = call_llm(client, args.model, WRITER_SYSTEM, writer_prompt(ex),
                                args.temperature, log_f)
                if not args.verify_model:
                    break
                raw = call_llm(client, args.verify_model, VERIFIER_SYSTEM,
                               verifier_prompt(note), 0.0, log_f, json_mode=True)
                try:
                    problems = check_plan(ex, json.loads(raw))
                except json.JSONDecodeError:
                    problems = ["verifier returned invalid JSON"]
                if not problems:
                    break
                qc_f.write(f"{ex['note_id']},{attempt},regenerated,\"{'; '.join(problems)}\"\n")
            else:
                n_discarded += 1
                qc_f.write(f"{ex['note_id']},-,discarded,plan never satisfied\n")
                continue

            ex["note"] = note
            ex["writer_model"] = args.model
            ex["verify_model"] = args.verify_model
            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_ok += 1
            print(f"[{i + 1}/{args.n}] {ex['note_id']} ok")

    print(f"\nDone: {n_ok} saved to {out_path}, {n_discarded} discarded. Log: {log_path}")


if __name__ == "__main__":
    main()
