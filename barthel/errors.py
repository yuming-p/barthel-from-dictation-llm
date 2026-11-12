"""Error analysis: every decision at least one shown run got wrong, with all runs side by side.

    uv run python -m barthel.errors            # table of wrong decisions + summaries
    uv run python -m barthel.errors --notes    # also print note text + gold evidence for each affected note

H0/H6 = Haiku zero-/6-shot, S0/S6 = Sonnet zero-/6-shot, GB = Gemma 3 4B untrained, GL = Gemma 3 4B + LoRA.
The row table lists decisions wrong in H0, H6, S0, S6 or GL; GB (wrong on ~half of everything) only appears in summaries.
Runs whose file is missing are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from barthel import evaluation as ev
from barthel import prompt as P

RUNS = {
    "H0": "runs/claude-haiku-5-5_test_0shot.json",
    "H6": "runs/claude-haiku-5-5_test_6shot.json",
    "S0": "runs/claude-sonnet-5-5_test_0shot.json",
    "S6": "runs/claude-sonnet-5-5_test_6shot.json",
    "GB": "runs/gemma3-4b-base_test.json",
    "GL": "runs/gemma3-4b-lora_test.json",
}
SHOWN = ["H0", "H6", "S0", "S6", "GL"]
CLAUDE = ["H0", "H6", "S0", "S6"]


def kind(target, pred) -> str:
    if pred is None:
        return "parse_fail"
    if target == ev.NA:
        return "hallucination"
    if pred == ev.NA:
        return "missed"
    return "wrong_score"


def short(v) -> str:
    return "NA" if v == ev.NA else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", action="store_true")
    args = ap.parse_args()

    gold = P.load_split("test")
    preds = {k: json.load(open(p, encoding="utf-8")) for k, p in RUNS.items() if os.path.exists(p)}
    missing = [k for k in RUNS if k not in preds]
    if missing:
        print("skipped (file missing):", ", ".join(missing))
    shown = [k for k in SHOWN if k in preds]
    claude = [k for k in CLAUDE if k in preds]

    rows = []
    for n in gold:
        for item in ev.ITEMS:
            g = n["items"][item]
            out = {k: (preds[k].get(n["note_id"]) or {}).get(item) for k in preds}
            if any(out[k] != g["target"] for k in shown):
                rows.append((n["note_id"], item, g, out))

    print(f"{len(rows)} decisions wrong in at least one of: {', '.join(shown)}\n")
    print(f"{'note':<10}{'item':<20}{'gold':>5} {'mention':<9}" + "".join(f"{k:>5}" for k in shown) + "  kinds")
    for nid, item, g, out in rows:
        kinds = sorted({kind(g["target"], out[k]) for k in shown if out[k] != g["target"]})
        print(f"{nid:<10}{item:<20}{short(g['target']):>5} {g['mention']:<9}"
              + "".join(f"{short(out[k]):>5}" for k in shown) + "  " + ",".join(kinds))

    print("\nwrong decisions per run (all 400 decisions, incl. runs not shown above):")
    for k in preds:
        c = Counter()
        for n in gold:
            for item in ev.ITEMS:
                t = n["items"][item]["target"]
                v = (preds[k].get(n["note_id"]) or {}).get(item)
                if v != t:
                    c[kind(t, v)] += 1
        print(f"  {k}: {sum(c.values()):>3}  {dict(c)}")
    if claude:
        all_claude = sum(1 for *_, g, out in rows if all(out[k] != g["target"] for k in claude))
        print(f"\nwrong in all {len(claude)} Claude runs: {all_claude}")

    if "GL" in preds:
        print("\nGemma + LoRA: over-scored items (gold = not_assessed but a score was given):")
        by_item, by_note = Counter(), Counter()
        for nid, item, g, out in rows:
            if g["target"] == ev.NA and out["GL"] not in (ev.NA, None):
                by_item[item] += 1
                by_note[nid] += 1
                print(f"  {nid} {item}: predicted {out['GL']}  (Claude runs: "
                      + ", ".join(f"{k}={short(out[k])}" for k in claude) + ")")
        print("  by item:", dict(by_item.most_common()))
        print("  by note:", dict(by_note.most_common()))

    print("\n0-shot vs 6-shot flips (same Claude model):")
    for z, s in (("H0", "H6"), ("S0", "S6")):
        if z in preds and s in preds:
            better = [1 for *_, g, o in rows if o[z] != g["target"] and o[s] == g["target"]]
            worse = [1 for *_, g, o in rows if o[z] == g["target"] and o[s] != g["target"]]
            print(f"  {z}->{s}: fixed by examples {len(better)}, broken by examples {len(worse)}")

    if args.notes:
        by_id = {n["note_id"]: n for n in gold}
        for nid in dict.fromkeys(r[0] for r in rows):
            n = by_id[nid]
            print(f"\n=== {nid} ===\n{n['note']}")
            for r in rows:
                if r[0] == nid:
                    print(f"  - {r[1]}: gold={short(r[2]['target'])} ({r[2]['mention']}), evidence={r[2].get('evidence')!r}")


if __name__ == "__main__":
    main()
