"""Error analysis: every decision at least one run got wrong, with all runs side by side.

    uv run python errors.py            # table of wrong decisions
    uv run python errors.py --notes    # also print note text + gold evidence for each affected note
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import eval as ev
import prompt as P

RUNS = {
    "H0": "runs/claude-haiku-5-5_test_0shot.json",
    "H6": "runs/claude-haiku-5-5_test_6shot.json",
    "S0": "runs/claude-sonnet-5-5_test_0shot.json",
    "S6": "runs/claude-sonnet-5-5_test_6shot.json",
}


def kind(target, pred) -> str:
    if pred is None:
        return "parse_fail"
    if target == ev.NA:
        return "hallucination"
    if pred == ev.NA:
        return "missed"
    return "wrong_score"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", action="store_true")
    args = ap.parse_args()

    gold = P.load_split("test")
    preds = {k: json.load(open(p, encoding="utf-8")) for k, p in RUNS.items()}
    rows = []
    for n in gold:
        for item in ev.ITEMS:
            g = n["items"][item]
            out = {k: (preds[k].get(n["note_id"]) or {}).get(item) for k in RUNS}
            if any(v != g["target"] for v in out.values()):
                rows.append((n["note_id"], item, g, out))

    print(f"{len(rows)} decisions wrong in at least one run\n")
    print(f"{'note':<10}{'item':<20}{'gold':>6} {'mention':<9}" + "".join(f"{k:>5}" for k in RUNS) + "  kinds")
    for nid, item, g, out in rows:
        kinds = sorted({kind(g["target"], v) for v in out.values() if v != g["target"]})
        print(f"{nid:<10}{item:<20}{str(g['target']):>6} {g['mention']:<9}"
              + "".join(f"{str(v):>5}" for v in out.values()) + "  " + ",".join(kinds))

    wrong_by_run = {k: sum(1 for *_, g, out in rows if out[k] != g["target"]) for k in RUNS}
    all4 = sum(1 for *_, g, out in rows if all(v != g["target"] for v in out.values()))
    print(f"\nwrong per run: {wrong_by_run}; wrong in all 4 runs: {all4}")
    print("error kinds (counted per run):",
          dict(Counter(kind(g["target"], v) for *_, g, out in rows for v in out.values() if v != g["target"])))

    print("\n0-shot vs 6-shot flips (same model):")
    for z, s in (("H0", "H6"), ("S0", "S6")):
        better = [(nid, it) for nid, it, g, o in rows if o[z] != g["target"] and o[s] == g["target"]]
        worse = [(nid, it) for nid, it, g, o in rows if o[z] == g["target"] and o[s] != g["target"]]
        print(f"  {z}->{s}: fixed by examples {len(better)}, broken by examples {len(worse)}")
        for nid, it in worse:
            print("     broken:", nid, it)

    if args.notes:
        by_id = {n["note_id"]: n for n in gold}
        for nid in dict.fromkeys(r[0] for r in rows):
            n = by_id[nid]
            print(f"\n=== {nid} ===\n{n['note']}")
            for r in rows:
                if r[0] == nid:
                    print(f"  - {r[1]}: gold={r[2]['target']} ({r[2]['mention']}), evidence={r[2].get('evidence')!r}")


if __name__ == "__main__":
    main()
