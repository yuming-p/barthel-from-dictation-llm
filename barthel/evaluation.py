"""Evaluation for Barthel extraction: metrics, floor baselines, note-level bootstrap.

Gold format (train.json / test.json): list of notes, each with
    note_id, items[item] = {target, mention, ...}
    target is a score (int) or "not_assessed".

Prediction format: {note_id: {item: score | "not_assessed"}}.
A missing note, missing item, or invalid value counts as a parse failure and as wrong.

Usage:
    python -m barthel.evaluation --gold data/test/test.json --preds preds.json
    python -m barthel.evaluation --gold data/test/test.json --baselines --train data/barthel_train_source/train.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import Counter

NA = "not_assessed"
# Official score levels per item (ICD-10-GM Barthel; same as BARTHEL in data/.../generate.py).
LEVELS: dict[str, set] = {
    "essen": {0, 5, 10}, "aufsetzen_umsetzen": {0, 5, 10, 15}, "sich_waschen": {0, 5},
    "toilettenbenutzung": {0, 5, 10}, "baden_duschen": {0, 5}, "aufstehen_gehen": {0, 5, 10, 15},
    "treppensteigen": {0, 5, 10}, "an_auskleiden": {0, 5, 10}, "stuhlkontinenz": {0, 5, 10},
    "harnkontinenz": {0, 5, 10},
}
ITEMS = [
    "essen", "aufsetzen_umsetzen", "sich_waschen", "toilettenbenutzung", "baden_duschen",
    "aufstehen_gehen", "treppensteigen", "an_auskleiden", "stuhlkontinenz", "harnkontinenz",
]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _valid(value, levels: set) -> bool:
    return value == NA or (isinstance(value, int) and not isinstance(value, bool) and value in levels)


# ---------------------------------------------------------------- per-note counts

# Sufficient statistics per note; metrics are ratios of sums, so bootstrap is cheap.
COUNT_KEYS = [
    "n", "correct",                      # all decisions
    "absent", "absent_scored",           # gold NA / of those, model scored (hallucination)
    "present", "present_scored",         # gold scored / of those, model scored (recall)
    "scored",                            # model scored (precision denominator)
    "scored_present",                    # model scored AND gold scored (precision numerator)
    "explicit", "explicit_scored", "explicit_correct",   # gold mention type: total / model scored / exactly right
    "inferred", "inferred_scored", "inferred_correct",
    "parse_fail",
]


def note_counts(gold_note: dict, pred: dict | None, levels: dict[str, set] = LEVELS) -> dict[str, int]:
    c = dict.fromkeys(COUNT_KEYS, 0)
    for item, g in gold_note["items"].items():
        target = g["target"]
        raw = pred.get(item) if isinstance(pred, dict) else None
        ok_format = pred is not None and _valid(raw, levels[item])
        if not ok_format:
            c["parse_fail"] += 1
            p = None  # counted wrong, and counted as "not scored" for detection metrics
        else:
            p = raw
        scored = p is not None and p != NA
        c["n"] += 1
        right = ok_format and p == target
        c["correct"] += right
        if scored:
            c["scored"] += 1
            c["scored_present"] += target != NA
        if target == NA:
            c["absent"] += 1
            c["absent_scored"] += scored
        else:
            c["present"] += 1
            c["present_scored"] += scored
            m = g.get("mention")
            if m in ("explicit", "inferred"):
                c[m] += 1
                c[m + "_scored"] += scored
                c[m + "_correct"] += right
    return c


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def metrics_from_counts(c: dict[str, int]) -> dict[str, float]:
    return {
        "accuracy": _ratio(c["correct"], c["n"]),
        "hallucination": _ratio(c["absent_scored"], c["absent"]),
        "recall": _ratio(c["present_scored"], c["present"]),
        "precision": _ratio(c["scored_present"], c["scored"]),
        # Restricted to items whose gold mention type is explicit / inferred:
        #   recall_<m>    = share the model scored at all (detection)
        #   precision_<m> = of those it scored, share with exactly the right score
        #   score_acc_<m> = precision_<m> * recall_<m>
        "recall_explicit": _ratio(c["explicit_scored"], c["explicit"]),
        "recall_inferred": _ratio(c["inferred_scored"], c["inferred"]),
        "precision_explicit": _ratio(c["explicit_correct"], c["explicit_scored"]),
        "precision_inferred": _ratio(c["inferred_correct"], c["inferred_scored"]),
        "score_acc_explicit": _ratio(c["explicit_correct"], c["explicit"]),
        "score_acc_inferred": _ratio(c["inferred_correct"], c["inferred"]),
        "parse_fail_rate": _ratio(c["parse_fail"], c["n"]),
    }


def _sum(counts: list[dict[str, int]]) -> dict[str, int]:
    return {k: sum(c[k] for c in counts) for k in COUNT_KEYS}


def evaluate(gold: list[dict], preds: dict, levels: dict[str, set] | None = None) -> dict[str, float]:
    levels = levels or LEVELS
    return metrics_from_counts(_sum([note_counts(n, preds.get(n["note_id"]), levels) for n in gold]))


# ---------------------------------------------------------------- bootstrap

def _percentile(sorted_vals: list[float], q: float) -> float:
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _ci(values: list[float]) -> tuple[float, float]:
    vals = sorted(v for v in values if v == v)  # drop NaN (e.g. resample with no absent items)
    if not vals:
        return (float("nan"), float("nan"))
    return (_percentile(vals, 0.025), _percentile(vals, 0.975))


def bootstrap(gold, preds, n_boot=1000, seed=0, levels=None) -> dict[str, tuple[float, float]]:
    """95% CI per metric, resampling whole notes (items within a note are not independent)."""
    levels = levels or LEVELS
    per_note = [note_counts(n, preds.get(n["note_id"]), levels) for n in gold]
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        m = metrics_from_counts(_sum([per_note[rng.randrange(len(per_note))] for _ in per_note]))
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
    return {k: _ci(v) for k, v in draws.items()}


def paired_bootstrap(gold, preds_a, preds_b, n_boot=1000, seed=0, levels=None):
    """CI of metric(A) - metric(B), using the SAME resampled notes for both models."""
    levels = levels or LEVELS
    a = [note_counts(n, preds_a.get(n["note_id"]), levels) for n in gold]
    b = [note_counts(n, preds_b.get(n["note_id"]), levels) for n in gold]
    rng = random.Random(seed)
    diffs: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = [rng.randrange(len(gold)) for _ in gold]
        ma = metrics_from_counts(_sum([a[i] for i in idx]))
        mb = metrics_from_counts(_sum([b[i] for i in idx]))
        for k in ma:
            diffs.setdefault(k, []).append(ma[k] - mb[k])
    return {k: _ci(v) for k, v in diffs.items()}


# ---------------------------------------------------------------- baselines

def baseline_always_na(gold: list[dict]) -> dict:
    return {n["note_id"]: dict.fromkeys(ITEMS, NA) for n in gold}


def baseline_majority(gold: list[dict], train: list[dict]) -> dict:
    """Per item, always predict the most frequent *score* in train (never abstains)."""
    mode = {}
    for item in ITEMS:
        cnt = Counter(n["items"][item]["target"] for n in train if n["items"][item]["target"] != NA)
        mode[item] = cnt.most_common(1)[0][0]
    return {n["note_id"]: dict(mode) for n in gold}


# ---------------------------------------------------------------- reporting

def format_row(name: str, m: dict, ci: dict | None = None) -> str:
    def f(k):
        s = f"{100 * m[k]:5.1f}"
        if ci:
            s += f" [{100 * ci[k][0]:.0f}-{100 * ci[k][1]:.0f}]"
        return s
    return (f"{name:<22} acc {f('accuracy')} | halluc {f('hallucination')} | recall {f('recall')} | "
            f"prec {f('precision')} | parse-fail {100 * m['parse_fail_rate']:.1f}")


def load_meta(preds_path: str) -> dict | None:
    """barthel.run_api writes <name>.meta.json next to <name>.json (cost, latency)."""
    p = Path(preds_path).with_suffix(".meta.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def markdown_table(rows: list[tuple[str, dict, dict, dict | None]]) -> str:
    """rows: (name, metrics, bootstrap CIs, meta or None). Percentages with 95% note-level CI in brackets."""
    def pct(m, ci, k):
        return "n/a" if m[k] != m[k] else f"{100 * m[k]:.1f} [{100 * ci[k][0]:.0f}-{100 * ci[k][1]:.0f}]"

    def plain(m, k):
        return "n/a" if m[k] != m[k] else f"{100 * m[k]:.1f}"

    out = ["| Model | Accuracy | Precision | Recall | Hallucination | Precision expl. / infer. "
           "| Recall expl. / infer. | USD / 1,000 notes | s / note |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m, ci, meta in rows:
        cost = f"{meta['usd_per_1000_notes']:.2f}" if meta else "–"
        secs = f"{meta['mean_seconds_per_note']:.1f}" if meta else "–"
        out.append(f"| {name} | {pct(m, ci, 'accuracy')} | {pct(m, ci, 'precision')} | {pct(m, ci, 'recall')} "
                   f"| {pct(m, ci, 'hallucination')} | {plain(m, 'precision_explicit')} / "
                   f"{plain(m, 'precision_inferred')} | {plain(m, 'recall_explicit')} / "
                   f"{plain(m, 'recall_inferred')} | {cost} | {secs} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--preds", nargs="*", default=[], help="prediction JSON file(s), optionally label=path")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--train", help="train.json, needed for the majority baseline")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", help="also write a markdown results table to this path")
    args = ap.parse_args()

    gold = load(args.gold)
    levels = LEVELS
    runs: dict[str, dict] = {}
    if args.baselines:
        runs["always not_assessed"] = baseline_always_na(gold)
        if args.train:
            runs["majority score"] = baseline_majority(gold, load(args.train))
    paths: dict[str, str] = {}
    for spec in args.preds:  # "label=path" or just "path"
        label, _, path = spec.partition("=")
        label, path = (label, path) if path else (spec, spec)
        runs[label] = json.load(open(path, encoding="utf-8"))
        paths[label] = path

    rows = []
    for name, preds in runs.items():
        m, ci = evaluate(gold, preds, levels), bootstrap(gold, preds, args.n_boot, args.seed, levels)
        print(format_row(name, m, ci))
        rows.append((name, m, ci, load_meta(paths[name]) if name in paths else None))
    if args.md:
        Path(args.md).write_text(markdown_table(rows) + "\n", encoding="utf-8")
        print(f"wrote {args.md}")

if __name__ == "__main__":
    main()
