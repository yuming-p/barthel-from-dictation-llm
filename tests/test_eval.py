import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import eval as ev

NA = ev.NA
ROOT = os.path.join(os.path.dirname(__file__), "..")


def make_note(nid, targets, mentions=None):
    """targets: dict item -> target; unspecified items are not_assessed/absent."""
    mentions = mentions or {}
    items = {}
    for i in ev.ITEMS:
        t = targets.get(i, NA)
        m = mentions.get(i, "absent" if t == NA else "explicit")
        items[i] = {"target": t, "mention": m}
    return {"note_id": nid, "items": items}


def all_na():
    return dict.fromkeys(ev.ITEMS, NA)


@pytest.fixture
def gold():
    return [
        make_note("a", {"essen": 10, "sich_waschen": 5}, {"sich_waschen": "inferred"}),
        make_note("b", {"essen": 0}),
    ]


def test_perfect_predictions(gold):
    preds = {n["note_id"]: {i: v["target"] for i, v in n["items"].items()} for n in gold}
    m = ev.evaluate(gold, preds)
    assert m["accuracy"] == 1 and m["recall"] == 1 and m["precision"] == 1
    assert m["hallucination"] == 0 and m["parse_fail_rate"] == 0


def test_always_na_baseline(gold):
    m = ev.evaluate(gold, ev.baseline_always_na(gold))
    assert m["accuracy"] == pytest.approx(17 / 20)  # 3 scored items of 20 decisions
    assert m["hallucination"] == 0
    assert m["recall"] == 0
    assert m["precision"] != m["precision"]  # NaN: nothing was scored


def test_hallucination_and_wrong_score(gold):
    p_a = {**all_na(), "essen": 10, "sich_waschen": 0, "baden_duschen": 5}  # 1 wrong score, 1 hallucination
    p_b = {**all_na(), "essen": 0}
    m = ev.evaluate(gold, {"a": p_a, "b": p_b})
    assert m["hallucination"] == pytest.approx(1 / 17)  # 17 absent items, 1 scored
    assert m["recall"] == pytest.approx(3 / 3)
    assert m["precision"] == pytest.approx(3 / 4)       # 4 scored, 3 present
    assert m["accuracy"] == pytest.approx(18 / 20)      # wrong: sich_waschen, baden_duschen
    assert m["score_acc_explicit"] == pytest.approx(2 / 2)
    assert m["score_acc_inferred"] == pytest.approx(0 / 1)


def test_missing_note_and_invalid_values_are_parse_failures(gold):
    preds = {"a": {**all_na(), "essen": 7, "sich_waschen": "yes"}}  # "b" missing, invalid values in "a"
    m = ev.evaluate(gold, preds)
    assert m["parse_fail_rate"] == pytest.approx((2 + 10) / 20)
    assert m["recall"] == 0 and m["hallucination"] == 0  # failures are never "scored"


def test_missing_item_is_parse_failure(gold):
    p = {i: NA for i in ev.ITEMS if i != "essen"}
    m = ev.evaluate(gold[:1], {"a": p})
    assert m["parse_fail_rate"] == pytest.approx(1 / 10)


def test_bool_is_not_a_valid_score(gold):
    m = ev.evaluate(gold[:1], {"a": {**all_na(), "essen": True}})
    assert m["parse_fail_rate"] == pytest.approx(1 / 10)


def test_majority_baseline_uses_train_mode_and_never_abstains(gold):
    train = [make_note(str(k), {i: 5 for i in ev.ITEMS}) for k in range(3)] + \
            [make_note("x", {i: 10 for i in ev.ITEMS})]
    preds = ev.baseline_majority(gold, train)
    assert all(v == 5 for v in preds["a"].values())
    assert ev.evaluate(gold, preds)["hallucination"] == 1


def test_bootstrap_is_deterministic_and_seed_sensitive():
    rng = random.Random(0)
    g = [make_note(f"n{k}", {i: rng.choice([0, 5, NA]) for i in ev.ITEMS},
                   {i: "inferred" for i in ev.ITEMS[:5]}) for k in range(15)]
    preds = {n["note_id"]: {i: rng.choice([0, 5, NA]) for i in ev.ITEMS} for n in g}
    a = ev.bootstrap(g, preds, n_boot=200, seed=1)
    assert a == ev.bootstrap(g, preds, n_boot=200, seed=1)
    assert a != ev.bootstrap(g, preds, n_boot=200, seed=2)


def test_bootstrap_ci_contains_point_estimate():
    rng = random.Random(0)
    g, preds = [], {}
    for k in range(40):
        t = {i: rng.choice([0, 5, 10, NA]) for i in ev.ITEMS}
        g.append(make_note(f"n{k}", t))
        preds[f"n{k}"] = {i: (v if rng.random() < 0.8 else NA) for i, v in t.items()}
    point = ev.evaluate(g, preds)["accuracy"]
    lo, hi = ev.bootstrap(g, preds, n_boot=500)["accuracy"]
    assert lo <= point <= hi


def test_bootstrap_resamples_notes_not_items():
    # One note all right, one note with a wrong item: note-level resampling gives accuracy
    # in {0.9, 0.95, 1.0}; this pins the unit of resampling to the note.
    g = [make_note("good", {"essen": 10}), make_note("bad", {"essen": 10})]
    p = {"good": {**all_na(), "essen": 10}, "bad": {**all_na(), "essen": 5}}
    lo, hi = ev.bootstrap(g, p, n_boot=500)["accuracy"]
    assert lo == pytest.approx(0.9) and hi == pytest.approx(1.0)


def test_paired_bootstrap_identical_models_have_zero_diff(gold):
    preds = ev.baseline_always_na(gold)
    assert ev.paired_bootstrap(gold, preds, preds, n_boot=100)["accuracy"] == (0.0, 0.0)


def test_paired_bootstrap_detects_clear_winner():
    g = [make_note(f"n{k}", {"essen": 10}) for k in range(20)]
    good = {n["note_id"]: {**all_na(), "essen": 10} for n in g}
    lo, _ = ev.paired_bootstrap(g, good, ev.baseline_always_na(g), n_boot=200)["accuracy"]
    assert lo > 0


def test_levels_are_official_not_inferred_from_gold(gold):
    # 5 never occurs for baden_duschen in this gold, but it is a valid level: wrong, not a parse failure
    m = ev.evaluate(gold[:1], {"a": {**all_na(), "baden_duschen": 5}})
    assert m["parse_fail_rate"] == 0 and m["hallucination"] > 0
    # 10 is not a valid level for baden_duschen
    m = ev.evaluate(gold[:1], {"a": {**all_na(), "baden_duschen": 10}})
    assert m["parse_fail_rate"] == pytest.approx(1 / 10)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data/test/test.json")), reason="data missing")
def test_real_test_set_always_na_baseline():
    """Regression: pins the numbers quoted in data/test/README_barthel_test_40.md."""
    gold = ev.load(os.path.join(ROOT, "data/test/test.json"))
    assert len(gold) == 40
    m = ev.evaluate(gold, ev.baseline_always_na(gold))
    assert m["accuracy"] == pytest.approx(263 / 400)
    assert m["hallucination"] == 0 and m["recall"] == 0


def test_markdown_table_formats_rows_and_missing_meta(gold):
    preds = ev.baseline_always_na(gold)
    m, ci = ev.evaluate(gold, preds), ev.bootstrap(gold, preds, n_boot=50)
    table = ev.markdown_table([("floor", m, ci, None), ("api", m, ci, {"usd_per_1000_notes": 0.389, "mean_seconds_per_note": 3.56})])
    lines = table.splitlines()
    assert len(lines) == 4 and lines[2].startswith("| floor |") and lines[2].endswith("| – | – |")
    assert "0.39 | 3.6 |" in lines[3]
    assert "n/a" in lines[2]  # precision is undefined when nothing was scored
