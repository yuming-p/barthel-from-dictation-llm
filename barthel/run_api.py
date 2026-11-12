"""Run a Claude model over a split and save predictions in evaluation.py format.

    uv run python -m barthel.run_api --model claude-haiku-5-5 --shots 0 --dry-run
    uv run python -m barthel.run_api --model claude-haiku-5-5 --split test --shots 6   # final run only

Auth: uses the default client (ANTHROPIC_API_KEY or the `ant auth login` profile).
Writes runs/<name>.json (predictions) and runs/<name>.meta.json (tokens, latency, cost).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from barthel import evaluation as ev
from barthel import prompt as P

# USD per 1M tokens (input, output); from the model table, checked 2026-11.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
RUNS = P.ROOT / "runs"


CACHE = {"type": "ephemeral"}  # 5-min TTL; every read refreshes it, so a continuous run stays warm
CACHE_WRITE_X, CACHE_READ_X = 1.25, 0.10  # multipliers on the base input price


def build_request(model: str, note: str, shots: int, cache: bool = True) -> dict:
    """Prompt caching is a prefix match: everything before the note must be byte-identical across requests.
    The breakpoint goes at the END of the shared part (system prompt, then the few-shot turns), never after
    the note, otherwise every request writes a new cache entry and none is ever read."""
    system: list | str = P.system_prompt()
    messages = P.few_shot_messages(shots)
    if cache:
        system = [{"type": "text", "text": system, "cache_control": CACHE}]
        if messages:  # last few-shot turn = end of the shared prefix
            last = messages[-1]
            messages[-1] = {**last, "content": [{"type": "text", "text": last["content"], "cache_control": CACHE}]}
    return dict(
        model=model,
        max_tokens=4000,  # headroom: adaptive thinking tokens count against this
        system=system,
        messages=messages + [{"role": "user", "content": note}],
        output_config={"format": {"type": "json_schema", "schema": P.output_schema()}},
    )


def call(client: anthropic.Anthropic, req: dict) -> dict:
    t0 = time.time()
    try:
        resp = client.messages.create(**req)
    except anthropic.APIError as e:
        return {"pred": None, "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0,
                "in": 0, "cache_write": 0, "cache_read": 0, "out": 0}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    u = resp.usage
    return {
        "pred": P.parse_output(text) if resp.stop_reason == "end_turn" else None,
        "error": None if resp.stop_reason == "end_turn" else f"stop_reason={resp.stop_reason}",
        "seconds": time.time() - t0,
        "in": u.input_tokens,  # uncached input only
        "cache_write": u.cache_creation_input_tokens or 0,
        "cache_read": u.cache_read_input_tokens or 0,
        "out": u.output_tokens,
    }


def cost_usd(results: list[dict], model: str) -> float:
    pin, pout = PRICES[model]
    return sum(
        r["in"] * pin + r["cache_write"] * pin * CACHE_WRITE_X + r["cache_read"] * pin * CACHE_READ_X + r["out"] * pout
        for r in results
    ) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["test", "val"], default="val",
                    help="iterate on val; run test once, at the end")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only first N notes (smoke test)")
    ap.add_argument("--no-cache", action="store_true", help="disable prompt caching (for cost comparison)")
    ap.add_argument("--dry-run", action="store_true", help="count input tokens of one request; no generation")
    args = ap.parse_args()

    notes = P.load_split(args.split)[: args.limit]
    client = anthropic.Anthropic()
    cache = not args.no_cache

    if args.dry_run:
        req = build_request(args.model, notes[0]["note"], args.shots, cache)
        n = client.messages.count_tokens(model=req["model"], system=req["system"], messages=req["messages"])
        pin, pout = PRICES[args.model]
        est = len(notes) * (n.input_tokens * pin + 300 * pout) / 1e6
        print(f"input tokens/note: {n.input_tokens} (excl. JSON schema); uncached cost for {len(notes)} notes: "
              f"${est:.3f} (assumes ~300 output tokens)")
        return

    def run(n: dict) -> dict:
        return call(client, build_request(args.model, n["note"], args.shots, cache))

    # First request alone: it writes the cache. Fanning out before that makes every worker write its own entry.
    results = [run(notes[0])]
    with ThreadPoolExecutor(args.workers) as ex:
        results += list(ex.map(run, notes[1:]))

    name = f"{args.model}_{args.split}_{args.shots}shot" + ("" if cache else "_nocache")
    if args.limit:
        name += f"_n{args.limit}"
    RUNS.mkdir(exist_ok=True)
    preds = {n["note_id"]: r["pred"] for n, r in zip(notes, results)}
    (RUNS / f"{name}.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")

    total = {k: sum(r[k] for r in results) for k in ("in", "cache_write", "cache_read", "out")}
    meta = {
        "model": args.model, "split": args.split, "shots": args.shots, "n_notes": len(notes), "cache": cache,
        "tokens": total,
        "usd_per_1000_notes": round(cost_usd(results, args.model) / len(notes) * 1000, 3),
        "mean_seconds_per_note": round(sum(r["seconds"] for r in results) / len(notes), 2),
        "errors": [{"note_id": n["note_id"], "error": r["error"]} for n, r in zip(notes, results) if r["error"]],
    }
    (RUNS / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "errors"}), f"errors={len(meta['errors'])}")
    print(ev.format_row(name, ev.evaluate(notes, preds), ev.bootstrap(notes, preds)))


if __name__ == "__main__":
    main()
