import json, glob
from collections import Counter
plans = {p["note_id"][-3:]: p for p in json.load(open("plans.json"))}
out = []
for f in sorted(glob.glob("batches/batch_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        p = plans[r["id"]]
        items = {}
        for it, d in p["items"].items():
            items[it] = {"target": d["target"], "true_score": d["true_score"],
                         "mention": d["mention"], "evidence": r["evidence"].get(it)}
        out.append({"note_id": p["note_id"], "split": "train", "note": r["note"],
                    "items": items, "archetype": p["archetype"], "style": p["style"],
                    "distractors": p["distractors"],
                    "label_source": "generate.py sample_example, seed=1",
                    "writer": "Claude Opus 5.5 (chat session, hand-written per plan)"})
out.sort(key=lambda x: x["note_id"])
json.dump(out, open("train.json", "w"), ensure_ascii=False, indent=1)

# stats for the data card
mention = Counter(d["mention"] for e in out for d in e["items"].values())
words = [len(e["note"].split()) for e in out]
print(len(out), "examples;", "words min/median/max:", min(words), sorted(words)[len(words)//2], max(words))
print("mentions:", dict(mention))
print("archetypes:", dict(Counter(e["archetype"] for e in out)))
for it in out[0]["items"]:
    c = Counter(str(e["items"][it]["target"]) for e in out)
    print(f"  {it:20s}", dict(sorted(c.items())))
