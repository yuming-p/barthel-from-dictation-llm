import json, random, sys
sys.path.insert(0, "/home/claude")
from generate import sample_example, ITEMS
rng = random.Random(1)
plans = [sample_example(rng, i, "train") for i in range(200)]
json.dump(plans, open("plans.json", "w"), ensure_ascii=False, indent=1)
short = {"essen":"essen","aufsetzen_umsetzen":"transfer","sich_waschen":"waschen","toilettenbenutzung":"toilette",
 "baden_duschen":"duschen","aufstehen_gehen":"gehen","treppensteigen":"treppe","an_auskleiden":"kleiden",
 "stuhlkontinenz":"stuhl","harnkontinenz":"harn"}
st = {"short, bullet-like fragments":"STICH","spoken, with filler words and small self-corrections":"SPOKEN",
 "structured shift handover":"UEBERGABE","rushed, with typical German nursing abbreviations (e.g. RR, Pat., Mobi, DK)":"ABK"}
with open("plans_compact.txt","w") as f:
    for p in plans:
        parts=[f"{short[i]}={d['true_score']}{d['mention'][0].upper()}" for i,d in p["items"].items() if d["mention"]!="absent"]
        f.write(f"{p['note_id'][-3:]} {st[p['style']]} | {' '.join(parts)} | +{', '.join(p['distractors'])}\n")
from collections import Counter
print(Counter(p["archetype"] for p in plans))
print(Counter(d["mention"] for p in plans for d in p["items"].values()))
