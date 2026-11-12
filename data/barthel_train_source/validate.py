"""Validate hand-written note batches against their label plans.

Hard errors: bad JSON, id mismatch, evidence keys != non-absent items,
evidence not a verbatim substring of the note.
Warnings: length out of range, distractor not found, possible leak of an
item that should be absent (keyword heuristic -> review by hand).
"""
import glob
import json
import re
import sys

plans = {p["note_id"][-3:]: p for p in json.load(open("plans.json"))}

DISTRACTOR_KW = {
    "Blutdruck und Puls": ["RR", "Blutdruck", "Puls", "P "],
    "Temperatur": ["Temp", "Fieber", "°"],
    "Schmerzskala": ["NRS", "Schmerz"],
    "Verbandswechsel": ["VW", "Verband"],
    "Medikamentengabe": ["Medi", "gegeben", "Tablette", "Antibiose", "AO"],
    "Trinkmenge": ["Trinkmenge", "getrunken", "ml"],
    "Stimmung / Schlaf in der Nacht": ["Nacht", "schlaf", "Schlaf", "Stimmung"],
    "Besuch von Angehörigen": ["Besuch", "Tochter", "Sohn", "Ehefrau", "Ehemann", "Angehörige", "Enkel", "Frau kommt", "Mann kommt"],
    "Blutzucker": ["BZ", "Blutzucker"],
    "Sauerstoffsättigung": ["SpO2", "Sättigung", "Sauerstoff"],
}

LEAK_KW = {
    "essen": [r"\bisst\b", r"\bEssen\b", r"Frühstück", r"angereicht", r"Tablett", r"\bPEG\b", r"Sonde"],
    "aufsetzen_umsetzen": [r"Bettkante", r"Transfer", r"Sessel", r"aus dem Bett", r"Lifter", r"Mobilisation", r"rübersetzen"],
    "sich_waschen": [r"Zähne", r"Zahn", r"Rasur", r"rasier", r"Waschbecken", r"\bWB\b", r"kämm", r"gekämmt"],
    "toilettenbenutzung": [r"Toilette", r"Steckbecken", r"\bWC\b"],
    "baden_duschen": [r"Dusch", r"dusch", r"Badewanne", r"Ganzkörper"],
    "aufstehen_gehen": [r"Rollator", r"Rollstuhl", r"\bRS\b", r"\bgeht\b", r"läuft", r"Gehen", r"gehen", r"Flur", r"Gehwagen"],
    "treppensteigen": [r"Treppe", r"Stockwerk", r"Etage", r"Stufen"],
    "an_auskleiden": [r"anzieh", r"Anzieh", r"angezogen", r"Hemd", r"Pulli", r"Bluse", r"Schuhe", r"Kleid", r"auskleid"],
    "stuhlkontinenz": [r"Stuhlgang", r"Abführ", r"Klysma", r"Laxans", r"Stoma", r"mit Stuhl", r"wegen Stuhl", r"Stuhl:", r"Stuhl kontinent", r"Stuhlinkontinen"],
    "harnkontinenz": [r"Urin", r"\bDK\b", r"Katheter", r"einnäss", r"nässt", r"Wasserlassen", r"Vorlage", r"harn"],
}

errors, warnings = [], []
seen = set()
for path in sorted(glob.glob("batches/batch_*.jsonl")):
    for ln, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{path}:{ln} bad JSON: {e}")
            continue
        nid = rec["id"]
        if nid in seen:
            errors.append(f"{nid}: duplicate")
        seen.add(nid)
        plan = plans[nid]
        note, ev = rec["note"], rec["evidence"]
        present = {i for i, d in plan["items"].items() if d["mention"] != "absent"}
        if set(ev) != present:
            errors.append(f"{nid}: evidence keys {sorted(set(ev) ^ present)} mismatch plan")
        for item, quote in ev.items():
            if quote not in note:
                errors.append(f"{nid}: evidence for {item} not in note: {quote!r}")
        n_words = len(note.split())
        if not 45 <= n_words <= 160:
            warnings.append(f"{nid}: {n_words} words")
        for d in plan["distractors"]:
            if not any(k in note for k in DISTRACTOR_KW[d]):
                warnings.append(f"{nid}: distractor missing? {d}")
        # leak check: strip evidence spans, then look for absent-item keywords
        rest = note
        for q in ev.values():
            rest = rest.replace(q, " ")
        for item in ITEMS if (ITEMS := list(plan["items"])) else []:
            if item in present:
                continue
            hits = [k for k in LEAK_KW[item] if re.search(k, rest)]
            if hits:
                warnings.append(f"{nid}: possible leak of absent '{item}': {hits}")

print(f"{len(seen)} notes checked")
print(f"{len(errors)} errors")
for e in errors:
    print("  ERR ", e)
print(f"{len(warnings)} warnings")
for w in warnings:
    print("  WARN", w)
sys.exit(1 if errors else 0)
