import json, os, re, sys
B = sys.argv[1] if len(sys.argv) > 1 else "."
SETS = ["conll03_test", "wnut17_test", "conll02_es_test", "conll02_nl_test", "tab_test"]
TITLE = re.compile(r"^(?:(?:Mr|Mrs|Ms|Miss|Dr|Prof|Lord|Lady|Sir|Judge|President|Colonel|General|Senator)\.? )+")
def strip(t, a, b):
    m = TITLE.match(t[a:b])
    return (a + m.end(), b) if m and a + m.end() < b else (a, b)
def load(p): return {r["id"]: r for r in map(json.loads, open(p))}
print("| Set | Model | Exact P | Exact R | Exact F1 | F1, titles ignored | Names fully masked | Names missed (0 chars) | Name chars masked | Masked chars that are names |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s in SETS:
    gold = load(f"{B}/data/{s}.jsonl")
    for m in ("veil", "spacy", "newmean-nano", "newmean-small", "newmean-large"):
        if not os.path.exists(f"{B}/out/{m}_{s}.jsonl"): continue
        pred = load(f"{B}/out/{m}_{s}.jsonl")
        tt = tp = np = ng = full = miss = gc = gch = pc = pch = 0
        for i, g in gold.items():
            G = {tuple(x) for x in g["gold"]}; P = {tuple(x) for x in pred[i]["pred"]}
            tt += len({strip(g["text"], a, b) for a, b in G} & P); tp += len(G & P); np += len(P); ng += len(G)
            pm = set(c for a, b in P for c in range(a, b))
            gm = set(c for a, b in G for c in range(a, b))
            for a, b in G:
                hit = sum(c in pm for c in range(a, b))
                full += hit == b - a; miss += hit == 0
            text = g["text"]
            gc += sum(not text[c].isspace() for c in gm); gch += sum(not text[c].isspace() for c in gm & pm)
            pc += sum(not text[c].isspace() for c in pm); pch += sum(not text[c].isspace() for c in pm & gm)
        p = tp / np if np else 0; r = tp / ng; f = 2 * p * r / (p + r) if p + r else 0
        pct = lambda x: f"{100 * x:.1f}"
        print(f"| {s} ({ng}) | {m} | {pct(p)} | {pct(r)} | {pct(f)} | {pct(2 * tt / (np + ng))} | {pct(full / ng)} | {miss} | {pct(gch / gc)} | {pct(pch / pc) if pc else '-'} |")
