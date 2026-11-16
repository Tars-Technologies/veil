import json, sys
import pyarrow.parquet as pq
B = sys.argv[1] if len(sys.argv) > 1 else "."
PER = {"conll03_test": (1, 2), "conll02_es_test": (1, 2), "conll02_nl_test": (1, 2), "wnut17_test": (9, 10)}
for name, (b, i) in PER.items():
    rows = pq.read_table(f"{B}/data/{name}.parquet").to_pylist()
    with open(f"{B}/data/{name}.jsonl", "w") as out:
        n = 0
        for r in rows:
            toks, tags = r["tokens"], r["ner_tags"]
            if not toks or toks[0] == "-DOCSTART-":
                continue
            text, spans, pos, cur = "", [], 0, None
            for k, (t, g) in enumerate(zip(toks, tags)):
                if k:
                    text += " "
                s = len(text); text += t; e = len(text)
                if g == b or (g == i and cur is None):
                    if cur: spans.append(cur)
                    cur = [s, e]
                elif g == i:
                    cur[1] = e
                else:
                    if cur: spans.append(cur)
                    cur = None
            if cur: spans.append(cur)
            out.write(json.dumps({"id": f"{name}:{n}", "text": text, "gold": spans}) + "\n"); n += 1
    print(name, n)
docs = json.load(open(f"{B}/data/tab_test.json"))
with open(f"{B}/data/tab_test.jsonl", "w") as out:
    for d in docs:
        ann = d["annotations"]; first = sorted(ann)[0]
        spans = sorted({(m["start_offset"], m["end_offset"]) for m in ann[first]["entity_mentions"] if m["entity_type"] == "PERSON"})
        out.write(json.dumps({"id": d["doc_id"], "text": d["text"], "gold": [list(s) for s in spans]}) + "\n")
print("tab_test", len(docs))
