import json, sys, time, spacy
model, inp, outp = sys.argv[1:4]
nlp = spacy.load(model, disable=["parser", "lemmatizer", "tagger", "morphologizer", "attribute_ruler"])
nlp.max_length = 2_000_000
rows = [json.loads(l) for l in open(inp)]
t0 = time.perf_counter()
with open(outp, "w") as out:
    for r, doc in zip(rows, nlp.pipe((r["text"] for r in rows), batch_size=64)):
        out.write(json.dumps({"id": r["id"], "pred": [[e.start_char, e.end_char] for e in doc.ents if e.label_ in ("PERSON", "PER")]}) + "\n")
print(inp.split("/")[-1], len(rows), "rows", round((time.perf_counter() - t0) * 1000), "ms")
