#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SETS="conll03_test wnut17_test conll02_es_test conll02_nl_test tab_test"
scripts/download.sh
python scripts/prep.py .
mkdir -p out
for s in $SETS; do
  ../node_modules/.bin/tsx scripts/run_veil.ts "data/$s.jsonl" "out/veil_$s.jsonl"
done
model() { case $1 in conll02_es_test) echo es_core_news_sm ;; conll02_nl_test) echo nl_core_news_sm ;; *) echo en_core_web_sm ;; esac; }
for s in $SETS; do python scripts/run_spacy.py "$(model "$s")" "data/$s.jsonl" "out/spacy_$s.jsonl"; done
for size in nano small large; do
  for s in $SETS; do node scripts/run_newmean.mjs "$size" "data/$s.jsonl" "out/newmean-${size}_$s.jsonl"; done
done
python scripts/score.py . | tee out/table.md
