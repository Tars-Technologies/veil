#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
H=https://huggingface.co/datasets
get() { curl -sfL -o "data/$1" "$2"; echo "data/$1"; }
get conll03_test.parquet "$H/eriktks/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003/test/0000.parquet"
get wnut17_test.parquet "$H/leondz/wnut_17/resolve/refs%2Fconvert%2Fparquet/wnut_17/test/0000.parquet"
get conll02_es_test.parquet "$H/eriktks/conll2002/resolve/refs%2Fconvert%2Fparquet/es/test/0000.parquet"
get conll02_nl_test.parquet "$H/eriktks/conll2002/resolve/refs%2Fconvert%2Fparquet/nl/test/0000.parquet"
get tab_test.json "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/master/echr_test.json"
