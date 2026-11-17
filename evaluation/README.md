# Evaluation

`PERSON` detection on five public test sets. None of them is in the Veil training data. The model that labelled
Veil's Wikipedia rows was trained on the CoNLL-2003 English and CoNLL-2002 Spanish and Dutch training splits; their
test splits were not used.

| Set | Text | Licence | Gold |
|---|---|---|---|
| CoNLL-2003 English test | Reuters news | research use | `PER` |
| CoNLL-2002 Spanish and Dutch test | EFE and De Morgen news | research use | `PER` |
| WNUT-17 test | tweets and other social media | CC BY 4.0 | `person` |
| TAB (Text Anonymization Benchmark) test | European Court of Human Rights judgments | MIT | `PERSON`, first annotator |

Each CoNLL and WNUT sentence is one call, with its tokens joined by spaces. TAB documents (up to 12,760 characters)
go through `detectChunked`. All models run with default settings.

Models: Veil 0.3.1 (this repository); spaCy 3.8 `en_core_web_sm`, `es_core_news_sm`, `nl_core_news_sm` 3.8.0;
`@newmean-ai/pii` 1.6.0 with model 1.1.0, sizes nano, small and large, counting `PERSON` and `PERSON_L2`.

## Results

| Set | Model | Exact P | Exact R | Exact F1 | F1, titles ignored | Names fully masked | Names missed (0 chars) | Name chars masked | Masked chars that are names |
|---|---|---|---|---|---|---|---|---|---|
| CoNLL-2003 EN (1617) | Veil 0.3.1 | 85.4 | 63.9 | 73.1 | 73.2 | 66.8 | 526 | 72.3 | 90.0 |
| CoNLL-2003 EN (1617) | spaCy sm | 73.5 | 61.3 | 66.9 | 66.9 | 65.0 | 543 | 71.6 | 79.7 |
| CoNLL-2003 EN (1617) | newmean-nano | 35.7 | 54.8 | 43.2 | 43.3 | 60.9 | 563 | 68.7 | 47.1 |
| CoNLL-2003 EN (1617) | newmean-small | 37.7 | 56.1 | 45.1 | 45.2 | 59.5 | 596 | 66.5 | 48.0 |
| CoNLL-2003 EN (1617) | newmean-large | 42.0 | 63.8 | 50.7 | 50.7 | 67.1 | 487 | 72.6 | 51.9 |
| WNUT-17 (429) | Veil 0.3.1 | 75.0 | 39.2 | 51.5 | 52.1 | 40.6 | 250 | 36.5 | 83.7 |
| WNUT-17 (429) | spaCy sm | 26.7 | 27.3 | 27.0 | 27.7 | 36.6 | 259 | 36.7 | 29.0 |
| WNUT-17 (429) | newmean-nano | 18.3 | 28.0 | 22.1 | 22.1 | 33.1 | 282 | 32.5 | 27.1 |
| WNUT-17 (429) | newmean-small | 28.3 | 34.3 | 31.0 | 31.0 | 38.5 | 258 | 37.4 | 37.1 |
| WNUT-17 (429) | newmean-large | 34.2 | 35.9 | 35.0 | 35.3 | 40.3 | 246 | 38.8 | 43.3 |
| CoNLL-2002 ES (735) | Veil 0.3.1 | 79.8 | 70.7 | 75.0 | 75.0 | 71.2 | 202 | 79.8 | 88.5 |
| CoNLL-2002 ES (735) | spaCy sm | 59.9 | 79.2 | 68.2 | 68.2 | 81.5 | 130 | 84.9 | 69.1 |
| CoNLL-2002 ES (735) | newmean-nano | 43.3 | 62.4 | 51.2 | 51.2 | 69.5 | 203 | 77.3 | 60.6 |
| CoNLL-2002 ES (735) | newmean-small | 50.7 | 65.2 | 57.1 | 57.1 | 68.8 | 211 | 76.6 | 66.0 |
| CoNLL-2002 ES (735) | newmean-large | 57.6 | 73.6 | 64.6 | 64.6 | 77.7 | 139 | 83.7 | 72.6 |
| CoNLL-2002 NL (1098) | Veil 0.3.1 | 85.5 | 60.0 | 70.5 | 70.5 | 61.7 | 415 | 71.6 | 89.7 |
| CoNLL-2002 NL (1098) | spaCy sm | 65.5 | 59.6 | 62.4 | 62.2 | 61.9 | 393 | 70.5 | 73.2 |
| CoNLL-2002 NL (1098) | newmean-nano | 44.4 | 67.6 | 53.6 | 53.5 | 74.0 | 258 | 80.0 | 58.7 |
| CoNLL-2002 NL (1098) | newmean-small | 39.6 | 68.7 | 50.2 | 50.2 | 72.6 | 260 | 78.4 | 55.8 |
| CoNLL-2002 NL (1098) | newmean-large | 54.5 | 74.6 | 63.0 | 63.0 | 77.0 | 227 | 83.5 | 67.5 |
| TAB (1018) | Veil 0.3.1 | 10.1 | 10.0 | 10.1 | 88.1 | 11.4 | 93 | 78.8 | 92.9 |
| TAB (1018) | spaCy sm | 29.2 | 30.2 | 29.7 | 52.1 | 31.3 | 199 | 73.2 | 79.8 |
| TAB (1018) | newmean-nano | 7.5 | 13.1 | 9.5 | 42.9 | 19.0 | 93 | 79.0 | 56.0 |
| TAB (1018) | newmean-small | 7.3 | 9.6 | 8.3 | 58.3 | 12.9 | 69 | 77.4 | 75.3 |
| TAB (1018) | newmean-large | 7.6 | 9.1 | 8.3 | 71.0 | 12.7 | 64 | 79.0 | 85.4 |

- Exact: the predicted span equals the gold span. "F1, titles ignored": a gold span that starts with a title
  (Mr, Mrs, Dr, Lord, ...) also counts when the prediction leaves the title out.
- Names missed: gold names with no masked character. Character columns ignore whitespace.
- WNUT-17 gold `person` includes user handles; newmean labels these `USERNAME`, which is not counted here.

## Run

```sh
pnpm install                      # in the repository root
cd evaluation
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
npm install                       # @newmean-ai/pii, for the comparison only
./run.sh                          # downloads the sets, runs every model, prints the table
```

The test sets are downloaded at run time and are not part of this repository.
