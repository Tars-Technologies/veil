# Veil

Veil finds names, addresses, phone numbers and other personal data in text, so you can mask them before the text is
stored, logged or sent to another service. The model is 1.18 MB. It runs in the browser or in Node, with no
dependencies and no network calls.

**[Try the demo](https://tars-technologies.github.io/veil/).** The model runs in your browser, so the text you type
stays on your machine.

```ts
import { createVeil } from "@tars-technologies/veil"
import { loadModel } from "@tars-technologies/veil/model"

const veil = createVeil({ model: loadModel() })

await veil.mask({
  text: "I'm Laura Bianchi, call me on +33 6 12 34 56 78 or write to laura.bianchi@example.com",
  types: ["PERSON", "PHONE", "EMAIL"]
})
// "I'm [NAME], call me on [PHONE] or write to [EMAIL]"
```

## Install

```sh
npm install github:Tars-Technologies/veil
```

The package ships TypeScript source. Use it with a bundler or runtime that compiles TypeScript, such as Vite,
esbuild, tsx or Bun.

## Usage

`detect` returns the spans and `mask` returns the text with each span replaced. Both are async.

```ts
await veil.mask({ text: "my manager Daniel Okafor told me to call" })
// "my manager [NAME] told me to call"

await veil.detect({ text: "call Anna Schmidt" })
// [{ start: 5, end: 17, type: "PERSON" }]

await veil.mask({ text: "Summer", context: "What is your name?", expectName: true })
// "[NAME]"
```

`model` takes the model bytes or a URL to `veil-0.3.1.bin`. `loadModel()` returns the bytes from inside the package,
which suits servers and serverless functions. It adds about 1.6 MB of base64 to a bundle, so in a browser load the
file by URL instead, through `@tars-technologies/veil/browser`. That entry point has the same API and runs the model
in a Web Worker, so the page does not wait for it. With Vite:

```ts
import { createVeil } from "@tars-technologies/veil/browser"
import modelUrl from "@tars-technologies/veil/veil-0.3.1.bin?url"

const veil = createVeil({ model: modelUrl })
```

| Option | Effect |
|---|---|
| `types` | Labels to find. Default `["PERSON"]`. |
| `threshold` | Between 0 and 1, not including 0 or 1. Default 0.5. A lower value masks more names, and more words that are not names. |
| `context` | The text that came before, for example the question this text answers. |
| `expectName` | The text answers a name question. A short answer is masked as a whole. |
| `always` | Names to mask wherever they occur as whole words, for example a name the user gave earlier. The spans have the label `PERSON`. |
| `never` | Terms to leave alone, for example your brand, product or assistant names. |
| `replacement` | One string that replaces every span. By default each label has its own: `[NAME]`, `[ADDRESS]`, `[DATE OF BIRTH]`, `[ID]`, `[FINANCIAL]`, `[PHONE]`, `[EMAIL]`. |

Pass options to `detect` or `mask` per call, or once as `createVeil({ model, defaults: { ... } })`. A call's options
override the defaults.

One call takes up to 20,000 characters and 3,000 tokens (words and punctuation marks), `context` included. For text
with more tokens, use
`detectChunked(veil, text, options)` from `@tars-technologies/veil/chunked`. It splits the text into overlapping calls
and returns offsets into the full text. The 20,000-character limit still applies. All offsets are UTF-16.

Veil throws on errors and never returns partial results. For a privacy feature, the safe response to an error is
usually to not store the text.

### React

```tsx
import { useMask, VeilProvider } from "@tars-technologies/veil/react"

<VeilProvider veil={veil}>…</VeilProvider>

const { data, spans, isPending, error } = useMask({ text: draft, types: ["PERSON", "EMAIL"] })
```

`useMask` masks the text again each time its options change, which suits a live preview. To mask on send, call
`useVeil().mask` in the handler.

## Model

| | |
|---|---|
| Author | Vaibhav Kubre, Tars Technologies |
| Labels | `PERSON`, `ADDRESS`, `DOB`, `ID`, `FINANCIAL`, `PHONE`, `EMAIL` |
| Architecture | Hashed character n-gram embeddings (32,768 buckets × 64, 2-bit codes), word-shape features and three word-list flags, into a 2-layer bidirectional LSTM with 96 units. Weights are int8. |
| Size | 1,177,096 bytes (`veil-0.3.1.bin`) |
| Languages | Latin script. Most training text is English, German, French, Spanish, Dutch, Portuguese or Italian. |
| Input | Windows of 256 tokens with a 64-token overlap. Up to 300 characters of `context`. |
| Licence | Apache-2.0 for code and model. `NOTICE` credits the data sources. |

The model file carries its name, version and format in a header and ends with a checksum. The library reads format 3
and rejects a file whose checksum does not match. Each new model gets a new file name.

## Evaluation

We measured `PERSON` on five public test sets. None of them is in the training data. All models use their default
settings. Exact-match F1:

| Test set | Names | Veil 0.3.1 (1.2 MB) | spaCy sm (15 MB) | newmean nano (2.6 MB) | newmean small (7.1 MB) | newmean large (20.7 MB) |
|---|---|---|---|---|---|---|
| CoNLL-2003 English, news | 1,617 | **73.1** | 66.9 | 43.2 | 45.1 | 50.7 |
| WNUT-17, social media | 429 | **51.5** | 27.0 | 22.1 | 31.0 | 35.0 |
| CoNLL-2002 Spanish, news | 735 | **75.0** | 68.2 | 51.2 | 57.1 | 64.6 |
| CoNLL-2002 Dutch, news | 1,098 | **70.5** | 62.4 | 53.6 | 50.2 | 63.0 |
| TAB, court judgments | 1,018 | **88.1** | 52.1 | 42.9 | 58.3 | 71.0 |

TAB includes titles in its names ("Mrs Karin Schieck") and Veil does not, so the TAB row ignores a leading title.
With strict matching, Veil scores 10.1 and spaCy 29.7.

F1 does not tell the whole story for masking. A missed name is a leak, while an extra mask costs little. This table
counts the names that had no character masked at all (lower is better):

| Test set | Veil 0.3.1 | spaCy sm | newmean large |
|---|---|---|---|
| CoNLL-2003 English | 526 | 543 | **487** |
| WNUT-17 | 250 | 259 | **246** |
| CoNLL-2002 Spanish | 202 | **130** | 139 |
| CoNLL-2002 Dutch | 415 | 393 | **227** |
| TAB | 93 | 199 | **64** |

newmean large misses the fewest names on four of the five sets, but masks many words that are not names. Of the characters Veil masks, 84 to
93 % are part of a name. For spaCy that figure is 29 to 80 %, and for newmean 27 to 85 %. Versions: spaCy 3.8 with
the `*_core_*_sm` 3.8.0 models, and `@newmean-ai/pii` 1.6.0 with model 1.1.0, counting `PERSON` and `PERSON_L2`.
[`evaluation/`](evaluation/README.md) has the full results and the scripts to reproduce them.

## Training data

The training data includes these public datasets:

| Source | Rows | Licence |
|---|---|---|
| Wikipedia articles (DE, FR, EN), labelled by a larger model | 181,689 | CC BY-SA |
| Wikipedia talk pages (FR, NL, ES, PT, DE, IT) | 40,371 | CC BY-SA 4.0 |
| ABCD customer-service dialogues | 74,792 | MIT |
| Taskmaster dialogues | 39,723 | CC BY 4.0 |
| OASST2 assistant conversations | 35,793 | Apache-2.0 |
| MASSIVE utterances | 43,504 | CC BY 4.0 |
| Stack Exchange posts | 8,455 | CC BY-SA 4.0 |
| French public-service texts | 7,206 | Licence Ouverte 2.0 |
| NVIDIA Nemotron-PII | 96,050 | CC BY 4.0 |

The word-list flags come from three lists in `training/assets/gazetteer.json.gz`: given names and surnames
(including Wikidata), places from GeoNames, and common words from the training text. The training set itself is not
published. [`training/`](training/README.md) has the trainer, the checkpoint and the export script, which rebuilds
`veil-0.3.1.bin` byte for byte.

## Limitations

- Veil misses names. In the news test sets, 27 to 38 % of names have no character masked, and social media is
  worse (lower-case handles, lone surnames). Treat it as a way to reduce exposure, not as anonymisation.
- Latin script only. Names in Greek, Cyrillic, Arabic and other scripts are not found.
- The most common false masks are brand, product and place names. Add the ones you know to `never`.
- Titles such as Mr or Dr are not part of the span.
- Only `PERSON` has been measured on public data. The other six labels have no published evaluation yet.

## Development

```sh
pnpm install
pnpm test:ci
pnpm type-check
pnpm demo           # demo page on http://localhost:5173
pnpm embed          # after a model change, regenerates src/model.ts from the .bin file
```

## Citation

```bibtex
@software{veil_2026,
  title        = {Veil: a 1 MB on-device model for masking personal data},
  author       = {Kubre, Vaibhav},
  organization = {Tars Technologies},
  year         = {2026},
  version      = {0.3.1},
  url          = {https://github.com/Tars-Technologies/veil}
}
```
