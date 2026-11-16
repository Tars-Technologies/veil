# Training

The trainer, the export and the trained state of model 0.3.1.

| Path | Content |
|---|---|
| `train.py` | Trainer. Its feature code must match `../src/engine.ts` exactly. |
| `export_veil.py` | Writes a Veil model file (format 3) from a checkpoint. |
| `checkpoint/` | The trained state of model 0.3.1 and its settings. |
| `assets/` | The word lists (names, places, common words) and the fixed rotation of the 2-bit embedding table. |

The training set is not published. The [main README](../README.md#training-data) lists its public sources.

## Rebuild the published model file

```sh
uv sync            # Python 3.11 or 3.12; pyproject.toml takes torch from the CUDA 12.8 wheel index
uv run python export_veil.py checkpoint /tmp/veil.bin 0.3.1
sha256sum /tmp/veil.bin ../veil-0.3.1.bin     # the two hashes are equal
```

## Train a model

A training row is one JSON object per line, with UTF-16 offsets:

```json
{"text": "Hi, I'm Anna Schmidt, call +49 151 23456789", "names": [[8, 20]], "spans": [[27, 43, "PHONE"]]}
```

```sh
uv run python train.py data/train.jsonl models/next --epochs 4
# One more pass with the embedding table rounded as the model file stores it.
uv run python train.py data/train.jsonl models/next-rounded --epochs 1 --lr 1e-3 --set round=true --init models/next
# In models/next-rounded/config.json add "bloom": 20 and "bloom_k": 8, then:
uv run python export_veil.py models/next-rounded ../veil-0.4.0.bin 0.4.0
```

About 35 minutes on one 4 GB GPU (RTX 3050). Results change a little with each seed: compare three seeds. Then point
`scripts/embed.mjs`, the tests and the README at the new file, run `pnpm embed`, record the test spans again and
read the difference.

The feature cache (`<train file>.enc-<buckets>`) is keyed by the file name and the bucket count only. Delete it after
a change to the training file or to the features.
