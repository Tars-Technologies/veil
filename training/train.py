import gzip
import json
import multiprocessing
import queue
import threading
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import regex
import torch
from torch import nn

TOKEN = regex.compile(r"[\p{L}\p{M}\p{N}]+(?:[\-‐‑][\p{L}\p{M}\p{N}]+)*|[^\s\p{L}\p{M}\p{N}]")
WORDISH = regex.compile(r"[\p{L}\p{N}]")
DIGIT = regex.compile(r"\p{Nd}")
LETTER = regex.compile(r"\p{L}")
TYPES = ["PERSON", "OTHER", "ADDRESS", "DOB", "ID", "FINANCIAL", "PHONE", "EMAIL"]
TYPE_ID = {t: i for i, t in enumerate(TYPES)}
NTAGS = 1 + 4 * len(TYPES)
MAX_TOKENS = 256
STRIDE = 192
VEIL = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CPU") != "1" else "cpu")
CFG = {"buckets": 32768, "dim": 64, "hidden": 96, "layers": 2}


def fnv1a(units: bytes) -> int:
    value = 0x811C9DC5
    for i in range(0, len(units), 2):
        value ^= units[i] | (units[i + 1] << 8)
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def h(text: str) -> int:
    return fnv1a(text.encode("utf-16-le", "surrogatepass"))


def tokens(text: str):
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + (2 if ord(character) > 0xFFFF else 1))
    out, last_end = [], -1
    for match in TOKEN.finditer(text):
        start, end = match.start(), match.end()
        is_word = bool(WORDISH.search(match.group(0)))
        if is_word and end - start == 1 and text[end : end + 1] == ".":
            end += 1
        if not is_word and last_end > start:
            continue
        out.append((offsets[start], offsets[end], text[start:end]))
        last_end = end
    return out


def is_upper(c):
    return c.lower() != c


def shape_id(word, index, previous, gap_has_newline):
    letters = LETTER.findall(word)
    if not letters:
        case = 0 if DIGIT.search(word) else 1
    else:
        upper = sum(is_upper(c) for c in letters)
        if upper == 0:
            case = 2
        elif upper == len(letters):
            case = 3 if len(letters) > 1 else 4
        elif upper == 1 and is_upper(letters[0]):
            case = 5
        else:
            case = 6
    position = 0 if index == 0 or gap_has_newline else (1 if previous in {".", "!", "?", ":", "-", "—", ","} else 2)
    return case * 3 + position


def word_shape(word):
    out = []
    for c in word:
        if DIGIT.match(c):
            out.append("0")
        elif LETTER.match(c):
            out.append("A" if is_upper(c) else "a")
        else:
            out.append(c)
    return "".join(out)


def hashes(word, cfg):
    buckets = cfg["buckets"]
    has_digit = bool(DIGIT.search(word))
    lower = word.lower()
    if has_digit:
        lower = DIGIT.sub("0", lower)
    out = [h("w:" + lower)]
    units = ("<" + lower + ">").encode("utf-16-le", "surrogatepass")
    length = len(units) // 2
    for n in range(2, 6):
        for i in range(0, length - n + 1):
            out.append(fnv1a(units[2 * i : 2 * (i + n)]))
    if has_digit:
        shape = word_shape(word)
        out.append(h("s:" + shape[:16]))
        out.append(h("c:" + regex.sub(r"(.)\1+", r"\1", shape)[:16]))
        letters = LETTER.findall(word)
        kind = "d" if not letters else "u" if all(is_upper(c) for c in letters) else "x"
        out.append(h("k:" + kind + str(min(len(word), 20))))
        if regex.fullmatch(r"[0-9]+", word):
            out.append(h("l:" + str(min(len(word), 20))))
            if len(word) <= 2:
                out.append(h("v:" + str(int(word))))
            elif len(word) == 4:
                out.append(h("y:" + word[:3]))
    return [x % buckets for x in out]


GAZ = {}
BLOOM = []
BLOOM_K = []
FLAGS = {}
SHAPE_IDS = 21 * 8


def bloom_probes(word, bits, count=4):
    value = 0x811C9DC5
    for byte in word.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    step = ((value * 0x9E3779B1) & 0xFFFFFFFF) | 1
    return [((value + i * step) & 0xFFFFFFFF) % bits for i in range(count)]


def flags_of(word):
    if not BLOOM:
        return GAZ.get(word, 0)
    found = FLAGS.get(word)
    if found is None:
        found = 0
        for bit, table in enumerate(BLOOM):
            if all(table[at >> 3] >> (at & 7) & 1 for at in bloom_probes(word, 8 * len(table), len(BLOOM_K))):
                found |= 1 << bit
        if len(FLAGS) < 3000000:
            FLAGS[word] = found
    return found


LEVELS = [-1.5104, -0.4528, 0.4528, 1.5104]
_ROT = {}


def round_rows(weight):
    import export_veil

    width = weight.shape[1]
    key = (width, str(weight.device), weight.dtype)
    if key not in _ROT:
        _ROT[key] = torch.tensor(export_veil.rotation(), dtype=weight.dtype, device=weight.device)
    rot = _ROT[key]
    length = weight.norm(dim=1, keepdim=True)
    safe = torch.where(length == 0, torch.ones_like(length), length)
    turned = (weight / safe) @ rot.T * (width**0.5)
    levels = torch.tensor(LEVELS, dtype=weight.dtype, device=weight.device)
    codes = (turned.unsqueeze(-1) - levels).abs().argmin(-1)
    return (levels[codes] / (width**0.5)) @ rot * length


def setup(cfg):
    GAZ.clear()
    BLOOM.clear()
    BLOOM_K[:] = [0] * cfg.get("bloom_k", 4)
    FLAGS.clear()
    lists = json.load(gzip.open(VEIL / "assets/gazetteer.json.gz", "rt", encoding="utf-8"))
    for bit, key in enumerate(("names", "places", "common")):
        for word in lists[key]:
            GAZ[word] = GAZ.get(word, 0) | (1 << bit)
    if cfg.get("bloom"):
        for bit in range(3):
            words = [w for w, flags in GAZ.items() if flags >> bit & 1]
            table = bytearray((len(words) * cfg["bloom"] + 7) // 8)
            for word in words:
                for at in bloom_probes(word, 8 * len(table), len(BLOOM_K)):
                    table[at >> 3] |= 1 << (at & 7)
            BLOOM.append(table)


def featurize(text, cfg, toks=None):
    toks = tokens(text) if toks is None else toks
    units = text.encode("utf-16-le", "surrogatepass")
    feats, shapes = [], []
    previous_end, previous_word = 0, None
    for index, (start, end, word) in enumerate(toks):
        gap = units[2 * previous_end : 2 * start].decode("utf-16-le", "surrogatepass")
        lowered = word.lower()
        shapes.append(shape_id(word, index, previous_word, "\n" in gap) + 21 * flags_of(lowered))
        feats.append(hashes(word, cfg))
        previous_end, previous_word = end, word
    return toks, feats, shapes


def mark(tags, toks, start, end, base, only_free=False):
    inside = [i for i, (s, e, _) in enumerate(toks) if s < end and start < e]
    while inside and not WORDISH.search(toks[inside[0]][2]):
        inside.pop(0)
    while inside and not WORDISH.search(toks[inside[-1]][2]):
        inside.pop()
    if not inside or (only_free and any(tags[i] for i in inside)):
        return
    if len(inside) == 1:
        tags[inside[0]] = base + 3
    else:
        tags[inside[0]] = base
        for i in inside[1:-1]:
            tags[i] = base + 1
        tags[inside[-1]] = base + 2


def row_tags(toks, row):
    tags = [0] * len(toks)
    mask = [1] * len(toks)
    spans = [(s, e, "PERSON") for s, e in row.get("names", [])] + [tuple(x) for x in row.get("spans", [])]
    for s, e, label in sorted(spans, key=lambda x: (x[2] != "PERSON", x[0])):
        mark(tags, toks, s, e, 1 + 4 * TYPE_ID[label], only_free=label != "PERSON")
    for s, e, *_ in list(row.get("optional", [])) + list(row.get("entities", [])):
        mark(tags, toks, s, e, 1 + 4 * TYPE_ID["OTHER"], only_free=True)
    for s, e in row.get("uncertain", []):
        for i, (ts, te, _) in enumerate(toks):
            if ts < e and s < te and tags[i] == 0:
                mask[i] = 0
    return tags, mask


def _encode_chunk(job):
    rows, cfg = job
    out = [[], [], [], [], [], []]
    for row in rows:
        all_toks = tokens(row["text"])
        if not all_toks:
            continue
        tags, mask = row_tags(all_toks, row)
        _, feats, shapes = featurize(row["text"], cfg, all_toks)
        for start in range(0, len(all_toks), STRIDE):
            sl = slice(start, start + MAX_TOKENS)
            f = feats[sl]
            out[0].append(np.fromiter((i for ids in f for i in ids), dtype=np.int32))
            out[1].append(np.fromiter((len(ids) for ids in f), dtype=np.int32, count=len(f)))
            out[2].append(np.asarray(shapes[sl], dtype=np.int32))
            out[3].append(np.asarray(tags[sl], dtype=np.int8))
            out[4].append(np.asarray(mask[sl], dtype=np.int8))
            out[5].append(len(f))
            if start + MAX_TOKENS >= len(all_toks):
                break
    if not out[5]:
        return None
    return [np.concatenate(c) for c in out[:5]] + [np.asarray(out[5], dtype=np.int32)]


class Encoded:
    def __init__(self, arrays):
        self.flat, self.counts, self.shapes, self.tags, self.mask, self.lengths = arrays
        self.token_start = np.concatenate([[0], np.cumsum(self.lengths, dtype=np.int64)])
        self.id_start = np.concatenate([[0], np.cumsum(self.counts, dtype=np.int64)])

    def __len__(self):
        return len(self.lengths)

    def batch(self, rows):
        spans = [(self.token_start[r], self.token_start[r + 1]) for r in rows]
        lengths = [int(e - s) for s, e in spans]
        width = max(lengths)
        counts = np.concatenate([self.counts[s:e] for s, e in spans])

        def pad(column):
            out = np.zeros((len(rows), width), dtype=np.int64)
            for i, (s, e) in enumerate(spans):
                out[i, : e - s] = column[s:e]
            return torch.from_numpy(out)

        flat = np.concatenate([self.flat[self.id_start[s] : self.id_start[e]] for s, e in spans]).astype(np.int64)
        return torch.from_numpy(flat), torch.from_numpy(np.cumsum(counts, dtype=np.int64) - counts), pad(self.shapes), lengths, pad(self.tags), pad(self.mask)


def encode_file(path, cfg, cache):
    cache = Path(cache)
    names = ("flat", "counts", "shapes", "tags", "mask", "lengths")
    if not cache.exists():
        def jobs():
            rows = []
            with open(path, encoding="utf-8") as source:
                for line in source:
                    rows.append(json.loads(line))
                    if len(rows) == 2000:
                        yield rows, cfg
                        rows = []
            if rows:
                yield rows, cfg

        tmp = Path(str(cache) + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        dtypes = (np.int32, np.int32, np.int32, np.int8, np.int8, np.int32)
        sizes = [0] * len(names)
        outputs = [open(tmp / f"{name}.raw", "wb") for name in names]
        with multiprocessing.get_context("fork").Pool(int(os.environ.get("ENCODE_WORKERS", "6"))) as pool:
            for chunk in pool.imap(_encode_chunk, jobs(), chunksize=1):
                if chunk is not None:
                    for i, column in enumerate(chunk):
                        column.astype(dtypes[i]).tofile(outputs[i])
                        sizes[i] += len(column)
        for output in outputs:
            output.close()
        for name, dtype, size in zip(names, dtypes, sizes):
            with open(tmp / f"{name}.npy", "wb") as output, open(tmp / f"{name}.raw", "rb") as source:
                np.lib.format.write_array_header_1_0(output, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": (size,)})
                shutil.copyfileobj(source, output, length=1 << 20)
            (tmp / f"{name}.raw").unlink()
        tmp.rename(cache)
    return Encoded([np.load(cache / f"{name}.npy", mmap_mode="r") for name in names])


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.EmbeddingBag(cfg["buckets"], cfg["dim"], mode="mean", sparse=True)
        self.shape = nn.Embedding(SHAPE_IDS, 16)
        self.lstm = nn.LSTM(cfg["dim"] + 16, cfg["hidden"], num_layers=cfg["layers"], bidirectional=True, batch_first=True)
        self.out = nn.Linear(2 * cfg["hidden"], NTAGS)
        self.dropout = nn.Dropout(0.2)
        self.round = bool(cfg.get("round"))

    def forward(self, flat, offsets, shapes, lengths):
        if self.round:
            weight = self.embedding.weight
            vectors = nn.functional.embedding_bag(flat, weight + (round_rows(weight) - weight).detach(), offsets, mode="mean")
        else:
            vectors = self.embedding(flat, offsets)
        batch, cursor = [], 0
        for length in lengths:
            batch.append(vectors[cursor : cursor + length])
            cursor += length
        padded = nn.utils.rnn.pad_sequence(batch, batch_first=True)
        if self.training:
            shapes = torch.where(torch.rand(shapes.shape, device=shapes.device) < 0.25, shapes % 21, shapes)
        x = self.dropout(torch.cat([padded, self.shape(shapes)], dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(lengths), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.out(self.dropout(out))


def train(train_path, out_dir, cfg, epochs=3, lr=3e-3, batch=64, seed=0, init=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    setup(cfg)
    data = encode_file(train_path, cfg, f"{train_path}.enc-{cfg['buckets']}")
    if data.flat.nbytes < 5e9:
        for name in ("flat", "counts", "shapes", "tags", "mask"):
            setattr(data, name, np.array(getattr(data, name)))
    model = Model(cfg).to(DEVICE)
    if init:
        model.load_state_dict(torch.load(Path(init) / "model.pt", map_location=DEVICE, weights_only=True))
    optimizers = [
        (torch.optim.Adam if cfg.get("round") else torch.optim.SparseAdam)(list(model.embedding.parameters()), lr=lr),
        torch.optim.AdamW([p for n, p in model.named_parameters() if not n.startswith("embedding") and p.requires_grad], lr=lr, weight_decay=1e-5),
    ]
    weights = torch.tensor([1.0] + [2.0] * (NTAGS - 1), device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="none")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg))
    order = np.argsort(data.lengths, kind="stable")
    batches = [order[i : i + batch] for i in range(0, len(order), batch)]
    total_steps = epochs * len(batches)
    step = 0
    for epoch in range(epochs):
        model.train()
        random.shuffle(batches)
        total, started = 0.0, time.time()
        ready = queue.Queue(maxsize=16)

        def feed(todo=list(batches)):
            try:
                for items in todo:
                    ready.put(data.batch(np.sort(items)))
            except BaseException as error:
                ready.put(error)
                return
            ready.put(None)

        threading.Thread(target=feed, daemon=True).start()
        while True:
            got = ready.get()
            if got is None:
                break
            if isinstance(got, BaseException):
                raise got
            built = got
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr * (1.0 if step < total_steps * 0.4 else max(0.05, (1 - step / total_steps) / 0.6))
            step += 1
            flat, offsets, shapes, lengths, tags, mask = (t.to(DEVICE) if torch.is_tensor(t) else t for t in built)
            if not mask.any():
                continue
            logits = model(flat, offsets, shapes, lengths)
            loss = (loss_fn(logits.reshape(-1, NTAGS), tags.reshape(-1)) * mask.reshape(-1)).sum() / mask.sum()
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if not n.startswith("embedding")], 5.0)
            for optimizer in optimizers:
                optimizer.step()
            total += loss.item()
        print(json.dumps({"epoch": epoch, "loss": round(total / len(batches), 4), "sec": round(time.time() - started)}), flush=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--init", default="")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    cfg = dict(CFG)
    for item in args.set:
        key, value = item.split("=")
        cfg[key] = json.loads(value)
    train(args.train, args.out, cfg, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed, init=args.init or None)
