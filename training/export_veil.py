import json
import sys
from pathlib import Path

import numpy as np
import torch

import train as pm

LEVELS = [-1.5104, -0.4528, 0.4528, 1.5104]
PERSON_BIAS = {"PERSON": -0.7}


def rotation():
    return np.load(pm.VEIL / "assets/rotation.npy").astype(np.float64)


def codes_of(rows, rot):
    width = rows.shape[1]
    length = np.linalg.norm(rows, axis=1, keepdims=True)
    safe = np.where(length == 0, 1.0, length)
    turned = (rows / safe) @ rot.T * np.sqrt(width)
    return np.abs(turned[:, :, None] - np.asarray(LEVELS)).argmin(-1), length[:, 0]


def pack(rows):
    rows = rows.astype(np.float64)
    rot = rotation()
    codes, length = codes_of(rows, rot)
    log = np.log(np.maximum(length, 1e-12))
    low, high = float(log.min()), float(log.max())
    length_code = np.round((log - low) / max(high - low, 1e-9) * 255).astype(np.uint8)
    grouped = codes.astype(np.uint8).reshape(len(rows), -1, 4)
    packed = sum(grouped[:, :, i] << (2 * i) for i in range(4)).astype(np.uint8)
    return packed, length_code, [low, high], rot.astype(np.float32)


def quantize(matrix):
    scale = np.abs(matrix).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    return np.clip(np.round(matrix / scale[:, None]), -127, 127).astype(np.int8), scale.astype(np.float32)


def hash_bytes(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def export(directory, out_path, version):
    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text())
    pm.setup(cfg)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    tensors = {}
    codes, length, emb_range, rot = pack(state["embedding.weight"].numpy())
    tensors |= {"emb_codes": codes, "emb_length": length, "emb_rotation": rot}
    tensors["shape"] = state["shape.weight"].numpy().astype(np.float32)
    if not cfg.get("round"):
        raise SystemExit("the checkpoint was not trained with round=true; run the rounded pass first (see README.md)")
    if not cfg.get("bloom"):
        raise SystemExit('add "bloom": 20 and "bloom_k": 8 to config.json; the model file stores the word lists as Bloom filters')
    for bit, table in enumerate(pm.BLOOM):
        tensors[f"bloom_{bit}"] = np.frombuffer(bytes(table), dtype=np.uint8)
    header_cfg = {"buckets": cfg["buckets"], "hidden": cfg["hidden"], "layers": cfg["layers"], "types": pm.TYPES, "bias": PERSON_BIAS, "lists": len(pm.BLOOM), "bloom": cfg["bloom_k"], "levels": LEVELS, "emb_length_range": emb_range}
    for layer in range(cfg["layers"]):
        for suffix, name in (("", "f"), ("_reverse", "b")):
            for part, key in (("wih", "weight_ih"), ("whh", "weight_hh")):
                q, scale = quantize(state[f"lstm.{key}_l{layer}{suffix}"].numpy())
                tensors[f"l{layer}{name}_{part}_q"], tensors[f"l{layer}{name}_{part}_scale"] = q, scale
            bias = (state[f"lstm.bias_ih_l{layer}{suffix}"] + state[f"lstm.bias_hh_l{layer}{suffix}"]).numpy().astype(np.float32)
            tensors[f"l{layer}{name}_b"] = bias
    q, scale = quantize(state["out.weight"].numpy())
    tensors["out_w_q"], tensors["out_w_scale"] = q, scale
    tensors["out_b"] = state["out.bias"].numpy().astype(np.float32)
    entries, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        pad = (-offset) % 4
        blobs.append(b"\0" * pad + data)
        offset += pad
        entries[name] = {"offset": offset, "shape": list(array.shape)}
        offset += len(data)
    head = json.dumps({"name": "veil", "version": version, "format": 3, "config": header_cfg, "tensors": entries}, separators=(",", ":"), ensure_ascii=False).encode()
    head += b" " * ((-(4 + len(head))) % 8)
    body = len(head).to_bytes(4, "little") + head + b"".join(blobs)
    body += b"\0" * ((-len(body)) % 4)
    Path(out_path).write_bytes(body + hash_bytes(body).to_bytes(4, "little"))
    sizes = {name: int(np.ascontiguousarray(a).nbytes) for name, a in tensors.items()}
    print(out_path, Path(out_path).stat().st_size, "bytes", json.dumps({k: v for k, v in sorted(sizes.items(), key=lambda x: -x[1])[:8]}))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: export_veil.py <checkpoint dir> <out.bin> <version>")
    export(sys.argv[1], sys.argv[2], sys.argv[3])
