"""Read Qwen3 weights straight from the safetensors files in the local HF cache (no transformers).

safetensors layout: 8-byte little-endian header length N, N bytes of JSON {name: {dtype, shape, data_offsets}},
then the raw little-endian tensor bytes. We memory-map and return either the raw uint16 bit patterns (for the
bitplanes) or exact float64 values decoded from bfloat16 (1 sign, 8 exponent, 7 mantissa bits).
"""
import glob
import json
import os
import struct

import numpy as np
import torch

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def model_files(model="Qwen3-0.6B"):
    fs = sorted(glob.glob(f"{HUB}/models--Qwen--{model}/snapshots/*/*.safetensors"))
    if not fs:
        raise FileNotFoundError(model)
    return fs


class SafeTensors:
    def __init__(self, model="Qwen3-0.6B"):
        self.model = model
        self.index = {}
        for f in model_files(model):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            mm = np.memmap(f, dtype=np.uint8, mode="r")
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                self.index[k] = (mm, 8 + n, v)

    def names(self):
        return list(self.index)

    def info(self, name):
        return self.index[name][2]

    def raw_u16(self, name):
        mm, base, v = self.index[name]
        assert v["dtype"] in ("BF16", "F16"), v["dtype"]
        a, b = v["data_offsets"]
        return np.frombuffer(mm[base + a: base + b], dtype="<u2").reshape(v["shape"])

    def tensor(self, name, dtype=torch.float64):
        v = self.info(name)
        raw = self.raw_u16(name)
        if v["dtype"] == "BF16":
            t = torch.from_numpy(raw.astype(np.int16, copy=True)).view(torch.bfloat16)
        else:
            t = torch.from_numpy(raw.astype(np.int16, copy=True)).view(torch.float16)
        return t.to(dtype)


def layer_name(i, kind):
    sub = "self_attn" if kind in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    return f"model.layers.{i}.{sub}.{kind}.weight"


KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def decode_bf16_numpy(raw):
    """Independent numpy decode of bf16 bit patterns (used to cross-check the torch view)."""
    raw = raw.astype(np.int64)
    s = (raw >> 15) & 1
    e = (raw >> 7) & 0xFF
    m = raw & 0x7F
    val = np.where(e == 0, (m / 128.0) * 2.0 ** (-126), (1 + m / 128.0) * 2.0 ** (e - 127.0))
    return np.where(s == 1, -val, val)
