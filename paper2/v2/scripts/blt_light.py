"""BLT-1B as blt_load.load gives it (bf16, 512-byte windows), but with its 3.07B-parameter hash-embedding table read
row by row from the checkpoint's float32 through a memory map on the CPU, instead of held as 6.1 GB of bf16 on MPS.

The rows are cast to bf16 exactly as from_pretrained casts the table, so outputs are unchanged; the model's MPS
footprint drops from about 9.3 to 3.2 GB, which keeps the 16 GB Mac out of swap. realblt_copy_gate.py checks the
outputs against the saved quadrant rows (its G0).

    import blt_light; model = blt_light.load(MODEL, dev)
"""
import json
import struct

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache

import _seg  # noqa: F401  (segresearch submodule on sys.path)
import blt_load

NAME = "model.encoder_hash_tok_embedding.weight"


class HashRows(torch.nn.Module):
    """Stands in for model.encoder_hash_tok_embedding: a CPU lookup in the memory-mapped float32 table."""

    def __init__(self, path):
        super().__init__()
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            v = json.loads(f.read(n))[NAME]
        assert v["dtype"] == "F32", v["dtype"]
        self.rows = np.memmap(path, np.float32, "r", offset=8 + n + v["data_offsets"][0], shape=tuple(v["shape"]))

    def forward(self, ids):
        return torch.from_numpy(np.asarray(self.rows[ids.cpu().numpy()])).to(torch.bfloat16)


def load(model_id, dev):
    model = blt_load.load(model_id, "cpu")
    path = try_to_load_from_cache(model_id, "model.safetensors")
    assert isinstance(path, str), f"{model_id} is not in the local Hugging Face cache"
    model.model.encoder_hash_tok_embedding = HashRows(path)      # drops the 6.1 GB bf16 copy before the move
    return model.to(dev)
