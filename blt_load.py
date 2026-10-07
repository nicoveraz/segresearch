"""Load BLT-1B with its 512-byte sliding windows restored.

The Hugging Face conversion we use (itazap/blt-1b-hf) drops BLT's 512-byte sliding window in the entropy patcher
and in the local encoder and decoder (huggingface/transformers issue #49185). The fix proposed upstream (PR #49188,
not merged as of 2026-10-07) is in patches/transformers_pr49188_blt_window.diff; apply it to the installed
transformers (5.18.0) first:

    cd <site-packages> && patch -p2 < .../patches/transformers_pr49188_blt_window.diff

The checkpoint's config sets the windows to None, so they are also set explicitly here. SEGR_BLT_WINDOW=0 loads the
conversion as released (unbounded attention), which is how the results before 2026-10-07 were produced.

Results produced with the window carry window=512 in the registry and a _w512 suffix on per-target files.
"""
import inspect
import os

import torch

WINDOW = int(os.environ.get("SEGR_BLT_WINDOW", "512"))
SUFFIX = f"_w{WINDOW}" if WINDOW else ""


def load(model_id, dev, dtype=torch.bfloat16):
    from transformers import BltForCausalLM
    model = BltForCausalLM.from_pretrained(model_id, dtype=dtype)
    import transformers.models.blt.modeling_blt as mb
    patched = "create_sliding_window_causal_mask" in inspect.getsource(mb)
    if WINDOW and not patched:
        raise SystemExit("transformers is not patched: apply patches/transformers_pr49188_blt_window.diff (see blt_load.py)")
    # set both explicitly: the patched library defaults them to 512, so WINDOW=0 must switch them off
    w = WINDOW or None
    model.config.local_attention_window_len = w
    model.model.config.local_attention_window_len = w
    model.model.patcher.config.sliding_window = w
    return model.to(dev)
