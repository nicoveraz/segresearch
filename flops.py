"""Forward compute per byte, by patch rate: what a tighter patch budget actually saves.

Counts matrix-multiply FLOPs (2 per weight per use); attention-score FLOPs are left out (small at these context
lengths). Three parts:
    per byte    local encoder/decoder layers, pooling projections on bytes, output head
    per patch   global transformer layers, pooling/cross-attention projections on patches
    patcher     the small entropy model that decides boundaries, run on every byte (a dependence-table rule is a
                lookup and needs none)

    uv run flops.py
"""
import numpy as np


def harness(D=128, GL=4, V=256, W=32):
    """BLT-lite in harness.py with SEGR_POOL=xattn SEGR_LOCAL=window; patcher d=32, 1 layer (prepare.PATCHER)."""
    block = 2 * 12 * D * D                              # qkv 3D^2 + o D^2 + MLP 8D^2
    per_byte = block + 2 * 2 * D * D + 2 * D * D + 2 * D * V + 4 * D * W   # local block, xattn k/v, gproj, head, window attn
    per_patch = GL * block + 2 * 2 * D * D              # global layers, xattn q/o
    d = 32
    patcher = 2 * 12 * d * d + 2 * d * V
    return per_byte, per_patch, patcher


def blt1b():
    """Meta's BLT-1B (itazap/blt-1b-hf config): local D=1024 (1 encoder + 9 decoder layers, SwiGLU 2816), global
    D=2048 (25 layers, SwiGLU 5632), decoder cross-attention in every layer, entropy model D=768, 14 layers."""
    loc = 2 * (4 * 1024 ** 2 + 3 * 1024 * 2816)
    glo = 2 * (4 * 2048 ** 2 + 3 * 2048 * 5632)
    per_byte = 10 * loc + 9 * 2 * 2 * 1024 ** 2 + 2 * 2 * 1024 ** 2 + 2 * 1024 * 260   # layers, dec xattn q/o, enc xattn k/v, head
    per_patch = 25 * glo + 9 * 2 * 2 * 2048 * 1024 + 2 * 2 * 2048 * 1024              # global, dec xattn k/v, enc xattn q/o
    patcher = 14 * 2 * (4 * 768 ** 2 + 3 * 768 * 2048) + 2 * 768 * 260
    return per_byte, per_patch, patcher


def table(name, parts, rates, ref):
    per_byte, per_patch, patcher = parts
    print(f"\n{name}: per byte {per_byte / 1e6:.2f} MFLOPs, per patch {per_patch / 1e6:.2f} MFLOPs, "
          f"entropy model {patcher / 1e6:.2f} MFLOPs per byte")
    base = ref * per_patch + per_byte + patcher
    print(f"{'patch rate':>10} | {'with entropy model':>22} | {'with a lookup rule':>22}")
    for r in rates:
        a = r * per_patch + per_byte + patcher; b = r * per_patch + per_byte
        print(f"{r:10.0%} | {a / 1e6:8.2f} MFLOPs ({a / base:4.0%}) | {b / 1e6:8.2f} MFLOPs ({b / base:4.0%})")


if __name__ == "__main__":
    table("harness, D=128, 4 global layers (relative to entropy at 25%)", harness(), (0.25, 0.20, 0.185, 0.15, 0.10), 0.25)
    table("BLT-1B (relative to its default, ~28%)", blt1b(), (0.28, 0.20, 0.15, 0.10), 0.28)
