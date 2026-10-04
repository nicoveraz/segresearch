"""PyTorch port of the BLT-lite harness model (harness._model) and the small entropy model (prepare.lm_forward),
for the scaling pilot (#27). Runs on CUDA (bf16 autocast) and on Apple MPS (fp32).

Same architecture and parameter names as the MLX code, so MLX weights load directly (scale/check_port.py):
  - local encoder: byte embedding + offset-in-patch embedding
  - patch embedding: cross-attention pooling with the patch mean as query (SEGR_POOL=xattn) or a plain sum
  - global causal transformer, once per patch
  - local decoder: one block attending to the previous WINDOW bytes across patches (SEGR_LOCAL=window) or to
    its own patch only, plus the global state of the last completed patch (fresh if the byte starts a patch)

One difference from the MLX code, for speed only: the MLX global transformer runs over CTX patch slots whatever
the patch rate; here it runs over the real patches only (padded to the batch maximum). Empty slots come after every
real patch and never feed back into them, so outputs are identical; global compute now scales with the patch rate.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9          # the MLX code masks with -1e9 (not -inf); kept so fully masked rows behave the same


def ln(x, g):
    m = x.mean(-1, keepdim=True)
    v = ((x - m) ** 2).mean(-1, keepdim=True)
    return (x - m) * torch.rsqrt(v + 1e-5) * g


def gelu_approx(x):
    """MLX nn.gelu_approx (tanh approximation), the same as F.gelu(approximate="tanh")."""
    return F.gelu(x, approximate="tanh")


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        self.qkv = nn.Parameter(torch.empty(d, 3 * d))
        self.o = nn.Parameter(torch.empty(d, d))
        self.w1 = nn.Parameter(torch.empty(d, 4 * d))
        self.w2 = nn.Parameter(torch.empty(4 * d, d))

    def forward(self, h, mask, nh):
        B, T, D = h.shape
        hd = D // nh
        q, k, v = (ln(h, self.ln1) @ self.qkv).split(D, -1)
        sh = lambda t: t.view(B, T, nh, hd).transpose(1, 2)
        a = (sh(q) @ sh(k).transpose(-1, -2)) / math.sqrt(hd)
        a = a.masked_fill(~mask[:, None], NEG)
        h = h + (a.softmax(-1) @ sh(v)).transpose(1, 2).reshape(B, T, D) @ self.o
        return h + gelu_approx(ln(h, self.ln2) @ self.w1) @ self.w2


class XAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.Parameter(torch.ones(d))
        self.q, self.k, self.v, self.o = (nn.Parameter(torch.empty(d, d)) for _ in range(4))


class BLTLite(nn.Module):
    """forward(x, bd): x (B, T) bytes, bd (B, T+1) patch-start flags for x_0..x_T (bd[:, 0] == 1).
    Returns logits (B, T, 256); position t predicts x_{t+1}."""

    def __init__(self, d=64, glayers=2, ctx=128, pool="xattn", local="window", window=32, nh=4):
        super().__init__()
        assert pool in ("sum", "xattn") and local in ("patch", "window") and glayers >= 2
        self.d, self.ctx, self.pool, self.local, self.window, self.nh = d, ctx, pool, local, window, nh
        self.emb = nn.Parameter(torch.empty(256, d))
        self.off = nn.Parameter(torch.empty(ctx, d))
        self.ppos = nn.Parameter(torch.empty(ctx, d))
        self.g = nn.ModuleList(Block(d) for _ in range(glayers))
        self.loc = Block(d)
        self.gproj = nn.Parameter(torch.empty(d, d))
        self.lnf = nn.Parameter(torch.ones(d))
        if pool == "xattn":
            self.xattn = XAttn(d)
        self.reset_parameters()

    def reset_parameters(self, std=0.02):
        """Same distribution as harness._init: N(0, 0.02) for matrices and embeddings, ones for norm gains."""
        for name, p in self.named_parameters():
            if name.split(".")[-1].startswith("ln"):
                nn.init.ones_(p)
            else:
                nn.init.normal_(p, std=std)

    def _pool(self, e, pid, P):
        """Patch embeddings (B, P, D) from byte states e (B, T, D) and patch ids pid (B, T)."""
        B, T, D = e.shape
        oh = F.one_hot(pid, P).to(e.dtype)                                  # (B, T, P)
        if self.pool == "sum":
            return oh.transpose(1, 2) @ e
        X = self.xattn
        mean = (oh.transpose(1, 2) @ e) / oh.sum(1)[:, :, None].clamp(min=1.0)
        hd = D // 4
        en = ln(e, X.ln)
        q = (ln(mean, X.ln) @ X.q).view(B, P, 4, hd).transpose(1, 2)
        k = (en @ X.k).view(B, T, 4, hd).transpose(1, 2)
        v = (en @ X.v).view(B, T, 4, hd).transpose(1, 2)
        mine = (pid[:, None, :] == torch.arange(P, device=e.device)[None, :, None])[:, None]   # (B, 1, P, T)
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~mine, NEG)
        out = (a.softmax(-1) @ v).transpose(1, 2).reshape(B, P, D)
        return mean + out @ X.o

    def forward(self, x, bd):
        B, T = x.shape
        ar = torch.arange(T, device=x.device)
        bdT = bd[:, :T] == 1
        pid = bdT.long().cumsum(1) - 1
        start = torch.where(bdT, ar[None], torch.zeros_like(ar)[None])
        off = ar[None] - start.cummax(1).values
        e = self.emb[x] + self.off[off]
        P = int(pid.max()) + 1                                              # real patches only (see module doc)
        G = self._pool(e, pid, P) + self.ppos[:P][None]
        pa = torch.arange(P, device=x.device)
        causal_p = (pa[:, None] >= pa[None])[None].expand(B, P, P)
        for L in self.g:
            G = L(G, causal_p, self.nh)
        gidx = torch.where(bd[:, 1:T + 1] == 1, pid, pid - 1)              # fresh context iff x_{t+1} starts a patch
        ctx = torch.where((gidx >= 0)[..., None], torch.gather(G, 1, gidx.clamp(min=0)[..., None].expand(B, T, self.d)),
                          torch.zeros((), dtype=G.dtype, device=G.device))
        h = e + ctx @ self.gproj
        causal = ar[:, None] >= ar[None]
        if self.local == "window":
            local = (causal & ((ar[:, None] - ar[None]) < self.window))[None].expand(B, T, T)
        else:
            local = (pid[:, :, None] == pid[:, None, :]) & causal[None]
        h = self.loc(h, local, self.nh)
        return ln(h, self.lnf) @ self.emb.T


class EntropyLM(nn.Module):
    """The small causal byte LM that scores next-byte entropy (prepare.init_lm / lm_forward)."""

    def __init__(self, d=32, layers=1, heads=2, ctx=128):
        super().__init__()
        self.heads = heads
        self.emb = nn.Parameter(torch.empty(256, d))
        self.pos = nn.Parameter(torch.empty(ctx, d))
        self.layers = nn.ModuleList(Block(d) for _ in range(layers))
        self.lnf = nn.Parameter(torch.ones(d))
        for name, p in self.named_parameters():
            if name.split(".")[-1].startswith("ln"):
                nn.init.ones_(p)
            else:
                nn.init.normal_(p, std=0.02)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb[x] + self.pos[:T]
        ar = torch.arange(T, device=x.device)
        mask = (ar[:, None] >= ar[None])[None].expand(x.shape[0], T, T)
        for L in self.layers:
            h = L(h, mask, self.heads)
        return ln(h, self.lnf) @ self.emb.T


def n_params(model):
    return sum(p.numel() for p in model.parameters())
