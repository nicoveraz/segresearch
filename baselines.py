"""FIXED reference implementations (read-only). Same API as boundary.py.

Pilot results (single seed, simpler corpus with no long-range queries, 25% budget unless noted):
  fixed stride             answers acc 0.78 / 0.40 (easy / hard)
  raw entropy (BLT)        0.95 / 0.89   <- works mostly because it happens to hit record starts
  excess vs reference      0.85 / 0.56   <- best at *detecting* learnable bytes, worst at using them
  record starts only (9%)  1.00 / 1.00   <- alignment beat allocation at a third of the compute
"""
import numpy as np


def _final(sig):
    return sig.patcher_entropy[max(sig.patcher_entropy)]


class Stride:
    """Patch every 4 bytes (exactly the budget)."""
    @staticmethod
    def score(sig, state):
        return np.arange(len(sig.bytes)) % 4 == 0


class BLTEntropy:
    """Byte Latent Transformer style: patch where the small model's next-byte entropy is high."""
    @staticmethod
    def score(sig, state):
        return _final(sig)


class ExcessVsReference:
    """Rho-1 style reducible uncertainty: small-model entropy minus strong-model entropy."""
    @staticmethod
    def score(sig, state):
        return _final(sig) - sig.reference_entropy


class TrajectoryGated:
    """Entropy of the final small model, kept only where it fell between an early and the final checkpoint."""
    @staticmethod
    def score(sig, state):
        steps = sorted(sig.patcher_entropy)
        drop = sig.patcher_entropy[steps[0]] - _final(sig)
        return np.where(drop > 0.3, _final(sig), 0.0)


ALL = {"stride": Stride, "blt_entropy": BLTEntropy, "excess_vs_reference": ExcessVsReference,
       "trajectory_gated": TrajectoryGated}
