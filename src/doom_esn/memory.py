"""Memory back-ends with exact resident-state byte accounting.

The research object of the program (docs/RESEARCH_PROGRAM.md): given a *resident* memory
budget ``B`` bytes, how should the agent store its past? Each back-end here is one answer
and reports its own byte cost, so the benchmark can plot return against bytes rather than
against a vague "model size".

Two byte counts are reported per back-end:

* ``state_nbytes`` -- the per-step resident state, i.e. the thing that *is* the memory of
  the past and that lives in RAM every control step. This is the budgeted quantity ``B``.
* ``fixed_nbytes`` -- learned parameters that do not change per step (a projection, a
  codebook, the reservoir matrix). Reported separately, like the encoder and the readout.

The interesting trade the program measures: a codebook pays a *fixed* table to make the
*state* one or two bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.cluster.vq import kmeans2

from ._types import F32
from .features import Whitener
from .reservoir import ReservoirWeights


class MemoryBackend(Protocol):
    dim: int

    def state_nbytes(self) -> int: ...
    def fixed_nbytes(self) -> int: ...
    def reset(self) -> None: ...
    def step(self, u: F32) -> F32: ...


class WindowMemory:
    """Last ``k`` encoded frames, newest first: the classical delay embedding."""

    def __init__(self, d_in: int, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.d_in, self.k = d_in, k
        self.dim = d_in * k
        self._buf = np.zeros((k, d_in), np.float32)

    def state_nbytes(self) -> int:
        return self.dim * 4

    def fixed_nbytes(self) -> int:
        return 0

    def reset(self) -> None:
        self._buf[:] = 0.0

    def step(self, u: F32) -> F32:
        self._buf[1:] = self._buf[:-1]
        self._buf[0] = u
        return self._buf.ravel().copy()


class ReservoirMemory:
    """Echo-state reservoir: a fixed ``n``-unit fading memory (weights are fixed)."""

    def __init__(self, weights: ReservoirWeights, dt: float) -> None:
        self.weights, self.dt = weights, dt
        self.dim = weights.n
        self._x = weights.zeros()

    def state_nbytes(self) -> int:
        return self.dim * 4

    def fixed_nbytes(self) -> int:
        return self.weights.nbytes()

    def reset(self) -> None:
        self._x = self.weights.zeros()

    def step(self, u: F32) -> F32:
        self._x = self.weights.step(self._x, u, self.dt)
        return self._x.copy()


class QuantizedLatent:
    """A base memory whose whitened state is stored as one integer per axis.

    The projection is fixed; only the ``dim`` integer codes are resident. The feature the
    readout sees is the de-quantised vector, so the readout is unchanged in shape.
    """

    def __init__(
        self, base: MemoryBackend, whitener: Whitener, clip: float = 3.0, bits: int = 8
    ) -> None:
        if bits not in (8, 16):
            raise ValueError("bits must be 8 or 16")
        self.base, self.wh, self.clip, self.bits = base, whitener, clip, bits
        self.dim = whitener.k
        self.dtype = np.uint8 if bits == 8 else np.uint16
        self.levels = float(2**bits - 1)
        self._q = np.zeros(self.dim, self.dtype)

    def state_nbytes(self) -> int:
        return self.dim * np.dtype(self.dtype).itemsize

    def fixed_nbytes(self) -> int:
        w = self.wh
        return self.base.fixed_nbytes() + int(w.mean.size + w.V.size + w.scale.size) * 4

    def reset(self) -> None:
        self.base.reset()
        self._q = np.zeros(self.dim, self.dtype)

    def _dequant(self) -> F32:
        z = self._q.astype(np.float32) / self.levels * 2.0 - 1.0
        return (z * self.clip).astype(np.float32)

    def step(self, u: F32) -> F32:
        z = np.clip(self.wh(self.base.step(u)) / self.clip, -1.0, 1.0)
        self._q = ((z + 1.0) * 0.5 * self.levels).round().astype(self.dtype)
        return self._dequant()


class CodebookMemory:
    """A base memory whose whitened state is replaced by its nearest codeword index.

    A ``k``-row codebook is a fixed table; the resident state is a single index (1 byte
    for ``k <= 256``, else 2). This is the extreme point of the rate-distortion trade.
    """

    def __init__(self, base: MemoryBackend, whitener: Whitener, codebook: F32) -> None:
        self.base, self.wh = base, whitener
        self.cb = np.asarray(codebook, np.float32)
        self.dim = int(self.cb.shape[1])
        self.k = int(self.cb.shape[0])
        self.idx_dtype = np.uint8 if self.k <= 256 else np.uint16
        self._i = 0

    @classmethod
    def fit(
        cls, base: MemoryBackend, whitener: Whitener, U: F32, k: int, seed: int = 0
    ) -> CodebookMemory:
        Z = np.asarray([whitener(base.step(u)) for u in U], np.float32)
        k = min(k, len(Z))
        cb, _ = kmeans2(Z, k, seed=seed, minit="++")
        base.reset()
        return cls(base, whitener, cb.astype(np.float32))

    def state_nbytes(self) -> int:
        return np.dtype(self.idx_dtype).itemsize

    def fixed_nbytes(self) -> int:
        w = self.wh
        return (
            self.base.fixed_nbytes()
            + int(self.cb.size) * 4
            + int(w.mean.size + w.V.size + w.scale.size) * 4
        )

    def reset(self) -> None:
        self.base.reset()
        self._i = 0

    def step(self, u: F32) -> F32:
        z = self.wh(self.base.step(u))
        self._i = int(np.argmin(((self.cb - z) ** 2).sum(axis=1)))
        return self.cb[self._i].copy()


@dataclass(frozen=True)
class ByteReport:
    name: str
    state_bytes: int
    fixed_bytes: int
    dim: int


class RandomTanhFeatures:
    """Wrap a memory in fixed random ``tanh`` features.

    This exists to remove a confound in the return-vs-bytes comparison. A raw window feeds a
    *linear* readout on the last ``k`` frames, while a reservoir feeds it random *nonlinear*
    features, so "window vs reservoir" mixes memory length with nonlinearity. Applying the
    *same* fixed random projection to both isolates the memory mechanism: the only difference
    left is how the past is retained (hard window vs fading reservoir).

    The projection is a fixed table (``fixed_nbytes``); the resident state is unchanged
    (``state_nbytes``), because the features are recomputed each step, not stored.
    """

    def __init__(self, base: MemoryBackend, out_dim: int, seed: int = 0, scale: float = 1.0):
        self.base = base
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((out_dim, base.dim)) / np.sqrt(base.dim) * scale).astype(
            np.float32
        )
        self.b = (rng.uniform(-1.0, 1.0, out_dim) * scale).astype(np.float32)
        self.dim = out_dim

    def state_nbytes(self) -> int:
        return self.base.state_nbytes()

    def fixed_nbytes(self) -> int:
        return self.base.fixed_nbytes() + int(self.W.size + self.b.size) * 4

    def reset(self) -> None:
        self.base.reset()

    def step(self, u: F32) -> F32:
        return np.tanh(self.W @ self.base.step(u) + self.b).astype(np.float32)


class ProjectedMemory:
    """A back-end whose output is projected onto fixed whitened axes before the readout.

    The projection is a *fixed* table, so it adds to ``fixed_nbytes`` but not to the
    resident state: whitening reduces the readout's input dimension (helping a linear
    readout), it does not by itself shrink what is stored. Use QuantizedLatent or
    CodebookMemory to actually shrink the resident state.
    """

    def __init__(self, base: MemoryBackend, whitener: Whitener) -> None:
        self.base, self.wh = base, whitener
        self.dim = whitener.k

    def state_nbytes(self) -> int:
        return self.base.state_nbytes()

    def fixed_nbytes(self) -> int:
        w = self.wh
        return self.base.fixed_nbytes() + int(w.mean.size + w.V.size + w.scale.size) * 4

    def reset(self) -> None:
        self.base.reset()

    def step(self, u: F32) -> F32:
        return self.wh(self.base.step(u))
