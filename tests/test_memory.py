"""Byte accounting and behaviour of the memory back-ends and the solved readout."""

from __future__ import annotations

import numpy as np
import pytest

from doom_esn.features import Whitener
from doom_esn.memory import (
    CodebookMemory,
    ProjectedMemory,
    QuantizedLatent,
    RandomTanhFeatures,
    ReservoirMemory,
    WindowMemory,
)
from doom_esn.readout import LinearQ
from doom_esn.reservoir import ReservoirConfig, ReservoirWeights


@pytest.fixture
def whitener() -> Whitener:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 40)).astype(np.float32)
    return Whitener.fit(X, 12)


def test_window_bytes_and_exact_delay() -> None:
    m = WindowMemory(5, 3)
    assert m.state_nbytes() == 5 * 3 * 4
    assert m.fixed_nbytes() == 0
    m.step(np.arange(5, dtype=np.float32))
    out = m.step(np.full(5, 9, np.float32))
    np.testing.assert_allclose(out[:5], 9.0)
    np.testing.assert_allclose(out[5:10], np.arange(5))


def test_reservoir_bytes() -> None:
    w = ReservoirWeights.build(ReservoirConfig(n_res=64, density=0.1), 8)
    m = ReservoirMemory(w, dt=4.0)
    assert m.state_nbytes() == 64 * 4
    assert m.fixed_nbytes() == w.nbytes() > 0
    m.reset()
    assert np.isfinite(m.step(np.zeros(8, np.float32))).all()


def test_quantized_state_bytes_and_error(whitener: Whitener) -> None:
    w = ReservoirWeights.build(ReservoirConfig(n_res=40, density=0.1), 8)
    m8 = QuantizedLatent(ReservoirMemory(w, 4.0), whitener, clip=3.0, bits=8)
    m16 = QuantizedLatent(ReservoirMemory(w, 4.0), whitener, clip=3.0, bits=16)
    assert m8.state_nbytes() == m8.dim
    assert m16.state_nbytes() == 2 * m16.dim
    assert m8.state_nbytes() < m16.state_nbytes() < whitener.k * 4
    u = np.random.default_rng(1).standard_normal(8).astype(np.float32)
    q8, q16 = m8.step(u), m16.step(u)
    # both stay inside the clip box; 8-bit is at least as coarse as 16-bit
    assert np.abs(q8).max() <= 3.0 + 1e-6
    assert np.abs(q16).max() <= 3.0 + 1e-6
    assert np.abs(q8 - q16).max() <= 3.0 + 1e-6


def test_codebook_stores_one_index(whitener: Whitener) -> None:
    rng = np.random.default_rng(2)
    w = ReservoirWeights.build(ReservoirConfig(n_res=40, density=0.1), 8)
    U = rng.standard_normal((400, 8)).astype(np.float32)
    cb = CodebookMemory.fit(ReservoirMemory(w, 4.0), whitener, U, k=64, seed=0)
    assert cb.state_nbytes() == 1  # k <= 256 -> one byte
    assert cb.fixed_nbytes() > 0
    u = rng.standard_normal(8).astype(np.float32)
    out = cb.step(u)
    assert out.shape == (cb.dim,)
    # the returned vector is exactly one of the codewords
    assert np.min(np.abs(cb.cb - out).sum(axis=1)) < 1e-6


def test_projected_keeps_state_shrinks_readout_input(whitener: Whitener) -> None:
    w = ReservoirWeights.build(ReservoirConfig(n_res=64, density=0.1), 8)
    base = ReservoirMemory(w, dt=4.0)
    proj = ProjectedMemory(base, whitener)
    assert proj.state_nbytes() == base.state_nbytes()  # projection does not shrink the store
    assert proj.dim == whitener.k < base.dim  # it does shrink the readout input
    assert proj.fixed_nbytes() > base.fixed_nbytes()


def test_linear_q_solves_a_tiny_mdp() -> None:
    """Two states; action 1 is worth +1 from state 0. Greedy must pick it."""
    S = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], np.float32)
    A = np.array([0, 0, 1, 1])
    R = np.array([0.0, 0.0, 1.0, 0.0], np.float32)
    S2 = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], np.float32)
    D = np.array([0.0, 1.0, 0.0, 1.0], np.float32)
    q = LinearQ.fit(S, A, R, S2, D, n_actions=2, gamma=0.9, iters=20)
    assert q.act(np.array([0.0, 0.0], np.float32)) == 1
    assert q.n_params() == q.W.size
    assert q.nbytes() == q.W.size * 4


def test_state_bytes_ordering(whitener: Whitener) -> None:
    w = ReservoirWeights.build(ReservoirConfig(n_res=400, density=0.05), 40)
    sizes = {
        "window2": WindowMemory(40, 2).state_nbytes(),
        "q8": QuantizedLatent(ReservoirMemory(w, 4.0), whitener, bits=8).state_nbytes(),
        "q16": QuantizedLatent(ReservoirMemory(w, 4.0), whitener, bits=16).state_nbytes(),
        "esn": ReservoirMemory(w, 4.0).state_nbytes(),
    }
    assert sizes["q8"] < sizes["q16"] < sizes["esn"] == 1600


def test_random_tanh_does_not_change_resident_state() -> None:
    """The null layer is a fixed table: it must not inflate the budgeted state."""
    base = WindowMemory(8, 2)
    feats = RandomTanhFeatures(base, out_dim=32, seed=0)
    assert feats.state_nbytes() == base.state_nbytes()  # resident state unchanged
    assert feats.dim == 32
    assert feats.fixed_nbytes() > base.fixed_nbytes()  # the table is a fixed cost
    # deterministic and bounded in (-1, 1)
    feats.reset()
    out = feats.step(np.ones(8, np.float32))
    assert out.shape == (32,)
    assert np.all(np.abs(out) < 1.0)
    feats.reset()
    assert np.allclose(feats.step(np.ones(8, np.float32)), out)


def test_random_tanh_is_shared_across_memories() -> None:
    """Same seed and base dim => same projection, so window and reservoir see identical features."""
    w = ReservoirWeights.build(ReservoirConfig(n_res=16, density=0.1), 8)
    a = RandomTanhFeatures(WindowMemory(8, 2), out_dim=16, seed=3)  # window dim 16
    b = RandomTanhFeatures(ReservoirMemory(w, 4.0), out_dim=16, seed=3)  # reservoir dim 16
    assert a.W.shape == b.W.shape
    assert np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b)
