"""The collect-once / fit-offline path: episode resets and transition alignment.

The benchmark's speed comes from replaying memories over a *cached* latent sequence instead of
re-rolling the engine per arm. That only stays honest if the replayed transitions match what a
live rollout would have produced, so these tests pin the alignment (S is the state before an
action, S2 after it) and the episode-boundary reset against a direct ``collect``-style replay.
"""

from __future__ import annotations

import numpy as np

from doom_esn.harness import roll_memory_over_latents
from doom_esn.memory import ReservoirMemory, WindowMemory
from doom_esn.reservoir import ReservoirConfig, ReservoirWeights


def _reference(memory: object, U: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The live-rollout alignment: step U[0], then for each t emit (S, S2)."""
    S, S2 = [], []
    memory.reset()
    s = memory.step(U[0])
    for t in range(1, len(U)):
        S.append(s)
        s = memory.step(U[t])
        S2.append(s)
        if D[t - 1]:
            memory.reset()
            s = memory.step(U[t])
    return np.asarray(S, np.float32), np.asarray(S2, np.float32)


def test_roll_matches_live_rollout_alignment() -> None:
    rng = np.random.default_rng(0)
    U = rng.standard_normal((30, 4)).astype(np.float32)
    A = rng.integers(0, 3, 30)
    R = rng.standard_normal(30).astype(np.float32)
    D = np.zeros(30, np.float32)
    D[[9, 19, 29]] = 1.0  # three episodes
    m = WindowMemory(4, 2)
    S, Aa, _Rr, S2, Dd = roll_memory_over_latents(m, U, A, R, D)
    Sr, S2r = _reference(WindowMemory(4, 2), U, D)
    np.testing.assert_allclose(S[: len(Sr)], Sr)
    np.testing.assert_allclose(S2[: len(S2r)], S2r)
    np.testing.assert_array_equal(Aa[: len(Sr)], A[: len(Sr)])
    np.testing.assert_array_equal(Dd[: len(Sr)], D[: len(Sr)])
    assert len(S) == len(Sr) + 1  # + the terminal step of the last episode


def test_memory_resets_at_episode_boundary() -> None:
    # Two identical frames, one per episode: with a reset the second state equals the first.
    U = np.ones((2, 3), np.float32)
    D = np.array([1.0, 1.0], np.float32)
    w = ReservoirWeights.build(ReservoirConfig(n_res=32, density=0.2), 3)
    S, _A, _R, _S2, _D = roll_memory_over_latents(
        ReservoirMemory(w, 4.0), U, np.zeros(2, np.int64), np.zeros(2, np.float32), D
    )
    # last row is the terminal step of the second episode; its state must equal a fresh
    # memory driven by a single identical frame (i.e. the reset actually happened).
    fresh = ReservoirMemory(w, 4.0)
    fresh.reset()
    expected = fresh.step(U[0])
    np.testing.assert_allclose(S[-1], expected, rtol=1e-5)


def test_terminal_last_step_is_kept() -> None:
    U = np.ones((5, 3), np.float32)
    D = np.zeros(5, np.float32)
    D[-1] = 1.0
    m = WindowMemory(3, 1)
    S, _A, _R, _S2, Dd = roll_memory_over_latents(
        m, U, np.zeros(5, np.int64), np.ones(5, np.float32), D
    )
    assert len(S) == 5  # 4 interior transitions + the terminal one
    assert Dd[-1] == 1.0


def test_nonterminal_last_step_is_dropped() -> None:
    U = np.ones((5, 3), np.float32)
    D = np.zeros(5, np.float32)  # never terminal (e.g. truncation at the step cap)
    m = WindowMemory(3, 1)
    S, _A, _R, _S2, _D = roll_memory_over_latents(
        m, U, np.zeros(5, np.int64), np.ones(5, np.float32), D
    )
    assert len(S) == 4
