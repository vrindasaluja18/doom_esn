"""Shared experiment plumbing: encoders, random-policy data, reservoir config.

The benchmark, the play script and the topology probe all need the same three things: an
encoder selected by name, a batch of random-policy latents to fit a whitener/codebook on, and
the fixed reservoir configuration. Keeping them here means one definition instead of three
copies that can drift apart.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

from ._types import F32
from .encoder import FrameEncoder
from .memory import ProjectedMemory, ReservoirMemory
from .readout import LinearQ
from .reservoir import ReservoirConfig, ReservoirWeights

RES_CFG = ReservoirConfig(
    n_res=400, spectral_radius=1.1, density=0.05, input_scaling=3.0, tau=4.0, seed=0
)
DT = 4.0


class Encoder(Protocol):
    @property
    def dim(self) -> int: ...

    def __call__(self, frame: np.ndarray) -> F32: ...


def build_encoder(kind: str, env: Any, probe_steps: int) -> Encoder:
    """Return an encoder by name. ``pool`` is fitted here; ``mobilenet`` is pretrained."""
    if kind == "pool":
        enc = FrameEncoder(24, 32)
        enc.fit(encode_pool(env, enc, probe_steps, seed=0))
        return enc
    if kind == "mobilenet":
        from .perception import MobileNetEncoder

        return MobileNetEncoder(size=96, pretrained=True)
    raise ValueError(f"unknown encoder {kind!r}")


def encode_pool(env: Any, enc: Encoder, n_steps: int, seed: int = 0) -> F32:
    """Roll a uniform-random policy and keep the encoded frames (n_steps, enc.dim)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_steps, enc.dim), np.float32)
    obs, _ = env.reset(seed=seed)
    for t in range(n_steps):
        out[t] = enc(obs["screen"])
        obs, _, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            obs, _ = env.reset()
    return out


def probe_states(weights: ReservoirWeights, U: F32, dt: float = DT) -> F32:
    """Reservoir states driven by an already-encoded sequence (for fitting the whitener)."""
    x = weights.zeros()
    states = []
    for u in U:
        x = weights.step(x, u, dt)
        states.append(x)
    return np.asarray(states, np.float32)


def reservoir_memory(weights: ReservoirWeights, dt: float = DT) -> ReservoirMemory:
    return ReservoirMemory(weights, dt)


def projected_reservoir(
    weights: ReservoirWeights, whitener: Any, dt: float = DT
) -> ProjectedMemory:
    return ProjectedMemory(ReservoirMemory(weights, dt), whitener)


def collect(
    env: Any,
    enc: Encoder,
    memory: Any,
    policy: Any,
    n_episodes: int,
    seed0: int,
    max_steps: int,
    eps: float = 1.0,
) -> tuple[F32, np.ndarray, F32, F32, F32]:
    """Collect (s, a, r, s', done) transitions with an epsilon-greedy policy on ``policy``."""
    n_act = int(env.action_space.n)
    rng = np.random.default_rng(seed0)
    S, A, R, S2, D = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        memory.reset()
        s = memory.step(enc(obs["screen"]))
        for _ in range(max_steps):
            a = int(rng.integers(n_act)) if rng.random() < eps else policy(s)
            obs, r, term, trunc, _ = env.step(a)
            s2 = memory.step(enc(obs["screen"]))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(float(term or trunc))
            s = s2
            if term or trunc:
                break
    return (
        np.asarray(S, np.float32),
        np.asarray(A),
        np.asarray(R, np.float32),
        np.asarray(S2, np.float32),
        np.asarray(D, np.float32),
    )


def train_readout(
    env: Any,
    enc: Encoder,
    memory: Any,
    n_act: int,
    rounds: int,
    eps_episodes: int,
    seed0: int,
    max_steps: int,
) -> LinearQ:
    """Iterated fitted Q-iteration: collect with the current greedy policy, refit, improve."""
    q = LinearQ(np.zeros((memory.dim + 1, n_act), np.float32))
    for rnd in range(rounds):
        eps = 1.0 if rnd == 0 else max(0.1, 0.5**rnd)
        S, A, R, S2, D = collect(
            env, enc, memory, q.act, eps_episodes, seed0 + 1000 * rnd, max_steps, eps
        )
        q = LinearQ.fit(S, A, R, S2, D, n_act, gamma=0.99, ridge=1e-3, iters=15)
    return q


def rollout(
    env: Any,
    enc: Encoder,
    memory: Any,
    policy: Any,
    seed: int,
    max_steps: int,
    capture: bool = False,
) -> tuple[float, list[np.ndarray], list[float]]:
    """Run one episode. Returns (return, frames, per-step ms) -- frames only if capturing."""
    obs, _ = env.reset(seed=seed)
    memory.reset()
    s = memory.step(enc(obs["screen"]))
    frames: list[np.ndarray] = []
    times: list[float] = []
    total = 0.0
    for _ in range(max_steps):
        if capture:
            frames.append(np.asarray(obs["screen"], np.uint8).copy())
        t0 = time.perf_counter()
        a = policy(s)
        obs, r, term, trunc, _ = env.step(a)
        s = memory.step(enc(obs["screen"]))
        times.append((time.perf_counter() - t0) * 1e3)
        total += r
        if term or trunc:
            if capture:
                frames.append(np.asarray(obs["screen"], np.uint8).copy())
            break
    return total, frames, times


def evaluate(
    env: Any, enc: Encoder, memory: Any, policy: Any, n_episodes: int, seed0: int, max_steps: int
) -> tuple[F32, F32]:
    """Held-out greedy evaluation. Returns (returns, per-step ms)."""
    rets, times = [], []
    for ep in range(n_episodes):
        total, _, t = rollout(env, enc, memory, policy, seed0 + ep, max_steps)
        rets.append(total)
        times.extend(t)
    return np.asarray(rets, np.float32), np.asarray(times, np.float32)
