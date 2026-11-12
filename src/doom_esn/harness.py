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


def build_encoder(kind: str, env: Any, probe_steps: int, grid: int = 3) -> Encoder:
    """Return an encoder by name. ``pool`` is fitted here; ``mobilenet`` is pretrained.

    ``grid`` sets the MobileNet spatial resolution: ``grid=1`` is global average pooling
    (the original, spatially blind latent), ``grid=3`` keeps a 3x3 map so the readout can
    see *where* things are on screen.
    """
    if kind == "pool":
        enc = FrameEncoder(24, 32)
        enc.fit(encode_pool(env, enc, probe_steps, seed=0))
        return enc
    if kind == "mobilenet":
        from .perception import MobileNetEncoder

        return MobileNetEncoder(size=96, pretrained=True, grid=grid)
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


def evaluate_seeds(
    env: Any, enc: Encoder, memory: Any, policy: Any, seeds: list[int], max_steps: int
) -> F32:
    """Greedy evaluation on an explicit list of episode seeds (for paired comparisons)."""
    rets = []
    for sd in seeds:
        total, _, _ = rollout(env, enc, memory, policy, sd, max_steps)
        rets.append(total)
    return np.asarray(rets, np.float32)


def uniform_random_returns(env: Any, seeds: list[int], max_steps: int) -> F32:
    """Return of a uniform-random policy on the same episodes.

    The RNG is seeded once per episode (not per step) and draws a *fresh* action every
    step, so this is a genuine uniform-random baseline -- the bug the repo fixed in
    ``scripts/play.py`` was re-seeding every step, which silently made it constant-action.
    """
    n_act = int(env.action_space.n)
    rets = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        _, _ = env.reset(seed=sd)
        total = 0.0
        for _ in range(max_steps):
            _, r, term, trunc, _ = env.step(int(rng.integers(n_act)))
            total += r
            if term or trunc:
                break
        rets.append(total)
    return np.asarray(rets, np.float32)


def constant_action_returns(env: Any, action: int, seeds: list[int], max_steps: int) -> F32:
    """Return of a fixed action repeated every step (the 'always do the same thing' null)."""
    rets = []
    for sd in seeds:
        _, _ = env.reset(seed=sd)
        total = 0.0
        for _ in range(max_steps):
            _, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rets.append(total)
    return np.asarray(rets, np.float32)


def best_constant_action(env: Any, seeds: list[int], max_steps: int) -> tuple[int, F32]:
    """The best single fixed action, chosen on ``seeds`` (a strong memoryless baseline)."""
    best_a, best = 0, None
    for a in range(int(env.action_space.n)):
        r = constant_action_returns(env, a, seeds, max_steps)
        if best is None or r.mean() > best.mean():
            best_a, best = a, r
    assert best is not None
    return best_a, best


class _ReplayBuffer:
    """Append-only transition store. Fitted-Q refits on *all* data seen so far.

    The original ``train_readout`` refit on only the latest round's fresh transitions, so
    each refit could forget what it had learned (round-to-round return oscillated by +-100
    on TakeCover). Accumulating the data makes each refit strictly better-informed and the
    policy monotone-ish instead of noisy.
    """

    def __init__(self) -> None:
        self.S: list[F32] = []
        self.A: list[np.ndarray] = []
        self.R: list[F32] = []
        self.S2: list[F32] = []
        self.D: list[np.ndarray] = []

    def add(self, data: tuple[F32, np.ndarray, F32, F32, np.ndarray]) -> None:
        S, A, R, S2, D = data
        self.S.append(S)
        self.A.append(A)
        self.R.append(R)
        self.S2.append(S2)
        self.D.append(D)

    def arrays(self) -> tuple[F32, np.ndarray, F32, F32, np.ndarray]:
        return (
            np.concatenate(self.S, axis=0),
            np.concatenate(self.A, axis=0),
            np.concatenate(self.R, axis=0),
            np.concatenate(self.S2, axis=0),
            np.concatenate(self.D, axis=0),
        )

    def __len__(self) -> int:
        return int(sum(len(s) for s in self.S))


def train_readout_robust(
    env: Any,
    enc: Encoder,
    memory_ctor: Any,
    n_act: int,
    rounds: int,
    eps_episodes: int,
    seed0: int,
    max_steps: int,
    val_seeds: list[int],
    gamma: float = 0.99,
    ridge: float = 1e-3,
    iters: int = 20,
) -> tuple[LinearQ, dict[str, Any]]:
    """Iterated fitted-Q with a replay buffer and validation-based model selection.

    Each round: collect with an epsilon-greedy policy, add to the buffer, refit the linear
    readout on the whole buffer, and score the greedy policy on ``val_seeds``. The returned
    readout is the round with the best *validation* return, which removes the round-to-round
    regression that made the original learner no better than random. ``val_seeds`` must be
    disjoint from the evaluation seeds.
    """
    buf = _ReplayBuffer()
    q = LinearQ(np.zeros((memory_ctor().dim + 1, n_act), np.float32))
    best_q, best_val, best_round = q, -np.inf, -1
    history: list[dict[str, Any]] = []
    for rnd in range(rounds):
        eps = 1.0 if rnd == 0 else max(0.05, 0.6**rnd)
        data = collect(
            env, enc, memory_ctor(), q.act, eps_episodes, seed0 + 1000 * rnd, max_steps, eps
        )
        buf.add(data)
        S, A, R, S2, D = buf.arrays()
        q = LinearQ.fit(S, A, R, S2, D, n_act, gamma=gamma, ridge=ridge, iters=iters)
        val = float(evaluate_seeds(env, enc, memory_ctor(), q.act, val_seeds, max_steps).mean())
        history.append({"round": rnd, "eps": eps, "transitions": len(buf), "val_return": val})
        if val > best_val:
            best_q, best_val, best_round = q, val, rnd
    return best_q, {"best_round": best_round, "best_val_return": best_val, "history": history}
