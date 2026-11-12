"""Return vs resident-memory-bytes for several memory back-ends on one ViZDoom scenario.

Protocol (see docs/ANALYSIS_PLAN.md):
  * fit a frozen pretrained spatial encoder (no from-scratch training),
  * collect ONE latent dataset per seed under a uniform-random policy and cache it to disk;
    every back-end is then fitted on the *same* latents, so no arm can win by getting luckier
    rollouts and the expensive encoder rollout is paid once instead of once per arm,
  * fit a linear fitted-Q readout offline on that dataset (one ridge solve per iteration),
  * evaluate the greedy policy on held-out episodes (seeds disjoint from training) and compare
    against two nulls: uniform random and the best fixed action,
  * report mean return, standard error, resident state bytes, fixed bytes, and ms/frame.

Why offline: the encoder is the bottleneck (~3.5 ms/frame single-frame, ~0.4 ms/frame batched),
so re-rolling the engine and the CNN for every (arm, seed, round) made one scenario take ~30-45
minutes and lose all results if the process was killed. Collecting the latents once and fitting
each arm in pure numpy cuts that to a few minutes and makes the run checkpointable: each
``(backend, seed)`` row is appended to a ``.jsonl`` as soon as it is computed, and ``--resume``
skips rows already present.

Equal-bytes arms (``--window-ks``) match the reservoir's byte budget with windows and quantised
windows so "reservoir vs window" is a comparison at the same number of bytes.

Usage:
    python bench/memory_budget.py --scenario VizdoomTakeCover-v1 --perception mobilenet \
        --seeds 3 --eval-episodes 20
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from doom_esn.envs import make_doom_env
from doom_esn.features import Whitener
from doom_esn.harness import (
    RES_CFG,
    best_constant_action,
    build_encoder,
    collect_latents,
    encode_pool,
    evaluate_seeds,
    probe_states,
    roll_memory_over_latents,
    uniform_random_returns,
)
from doom_esn.memory import (
    CodebookMemory,
    ProjectedMemory,
    QuantizedLatent,
    QuantizedWindow,
    RandomTanhFeatures,
    ReservoirMemory,
    WhitenedWindow,
)
from doom_esn.readout import LinearQ
from doom_esn.reservoir import ReservoirWeights


def build_backends(
    enc: Any,
    weights: ReservoirWeights,
    whitener: Whitener,
    whitener_lat: Whitener,
    dt: float,
    codebook: np.ndarray,
    window_ks: tuple[int, ...] = (1, 2, 4, 8),
    axes: int = 48,
) -> dict[str, Callable[[], Any]]:
    """Named back-end constructors. Each returns a fresh memory with exact byte accounting.

    ``axes`` is the number of whitened axes per frame; a ``WhitenedWindow(axes, frames)``
    stores ``axes*frames`` floats, so sweeping ``frames`` traces a return-vs-bytes curve
    for the *same* representation. The raw (un-whitened) grid latent is far too wide to feed
    a solved linear readout directly, so every window arm projects per frame first (this is
    the repo's own ``FeatureObservation`` rule: project, then window).

    ``whitener`` projects *reservoir states*; ``whitener_lat`` projects *encoder latents*.
    The two live in different spaces (reservoir width vs encoder width), so the window arms
    must not reuse the reservoir whitener.
    """

    def res() -> ReservoirMemory:
        return ReservoirMemory(weights, dt)

    backends: dict[str, Callable[[], Any]] = {
        "esn400": res,
        "esn400_white24": lambda: ProjectedMemory(res(), whitener),
        "esn400_q8_24": lambda: QuantizedLatent(res(), whitener, clip=3.0, bits=8),
        "esn400_q16_24": lambda: QuantizedLatent(res(), whitener, clip=3.0, bits=16),
        "esn400_vq256": lambda: CodebookMemory(res(), whitener, codebook),
        # fair null: same fixed random nonlinearity on both memories, so "window vs
        # reservoir" compares the memory mechanism, not linear-vs-nonlinear features.
        "esn400_tanh400": lambda: RandomTanhFeatures(res(), 400, seed=0),
    }
    # the return-vs-bytes curve: a whitened window at 1..16 frames (192 B .. 3072 B)
    for k in window_ks:
        backends[f"win_white_k{k}"] = lambda k=k: WhitenedWindow(enc.dim, axes, k, whitener_lat)
    # quantised windows: same axes/frames, fewer bytes per axis
    backends["win_q8_k8"] = lambda: QuantizedWindow(enc.dim, axes, 8, whitener_lat, bits=8)
    backends["win_q16_k8"] = lambda: QuantizedWindow(enc.dim, axes, 8, whitener_lat, bits=16)
    # a memoryless (k=1) quantised arm: does quantisation help with no temporal memory?
    backends["win_q8_k1"] = lambda: QuantizedWindow(enc.dim, axes, 1, whitener_lat, bits=8)
    return backends


def load_or_collect_latents(
    env: Any,
    enc: Any,
    scenario: str,
    seed: int,
    episodes: int,
    max_steps: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One random-policy latent dataset per seed, cached on disk.

    The cache key is the scenario, seed and collection size, so changing any of them
    re-collects. Stored as ``.npz`` with ``allow_pickle=False`` (data only, no objects).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{scenario}_s{seed}_e{episodes}_n{max_steps}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z["U"], z["A"], z["R"], z["D"]
    U, A, R, D = collect_latents(env, enc, episodes, 1000 * seed + 1, max_steps)
    np.savez(path, U=U, A=A, R=R, D=D)
    return U, A, R, D


def fit_readout_offline(
    memory_ctor: Callable[[], Any],
    U: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    n_act: int,
    iters: int,
    gamma: float,
    ridge: float,
) -> LinearQ:
    """Roll a memory over the cached latents and solve the linear fitted-Q readout."""
    S, Aa, Rr, S2, Dd = roll_memory_over_latents(memory_ctor(), U, A, R, D)
    return LinearQ.fit(S, Aa, Rr, S2, Dd, n_act, gamma=gamma, ridge=ridge, iters=iters)


def load_checkpoint(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read the per-row ``.jsonl`` checkpoint; the last row for a key wins."""
    done: dict[tuple[str, int], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["backend"], row["seed"])] = row
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="VizdoomTakeCover-v1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--collect-episodes", type=int, default=40)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--iters", type=int, default=30, help="fitted-Q iterations per arm")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--probe-steps", type=int, default=800)
    ap.add_argument("--perception", choices=["pool", "mobilenet"], default="mobilenet")
    ap.add_argument("--grid", type=int, default=3, help="MobileNet spatial grid (1 = pooled)")
    ap.add_argument("--window-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--axes", type=int, default=48, help="whitened axes per frame")
    ap.add_argument("--only", default="", help="comma-separated substrings to keep backends")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("docs/results/memory"))
    ap.add_argument("--cache-dir", type=Path, default=Path("docs/results/latents"))
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse rows already in the .jsonl checkpoint and skip finished (backend, seed) pairs",
    )
    args = ap.parse_args()

    env = make_doom_env(args.scenario, 4)
    n_act = int(env.action_space.n)
    dt = 4.0
    args.out.mkdir(parents=True, exist_ok=True)

    # --- one-time setup: encoder, reservoir, whitener, codebook (all fitted on the probe) ---
    enc = build_encoder(args.perception, env, args.probe_steps, grid=args.grid)
    weights = ReservoirWeights.build(RES_CFG, enc.dim)
    U_probe = encode_pool(env, enc, args.probe_steps, seed=1)
    X = probe_states(weights, U_probe, dt)
    whitener = Whitener.fit(X, 48)
    # the window arms whiten the *encoder latents* (a different space from the reservoir
    # states), so fit a separate projection on the raw latents.
    whitener_lat = Whitener.fit(U_probe, 48)
    codebook = CodebookMemory.fit(ReservoirMemory(weights, dt), whitener, U_probe, k=256, seed=0).cb

    eval_seeds = [10**7 + i for i in range(args.eval_episodes)]

    # --- null baselines, evaluated on the *same* episodes as every arm (paired) ---
    rand = uniform_random_returns(env, eval_seeds, args.max_steps)
    best_a, best_const = best_constant_action(env, eval_seeds, args.max_steps)

    def baseline_row(backend: str, rets: np.ndarray) -> dict[str, Any]:
        return {
            "backend": backend,
            "scenario": args.scenario,
            "seed": -1,
            "state_bytes": 0,
            "fixed_bytes": 0,
            "readout_bytes": 0,
            "dim": 0,
            "return_mean": float(rets.mean()),
            "return_sem": float(rets.std() / np.sqrt(len(rets))),
            "ms_per_frame_p50": 0.0,
            "ms_per_frame_p95": 0.0,
            "n_transitions": 0,
            "best_round": -1,
            "paired_vs_random": float((rets - rand).mean()),
            "paired_vs_random_se": float((rets - rand).std() / np.sqrt(len(rets))),
            "paired_vs_const": float((rets - best_const).mean()),
        }

    baseline_rows = [
        baseline_row("random", rand),
        baseline_row(f"const_a{best_a}", best_const),
    ]

    backends = build_backends(
        enc, weights, whitener, whitener_lat, dt, codebook, tuple(args.window_ks), args.axes
    )
    if args.only:
        keep = tuple(s.strip() for s in args.only.split(",") if s.strip())
        backends = {n: c for n, c in backends.items() if any(k in n for k in keep)}

    # --- collect (or load) the shared latent dataset, one per seed ---
    datasets = [
        load_or_collect_latents(
            env, enc, args.scenario, seed, args.collect_episodes, args.max_steps, args.cache_dir
        )
        for seed in range(args.seeds)
    ]

    fname = f"{args.scenario}{args.tag}.json"
    ckpt = args.out / f"{args.scenario}{args.tag}.jsonl"
    done = load_checkpoint(ckpt) if args.resume else {}
    if not args.resume and ckpt.exists():
        ckpt.unlink()  # a fresh (non-resume) run must not mix with stale rows
    rows: list[dict[str, Any]] = []
    if args.resume:
        print(f"resumed {len(done)} rows from {ckpt}", flush=True)

    for name, ctor in backends.items():
        mem = ctor()
        for seed in range(args.seeds):
            if (name, seed) in done:
                continue
            U, A, R, D = datasets[seed]
            q = fit_readout_offline(ctor, U, A, R, D, n_act, args.iters, args.gamma, args.ridge)
            rets = evaluate_seeds(env, enc, ctor(), q.act, eval_seeds, args.max_steps)
            row = dict(
                backend=name,
                scenario=args.scenario,
                seed=seed,
                state_bytes=mem.state_nbytes(),
                fixed_bytes=mem.fixed_nbytes(),
                readout_bytes=q.nbytes(),
                dim=int(mem.dim),
                return_mean=float(rets.mean()),
                return_sem=float(rets.std() / np.sqrt(len(rets))),
                ms_per_frame_p50=0.0,
                ms_per_frame_p95=0.0,
                n_transitions=len(U),
                best_round=-1,
                paired_vs_random=float((rets - rand).mean()),
                paired_vs_random_se=float((rets - rand).std() / np.sqrt(len(rets))),
                paired_vs_const=float((rets - best_const).mean()),
            )
            rows.append(row)
            with ckpt.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    # final JSON = baselines + every arm row ever checkpointed (resumed rows included)
    all_rows = list(baseline_rows)
    all_rows.extend(r for (n, _s), r in sorted(done.items()) if n not in {"random", "const_a"})
    all_rows.extend(rows)
    (args.out / fname).write_text(json.dumps(all_rows, indent=2))
    print("reservoir_cfg", json.dumps(asdict(RES_CFG)))
    env.close()


if __name__ == "__main__":
    main()
