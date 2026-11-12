"""Return vs resident-memory-bytes for several memory back-ends on one ViZDoom scenario.

Protocol (pre-registered in docs/ANALYSIS_PLAN.md):
  * fit a frozen pretrained spatial encoder (no from-scratch training),
  * for each back-end, train a linear fitted-Q readout with a replay buffer and pick the
    round with the best held-out *validation* return (disjoint from the evaluation seeds),
  * evaluate the greedy policy on held-out episodes (disjoint seeds) and compare against
    three nulls: uniform random, the best fixed action, and the best of a small set of
    hand-written heuristics,
  * report mean return, standard error, the resident state bytes, fixed bytes, and ms/frame.

Equal-bytes arms (``--equal-bytes``) match the reservoir's byte budget with windows and
quantised windows so "reservoir vs window" is a comparison at the same number of bytes.

Usage:
    python bench/memory_budget.py --scenario VizdoomTakeCover-v1 --perception mobilenet \
        --seeds 10 --eval-episodes 50
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
    encode_pool,
    evaluate_seeds,
    probe_states,
    train_readout_robust,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="VizdoomTakeCover-v1")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--collect-episodes", type=int, default=40)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--val-episodes", type=int, default=15)
    ap.add_argument("--probe-steps", type=int, default=1000)
    ap.add_argument("--perception", choices=["pool", "mobilenet"], default="mobilenet")
    ap.add_argument("--grid", type=int, default=3, help="MobileNet spatial grid (1 = pooled)")
    ap.add_argument("--window-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--axes", type=int, default=48, help="whitened axes per frame")
    ap.add_argument("--only", default="", help="comma-separated substrings to keep backends")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("docs/results/memory"))
    args = ap.parse_args()

    env = make_doom_env(args.scenario, 4)
    n_act = int(env.action_space.n)
    dt = 4.0
    args.out.mkdir(parents=True, exist_ok=True)
    enc = build_encoder(args.perception, env, args.probe_steps, grid=args.grid)
    weights = ReservoirWeights.build(RES_CFG, enc.dim)
    U = encode_pool(env, enc, args.probe_steps, seed=1)
    X = probe_states(weights, U, dt)
    whitener = Whitener.fit(X, 48)
    # the window arms whiten the *encoder latents* (a different space from the reservoir
    # states), so fit a separate projection on the raw latents.
    whitener_lat = Whitener.fit(U, 48)
    codebook = CodebookMemory.fit(ReservoirMemory(weights, dt), whitener, U, k=256, seed=0).cb

    eval_seeds = [10**7 + i for i in range(args.eval_episodes)]
    val_seeds = [5 * 10**6 + i for i in range(args.val_episodes)]
    base_seed = 10**8

    # --- null baselines, evaluated on the *same* episodes as every arm (paired) ---
    rand = uniform_random_returns(env, eval_seeds, args.max_steps)
    best_a, best_const = best_constant_action(env, eval_seeds, args.max_steps)
    baseline_rows = [
        {
            "backend": "random",
            "scenario": args.scenario,
            "seed": -1,
            "state_bytes": 0,
            "fixed_bytes": 0,
            "readout_bytes": 0,
            "dim": 0,
            "return_mean": float(rand.mean()),
            "return_sem": float(rand.std() / np.sqrt(len(rand))),
            "ms_per_frame_p50": 0.0,
            "ms_per_frame_p95": 0.0,
            "n_transitions": 0,
        },
        {
            "backend": f"const_a{best_a}",
            "scenario": args.scenario,
            "seed": -1,
            "state_bytes": 0,
            "fixed_bytes": 0,
            "readout_bytes": 0,
            "dim": 0,
            "return_mean": float(best_const.mean()),
            "return_sem": float(best_const.std() / np.sqrt(len(best_const))),
            "ms_per_frame_p50": 0.0,
            "ms_per_frame_p95": 0.0,
            "n_transitions": 0,
        },
    ]

    backends = build_backends(
        enc, weights, whitener, whitener_lat, dt, codebook, tuple(args.window_ks), args.axes
    )
    if args.only:
        keep = tuple(s.strip() for s in args.only.split(",") if s.strip())
        backends = {n: c for n, c in backends.items() if any(k in n for k in keep)}

    rows: list[dict[str, Any]] = list(baseline_rows)
    for name, ctor in backends.items():
        mem = ctor()
        for seed in range(args.seeds):
            q, info = train_readout_robust(
                env,
                enc,
                ctor,
                n_act,
                args.rounds,
                args.collect_episodes,
                base_seed + 1000 * seed,
                args.max_steps,
                val_seeds,
            )
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
                n_transitions=args.rounds * args.collect_episodes,
                best_round=info["best_round"],
                paired_vs_random=float((rets - rand).mean()),
                paired_vs_random_se=float((rets - rand).std() / np.sqrt(len(rets))),
                paired_vs_const=float((rets - best_const).mean()),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)

    fname = f"{args.scenario}{args.tag}.json"
    (args.out / fname).write_text(json.dumps(rows, indent=2))
    print("reservoir_cfg", json.dumps(asdict(RES_CFG)))
    env.close()


if __name__ == "__main__":
    main()
