"""Return vs resident-memory-bytes for several memory back-ends on one ViZDoom scenario.

Protocol (pre-registered in docs/RESEARCH_PROGRAM.md):
  * collect transitions with a uniform-random policy (train seeds),
  * fit a linear fitted-Q readout per back-end (solved, no ES search noise),
  * evaluate the greedy policy on held-out episodes (disjoint seeds),
  * report mean return, the resident state bytes, fixed bytes, and ms/frame.

Usage:
    python bench/memory_budget.py --scenario VizdoomDefendLine-v1 --seeds 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from doom_esn.envs import make_doom_env
from doom_esn.features import Whitener
from doom_esn.harness import (
    RES_CFG,
    build_encoder,
    encode_pool,
    evaluate,
    probe_states,
    train_readout,
)
from doom_esn.memory import (
    CodebookMemory,
    ProjectedMemory,
    QuantizedLatent,
    RandomTanhFeatures,
    ReservoirMemory,
    WindowMemory,
)
from doom_esn.reservoir import ReservoirWeights


def build_backends(enc, weights, whitener24, dt, codebook, window_ks=(2, 4)):
    def res():
        return ReservoirMemory(weights, dt)

    backends = {
        "esn400": res,
        "esn400_white24": lambda: ProjectedMemory(res(), whitener24),
        "esn400_q8_24": lambda: QuantizedLatent(res(), whitener24, clip=3.0, bits=8),
        "esn400_q16_24": lambda: QuantizedLatent(res(), whitener24, clip=3.0, bits=16),
        "esn400_vq256": lambda: CodebookMemory(res(), whitener24, codebook),
        # fair null: same fixed random nonlinearity on both memories, so "window vs
        # reservoir" compares the memory mechanism, not linear-vs-nonlinear features.
        "esn400_tanh400": lambda: RandomTanhFeatures(res(), 400, seed=0),
    }
    for k in window_ks:
        backends[f"window_k{k}"] = lambda k=k: WindowMemory(enc.dim, k)
        backends[f"window_k{k}_tanh400"] = lambda k=k: RandomTanhFeatures(
            WindowMemory(enc.dim, k), 400, seed=0
        )
    return backends


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="VizdoomDefendLine-v1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--collect-episodes", type=int, default=40)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--probe-steps", type=int, default=1200)
    ap.add_argument("--perception", choices=["pool", "mobilenet"], default="pool")
    ap.add_argument("--window-ks", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--only", default="", help="comma-separated substrings to keep backends")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("docs/results/memory"))
    args = ap.parse_args()

    env = make_doom_env(args.scenario, 4)
    n_act = int(env.action_space.n)
    dt = 4.0
    args.out.mkdir(parents=True, exist_ok=True)
    enc = build_encoder(args.perception, env, args.probe_steps)
    weights = ReservoirWeights.build(RES_CFG, enc.dim)
    U = encode_pool(env, enc, args.probe_steps, seed=1)
    X = probe_states(weights, U, dt)
    whitener24 = Whitener.fit(X, 24)
    codebook = CodebookMemory.fit(ReservoirMemory(weights, dt), whitener24, U, k=256, seed=0).cb

    backends = build_backends(enc, weights, whitener24, dt, codebook, tuple(args.window_ks))
    if args.only:
        keep = tuple(s.strip() for s in args.only.split(",") if s.strip())
        backends = {n: c for n, c in backends.items() if any(k in n for k in keep)}
    rows = []
    for name, ctor in backends.items():
        mem = ctor()
        for seed in range(args.seeds):
            q = train_readout(
                env,
                enc,
                ctor(),
                n_act,
                args.rounds,
                args.collect_episodes,
                10_000 + seed,
                args.max_steps,
            )
            rets, times = evaluate(
                env, enc, ctor(), q.act, args.eval_episodes, 10**7 + seed, args.max_steps
            )
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
                ms_per_frame_p50=float(np.percentile(times, 50)),
                ms_per_frame_p95=float(np.percentile(times, 95)),
                n_transitions=args.rounds * args.collect_episodes,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    fname = f"{args.scenario}{args.tag}.json"
    (args.out / fname).write_text(json.dumps(rows, indent=2))
    print("reservoir_cfg", json.dumps(asdict(RES_CFG)))
    env.close()


if __name__ == "__main__":
    main()
