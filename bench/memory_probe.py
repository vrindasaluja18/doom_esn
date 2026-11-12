"""Direct, low-variance measurement of what each memory arm retains about the past.

Task return is a noisy *indirect* proxy: on TakeCover the random baseline is ~244 and the
per-seed spread swamps the effect, so "did the memory help" is hard to answer with returns
alone. This script measures the memory itself, on the *same* cached latents the return
benchmark uses, with a standard linear-probe test:

  for each delay k, fit a ridge readout from the memory state ``s_t`` to the whitened latent
  ``k`` steps in the past, and report held-out R^2.

R^2(k) is the *delay-memory capacity*: a memoryless arm (window k=1) can only decode k=1; a
memory that keeps more of the past decodes larger k. It is a regression score on a fixed,
cached dataset, so it has small error bars and takes seconds per arm -- the opposite of the
noisy return curve. It does not replace the return benchmark; it tells us *why* a return curve
looks flat (is the state informative at all?) and it can rank arms the return noise cannot.

    python bench/memory_probe.py --scenario VizdoomTakeCover-v1 --perception mobilenet --grid 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# reuse the exact arm set of the return benchmark so the two tables line up; bench/ is a
# script directory, not an installed package, so make it importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.memory_budget import build_backends

from doom_esn.envs import make_doom_env
from doom_esn.features import Whitener
from doom_esn.harness import (
    RES_CFG,
    build_encoder,
    encode_pool,
    probe_states,
    roll_memory_over_latents,
)
from doom_esn.memory import CodebookMemory, ReservoirMemory
from doom_esn.reservoir import ReservoirWeights


def episode_bounds(done: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` index of each episode in a concatenated transition stream."""
    ends = np.flatnonzero(done) + 1
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def ridge_r2(
    X: np.ndarray, Y: np.ndarray, train: np.ndarray, test: np.ndarray, alpha: float = 1e-2
) -> float:
    """Held-out R^2 of a ridge fit X -> Y (both standardised), scored per output column."""
    xm, xs = X[train].mean(0), X[train].std(0) + 1e-6
    ym, ys = Y[train].mean(0), Y[train].std(0) + 1e-6
    Xt = (X[train] - xm) / xs
    Xe = (X[test] - xm) / xs
    Yt = (Y[train] - ym) / ys
    d = Xt.shape[1]
    W = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(d, dtype=np.float64), Xt.T @ Yt)
    pred = Xe @ W
    resid = ((Y[test] - ym) / ys - pred) ** 2
    var = ((Y[test] - ym) / ys) ** 2
    # R^2 per output, averaged; guard near-constant targets
    per = 1.0 - resid.mean(0) / np.maximum(var.mean(0), 1e-8)
    return float(np.clip(per, -1.0, 1.0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="VizdoomTakeCover-v1")
    ap.add_argument("--perception", choices=["pool", "mobilenet"], default="mobilenet")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--collect-episodes", type=int, default=40)
    ap.add_argument("--probe-steps", type=int, default=800)
    ap.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--window-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--axes", type=int, default=48)
    ap.add_argument("--cache-dir", type=Path, default=Path("docs/results/latents"))
    ap.add_argument("--out", type=Path, default=Path("docs/results/memory_probe.json"))
    args = ap.parse_args()

    env = make_doom_env(args.scenario, 4)
    dt = 4.0
    enc = build_encoder(args.perception, env, args.probe_steps, grid=args.grid)
    weights = ReservoirWeights.build(RES_CFG, enc.dim)
    U_probe = encode_pool(env, enc, args.probe_steps, seed=1)
    whitener = Whitener.fit(probe_states(weights, U_probe, dt), 48)
    whitener_lat = Whitener.fit(U_probe, 48)
    codebook = CodebookMemory.fit(ReservoirMemory(weights, dt), whitener, U_probe, k=256, seed=0).cb

    path = args.cache_dir / f"{args.scenario}_s{args.seed}_e{args.collect_episodes}_n1000.npz"
    with np.load(path, allow_pickle=False) as z:
        U, A, R, D = z["U"], z["A"], z["R"], z["D"]

    backends: dict[str, Callable[[], Any]] = build_backends(
        enc, weights, whitener, whitener_lat, dt, codebook, tuple(args.window_ks), args.axes
    )
    eps = episode_bounds(D)
    n_test = max(1, len(eps) // 5)
    test_eps = set(range(len(eps) - n_test, len(eps)))
    test_idx = np.concatenate([np.arange(s, e) for i, (s, e) in enumerate(eps) if i in test_eps])
    train_idx = np.concatenate(
        [np.arange(s, e) for i, (s, e) in enumerate(eps) if i not in test_eps]
    )

    # targets: whitened latents at each delay (the past we ask the state to remember)
    Z = ((U - whitener_lat.mean) @ whitener_lat.V) / whitener_lat.scale
    # a pair (state after frame t, latent at frame t-k) is only valid if no episode ended in
    # between -- otherwise the memory was reset and the "forgetting" is by construction.
    cum = np.concatenate([[0.0], np.cumsum(D)])  # cum[t] = number of resets in frames [0, t)
    rows = []
    for name, ctor in backends.items():
        mem = ctor()
        S, _A, _R, _S2, _D = roll_memory_over_latents(mem, U, A, R, D)
        # S[i] is the state after frame i; align with Z[t - delay]
        r2 = {}
        for k in args.delays:
            if k >= len(S):
                continue
            X = S[k:]
            Y = Z[: len(Z) - k]
            m = len(X)
            same_ep = (cum[k : k + m] - cum[:m]) == 0.0  # no reset inside the window
            tr = train_idx[train_idx < m]
            te = test_idx[test_idx < m]
            tr = tr[same_ep[tr]]
            te = te[same_ep[te]]
            if len(tr) < 50 or len(te) < 50:
                continue
            r2[k] = ridge_r2(X, Y, tr, te)
        row = {
            "backend": name,
            "scenario": args.scenario,
            "state_bytes": mem.state_nbytes(),
            "fixed_bytes": mem.fixed_nbytes(),
            "dim": int(mem.dim),
            "r2": {str(k): round(v, 4) for k, v in r2.items()},
            "r2_mean": round(float(np.mean(list(r2.values()))), 4) if r2 else 0.0,
            "r2_far": round(float(np.mean([v for k, v in r2.items() if k >= 8])), 4)
            if any(k >= 8 for k in r2)
            else 0.0,
        }
        rows.append(row)
        print(
            f"{name:16} B={row['state_bytes']:>5} "
            + " ".join(f"k{k}={v:+.2f}" for k, v in sorted(r2.items()))
            + f"   mean={row['r2_mean']:+.3f}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()
