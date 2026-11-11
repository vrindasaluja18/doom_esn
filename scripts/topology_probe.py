"""Falsifiable probe: does a topological/Morse feature beat a cheap baseline at deciding
which memory states need precision?

Question (docs/RESEARCH_PROGRAM.md, section 6). A memory manager must spend its budget where
it matters. Concretely: if we store a state coarsely (8-bit), the greedy action sometimes
flips. Those "decision-flip" states are the ones that need precision. We ask whether a
topological feature of the readout-margin field predicts a flip better than a cheap spectral
baseline.

Features, all computed on the whitened latent z (dim k):
  T1 criticality   : ||grad margin(z)|| by finite differences  (Morse: near-critical points)
  T2 morse_sign    : local max/min/saddle signature of margin over the k-NN graph
  B1 margin_mag    : |margin(z)|, margin = Q1 - Q2             (cheap, no topology)
  B2 participation : participation ratio of the local covariance (spectral baseline)

Target y: greedy action flips when z is stored as 8-bit and de-quantised.

Metric: ROC AUC (rank-based). Falsifier: T1/T2 must beat max(B1, B2) by a clear margin, or
the topological feature is dropped.

    python scripts/topology_probe.py --scenario VizdoomTakeCover-v1 --n 1500
"""

from __future__ import annotations

import argparse

import numpy as np

from doom_esn.envs import make_doom_env
from doom_esn.features import Whitener
from doom_esn.harness import (
    RES_CFG,
    build_encoder,
    encode_pool,
    probe_states,
    projected_reservoir,
)
from doom_esn.readout import LinearQ
from doom_esn.reservoir import ReservoirWeights


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC AUC with tie handling."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    npos, nneg = y.sum(), len(y) - y.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (no scipy.stats dependency)."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="VizdoomTakeCover-v1")
    ap.add_argument("--perception", choices=["pool", "mobilenet"], default="mobilenet")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--probe-steps", type=int, default=1200)
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--out", default="docs/results/topology.json")
    args = ap.parse_args()

    env = make_doom_env(args.scenario, 4)
    n_act = int(env.action_space.n)
    enc = build_encoder(args.perception, env, args.probe_steps)
    weights = ReservoirWeights.build(RES_CFG, enc.dim)

    U = encode_pool(env, enc, args.probe_steps, seed=1)
    whitener = Whitener.fit(probe_states(weights, U), 24)
    mem = projected_reservoir(weights, whitener)

    # fit Q on eps-greedy data
    q = LinearQ(np.zeros((mem.dim + 1, n_act), np.float32))
    rng = np.random.default_rng(0)
    S, A, R, S2, D = [], [], [], [], []
    for ep in range(30):
        obs, _ = env.reset(seed=500 + ep)
        mem.reset()
        s = mem.step(enc(obs["screen"]))
        for _ in range(1000):
            a = int(rng.integers(n_act)) if rng.random() < 0.5 else q.act(s)
            obs, r, te, tr, _ = env.step(a)
            s2 = mem.step(enc(obs["screen"]))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(float(te or tr))
            s = s2
            if te or tr:
                break
    q = LinearQ.fit(
        np.asarray(S, np.float32),
        np.asarray(A),
        np.asarray(R, np.float32),
        np.asarray(S2, np.float32),
        np.asarray(D, np.float32),
        n_act,
        iters=15,
    )

    # gather evaluation states
    Z = []
    for ep in range(args.n // 60 + 1):
        obs, _ = env.reset(seed=9000 + ep)
        mem.reset()
        s = mem.step(enc(obs["screen"]))
        for _ in range(1000):
            a = q.act(s)
            obs, r, te, tr, _ = env.step(a)
            s = mem.step(enc(obs["screen"]))
            Z.append(s)
            if te or tr or len(Z) >= args.n:
                break
        if len(Z) >= args.n:
            break
    Z = np.asarray(Z[: args.n], np.float32)

    # targets under coarse storage: a rare binary flip AND a continuous degradation
    def quant(z):
        zc = np.clip(z / 3.0, -1.0, 1.0)
        qi = np.round((zc + 1.0) * 0.5 * (2**args.bits - 1))
        return (qi / (2**args.bits - 1) * 2.0 - 1.0) * 3.0

    Q = np.stack([q.q(z) for z in Z])
    Qq = np.stack([q.q(quant(z)) for z in Z])
    flip = (Q.argmax(axis=1) != Qq.argmax(axis=1)).astype(int)
    degrade = np.abs(Q.max(axis=1) - Qq.max(axis=1))

    # features
    eps = 0.05
    margin = np.sort(Q, axis=1)[:, -1] - np.sort(Q, axis=1)[:, -2]
    grad = np.zeros(len(Z))
    for d in range(Z.shape[1]):
        zp, zm = Z.copy(), Z.copy()
        zp[:, d] += eps
        zm[:, d] -= eps
        mp = np.array([np.sort(q.q(z))[-1] - np.sort(q.q(z))[-2] for z in zp])
        mm = np.array([np.sort(q.q(z))[-1] - np.sort(q.q(z))[-2] for z in zm])
        grad += ((mp - mm) / (2 * eps)) ** 2
    T1 = np.sqrt(grad)

    # T2: discrete Morse sign over the k-NN graph (is the point a local extremum/saddle?)
    from scipy.spatial import cKDTree

    tree = cKDTree(Z)
    _, idx = tree.query(Z, k=args.knn)
    morse = np.zeros(len(Z))
    for i in range(len(Z)):
        nb = margin[idx[i]]
        hi = (nb > margin[i]).sum()
        lo = (nb < margin[i]).sum()
        morse[i] = 1.0 if (hi > 0 and lo > 0) else 0.0  # saddle-ish (index between)

    # baseline B2: participation ratio of the local covariance
    pr = np.zeros(len(Z))
    for i in range(len(Z)):
        W = Z[idx[i]] - Z[idx[i]].mean(0)
        ev = np.linalg.eigvalsh(W.T @ W / len(W))
        ev = ev[ev > 1e-12]
        pr[i] = (ev.sum() ** 2) / (ev**2).sum() if ev.size else 0.0

    feats = {
        "T1_criticality": T1,
        "T2_morse_saddle": morse,
        "B1_margin_mag": margin,
        "B2_participation": pr,
    }
    aucs = {k: auc(v, flip) for k, v in feats.items()}
    spears = {k: spearman(v, degrade) for k, v in feats.items()}
    # orientation-agnostic discriminative power: a feature that predicts "not flip" is just
    # as useful as one that predicts "flip", so compare max(AUC, 1-AUC).
    power = {k: max(a, 1.0 - a) for k, a in aucs.items()}
    print(
        f"states={len(Z)} bits={args.bits} flip_rate={flip.mean():.3f} "
        f"degrade_mean={degrade.mean():.4f}"
    )
    print("  feature                 AUC   power  Spearman(degrade)")
    for k in feats:
        print(f"  {k:20} {aucs[k]:>6.3f} {power[k]:>7.3f} {spears[k]:>16.3f}")
    best_top = max(power["T1_criticality"], power["T2_morse_saddle"])
    best_base = max(power["B1_margin_mag"], power["B2_participation"])
    top_spear = max(abs(spears["T1_criticality"]), abs(spears["T2_morse_saddle"]))
    base_spear = max(abs(spears["B1_margin_mag"]), abs(spears["B2_participation"]))
    verdict = (
        "topology WINS"
        if (best_top > best_base + 0.02 and top_spear > base_spear)
        else "topology does NOT beat baseline"
    )
    print(f"best_topology power={best_top:.3f} best_baseline power={best_base:.3f} -> {verdict}")

    import json

    with open(args.out, "w") as f:
        json.dump(
            {
                "n": len(Z),
                "bits": args.bits,
                "flip_rate": float(flip.mean()),
                "degrade_mean": float(degrade.mean()),
                "auc_flip": aucs,
                "power_flip": power,
                "spearman_degrade": spears,
                "verdict": verdict,
                "scenario": args.scenario,
            },
            f,
            indent=2,
        )
    print("wrote", args.out)
    env.close()


if __name__ == "__main__":
    main()
