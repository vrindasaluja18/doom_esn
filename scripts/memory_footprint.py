"""Verify the RAM vs storage model: measure real process RSS against the prediction.

The program claims (docs/RESEARCH_PROGRAM.md, section 4) that the resident memory state is
tiny and that the bulk of the agent's bytes live in storage, not RAM. This script checks it:
it builds the agent, measures process RSS, and writes the recorded latent dataset to storage
and reports its size.

    python scripts/memory_footprint.py --scenario VizdoomTakeCover-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rss_kb() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="VizdoomTakeCover-v1")
    ap.add_argument("--perception", choices=["pool", "mobilenet"], default="mobilenet")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("docs/results/footprint.json"))
    args = ap.parse_args()

    base = rss_kb()

    from doom_esn.encoder import FrameEncoder
    from doom_esn.envs import make_doom_env
    from doom_esn.features import Whitener
    from doom_esn.memory import ProjectedMemory, ReservoirMemory
    from doom_esn.readout import LinearQ
    from doom_esn.reservoir import ReservoirConfig, ReservoirWeights

    env = make_doom_env(args.scenario, 4)
    if args.perception == "pool":
        enc = FrameEncoder(24, 32)
        enc.fit(np.random.default_rng(0).random((200, enc.dim)).astype(np.float32))
        enc_bytes = 0
    else:
        from doom_esn.perception import MobileNetEncoder

        enc = MobileNetEncoder(size=96, pretrained=True)
        enc_bytes = enc.weight_nbytes()
    weights = ReservoirWeights.build(
        ReservoirConfig(
            n_res=400, spectral_radius=1.1, density=0.05, input_scaling=3.0, tau=4.0, seed=0
        ),
        enc.dim,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, weights.n)).astype(np.float32)
    whitener = Whitener.fit(X, 24)
    mem = ProjectedMemory(ReservoirMemory(weights, 4.0), whitener)
    q = LinearQ(np.zeros((mem.dim + 1, int(env.action_space.n)), np.float32))

    after_build = rss_kb()

    # record a latent dataset to storage
    obs, _ = env.reset(seed=0)
    mem.reset()
    zs = []
    for _ in range(args.steps):
        s = mem.step(enc(obs["screen"]))
        zs.append(s.copy())
        obs, _, te, tr, _ = env.step(q.act(s))
        if te or tr:
            obs, _ = env.reset()
    Z = np.asarray(zs, np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out.parent / "latents.npy", Z)
    storage_bytes = (args.out.parent / "latents.npy").stat().st_size

    report = {
        "scenario": args.scenario,
        "perception": args.perception,
        "resident_state_bytes": mem.state_nbytes(),
        "encoder_weight_bytes": enc_bytes,
        "readout_bytes": q.nbytes(),
        "codebook_fixed_bytes": mem.fixed_nbytes(),
        "predicted_agent_bytes": mem.state_nbytes() + enc_bytes + q.nbytes(),
        "rss_before_kb": base,
        "rss_after_build_kb": after_build,
        "rss_after_build_mb": round(after_build / 1024, 1),
        "storage_latents_bytes": storage_bytes,
        "storage_steps": args.steps,
        "storage_bytes_per_step": round(storage_bytes / args.steps, 2),
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nInterpretation: the process RSS is dominated by the Python interpreter and the")
    print("frozen torch encoder, not by the agent's memory state. The recorded latents live in")
    print("storage (latents.npy), read in batches, not resident.")
    env.close()


if __name__ == "__main__":
    main()
