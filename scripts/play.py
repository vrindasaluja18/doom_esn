"""Train a memory-budgeted agent and record it playing, with a random baseline.

Iterated fitted Q-iteration: collect transitions with the current policy (epsilon-greedy on
the fitted Q, starting from random), refit the linear readout, improve, repeat. Then evaluate
greedily on held-out seeds and save a GIF of one episode next to a random-policy episode.

    python scripts/play.py --scenario VizdoomDefendLine-v1 --rounds 3 --out-gif assets/agent.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from doom_esn.envs import make_doom_env
from doom_esn.features import Whitener
from doom_esn.harness import (
    RES_CFG,
    build_encoder,
    collect,
    encode_pool,
    evaluate,
    probe_states,
    projected_reservoir,
    rollout,
)
from doom_esn.readout import LinearQ
from doom_esn.reservoir import ReservoirWeights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="VizdoomDefendLine-v1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes-per-round", type=int, default=40)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--probe-steps", type=int, default=1200)
    ap.add_argument("--out-gif", type=Path, default=Path("assets/agent.gif"))
    ap.add_argument("--perception", choices=["pool", "mobilenet"], default="pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_doom_env(args.scenario, 4)
    n_act = int(env.action_space.n)
    enc = build_encoder(args.perception, env, args.probe_steps)
    weights = ReservoirWeights.build(RES_CFG, enc.dim)
    U = encode_pool(env, enc, args.probe_steps, seed=1)
    whitener = Whitener.fit(probe_states(weights, U), 24)

    def memory():
        return projected_reservoir(weights, whitener)

    q = LinearQ(np.zeros((25, n_act), np.float32))
    for rnd in range(args.rounds):
        eps = 1.0 if rnd == 0 else max(0.1, 0.5**rnd)
        S, A, R, S2, D = collect(
            env,
            enc,
            memory(),
            q.act,
            args.episodes_per_round,
            100_000 + 1000 * rnd + args.seed,
            args.max_steps,
            eps,
        )
        q = LinearQ.fit(S, A, R, S2, D, n_act, gamma=0.99, ridge=1e-3, iters=15)
        rets, _ = evaluate(env, enc, memory(), q.act, args.eval_episodes, 10**7, args.max_steps)
        print(
            f"round {rnd} eps={eps:.2f} transitions={len(S)} "
            f"heldout_return={rets.mean():.1f} +/- {rets.std() / np.sqrt(len(rets)):.1f}",
            flush=True,
        )

    agent_ret, frames, _ = rollout(env, enc, memory(), q.act, 10**7, args.max_steps, capture=True)
    rng = np.random.default_rng(args.seed)
    rand_ret, _ = evaluate(
        env,
        enc,
        memory(),
        lambda s: int(rng.integers(n_act)),  # one generator: not re-seeded each step
        args.eval_episodes,
        10**7,
        args.max_steps,
    )
    print(f"AGENT greedy return (recorded episode): {agent_ret:.1f}")
    print(
        f"RANDOM policy mean return over {args.eval_episodes} eps: "
        f"{rand_ret.mean():.1f} +/- {rand_ret.std() / np.sqrt(len(rand_ret)):.1f}"
    )

    args.out_gif.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames[::2]]  # every 2nd frame, smaller GIF
    imgs[0].save(args.out_gif, save_all=True, append_images=imgs[1:], duration=66, loop=0)
    print(f"wrote {args.out_gif} ({len(imgs)} frames)")
    env.close()


if __name__ == "__main__":
    main()
