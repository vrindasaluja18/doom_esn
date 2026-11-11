"""Aggregate benchmark JSON into a summary table and a return-vs-bytes figure."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("docs/results/memory"))
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    args = ap.parse_args()

    panels = [
        ("VizdoomTakeCover-v1", "VizdoomTakeCover-v1.json"),
        ("VizdoomDefendLine-v1", "VizdoomDefendLine-v1.json"),
        ("MyWayHome (window-k)", "VizdoomMyWayHome-v1_windowk.json"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15.5, 4.2))
    for ax, (scenario, fname) in zip(axes, panels, strict=False):
        f = args.results / fname
        if not f.exists():
            ax.set_visible(False)
            continue
        rows = json.load(f.open())
        by = defaultdict(list)
        for r in rows:
            by[r["backend"]].append(r)
        print(f"\n== {scenario} ==")
        print(
            f"{'backend':16} {'state_B':>8} {'fixed_B':>9} {'readout_B':>9} "
            f"{'return':>8} {'sem':>6} {'ms_p50':>7}"
        )
        xs, ys, es = [], [], []
        for b, rs in sorted(by.items(), key=lambda kv: kv[1][0]["state_bytes"]):
            r = float(np.mean([x["return_mean"] for x in rs]))
            e = float(np.mean([x["return_sem"] for x in rs]))
            s0 = rs[0]
            print(
                f"{b:16} {s0['state_bytes']:>8} {s0['fixed_bytes']:>9} "
                f"{s0['readout_bytes']:>9} {r:>8.1f} {e:>6.2f} {s0['ms_per_frame_p50']:>7.2f}"
            )
            xs.append(s0["state_bytes"])
            ys.append(r)
            es.append(e)
        ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3)
        for x, y, b in zip(xs, ys, sorted(by, key=lambda k: by[k][0]["state_bytes"]), strict=False):
            ax.annotate(b, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("resident state bytes (log)")
        ax.set_ylabel("greedy return")
        ax.set_title(scenario)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "return_vs_bytes.png", dpi=140)
    print(f"\nwrote {args.out / 'return_vs_bytes.png'}")


if __name__ == "__main__":
    main()
