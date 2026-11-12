# Analysis plan (written BEFORE the comparison was run)

Style follows `credence/PREREGISTRATION.md`; the statistics are credence's own
`rigorous_validation` module, vendored as `src/doom_esn/credence_stats.py`.

**Unit of replication:** one independent training run (seed). Metric: mean return of that run's
final readout over 30 held-out episodes (seeds disjoint from training). n = 5 runs per arm,
distinct seeds; failed runs would be recorded at the worst value, not dropped.

**Arms** (equal budget: population 12, 5 episodes per candidate, 10 generations = 600 training
episodes; diagonal CMA-ES on a linear argmax readout):

| arm | state fed to the readout | readout params | initial sigma |
|-----|--------------------------|---------------:|--------------:|
| `reservoir_raw` | 400-unit reservoir, no projection (the v0.1 design) | 1604 | 0.05 |
| `reservoir_white` | same reservoir, top-24 whitened axes | 100 | 0.5 |
| `window1_white` | current frame, 12 whitened axes (no memory) | 52 | 0.5 |
| `window2_white` | last 2 frames, 12 axes each, whitened jointly (24) | 100 | 0.5 |

**Planned comparisons (exactly three; two-sided exact permutation test, Holm over the three, alpha 0.05):**

1. `reservoir_white` vs `reservoir_raw`: did the data-driven fix help?
2. `window2_white` vs `window1_white`: does memory matter in this scenario at all?
3. `reservoir_white` vs `window2_white`: does the reservoir beat the classical delay null?

With n = 5 vs 5 the smallest attainable two-sided p is 2/252 = 0.0079 (0.0238 after Holm), so
the design *can* reject. Outcomes are REJECT / INCONCLUSIVE / UNDERPOWERED only: there is no
"no effect" outcome. Effect sizes are reported as difference of means with a percentile
bootstrap CI when n >= the module's minimum, otherwise raw points are shown.

**Not planned (exploratory, will be labelled as such):** anything else, including per-seed stories.

---

## Amendment 1 — collect-once / fit-all-arms-offline (2026-11-12)

The original design re-rolled the engine and the CNN for every `(arm, seed, round)`. Measured on
the CPU box this is ~4.1 ms/frame, so one scenario (13 arms × 3 seeds × 176 episodes × ~58 steps)
takes **30–45 minutes** and — because results were only written at the end — loses *everything*
if the process is killed (which happened repeatedly). It also gave each arm a *different* random
rollout, confounding the comparison.

The benchmark now does:

1. collect ONE latent dataset per seed under a uniform-random policy, cached to
   `docs/results/latents/` (the encoder is paid once, not once per arm);
2. replay every memory back-end over those *identical* latents and fit the linear fitted-Q
   readout in pure numpy — no engine, no CNN per arm;
3. evaluate greedily on held-out seeds (unchanged).

This is a *protocol change*, so it is recorded here rather than silently substituted. It makes
all arms see identical data (a strictly cleaner comparison), cuts a scenario to a few minutes,
and checkpoints each `(backend, seed)` row to `docs/results/memory/<scenario>.jsonl` so a run
resumes instead of restarting. The preregistered statistics (exact permutation, Holm) still
apply to the per-seed returns; the `random` and `best-fixed-action` nulls are now evaluated on
the *same* episodes as every arm, so comparisons are paired.

The preregistered CMA-ES arm list above is superseded by the solved linear readout (see
`docs/MATHS.md` §5: a linear argmax readout needs one ridge solve, not a search); the arms are
now the representation/quantisation sweep in `bench/memory_budget.py::build_backends`.

