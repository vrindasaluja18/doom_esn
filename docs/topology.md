# Clustering and topological diagnostics: findings

Status: **measured**. Labels follow `docs/MATHS.md` (`derived` / `measured` / `hypothesis`).
The two questions and their falsifiers were fixed before the experiments ran
(`docs/RESEARCH_PROGRAM.md`, sections 5 and 6).

## Verdict

- **Clustering (codebook): supported as a memory mechanism, with a caveat.**
  Storing a 1-byte codeword index instead of the latent keeps most of the return on the
  pretrained-perception arm (see the return-vs-bytes table). On the frozen pooled encoder the
  effect is *unreliable*: every arm sits near the random baseline within noise, in both
  directions across runs. The codebook is a rate-distortion device: fixed table cost buys a
  small resident state.
- **Morse / topological diagnostics: refuted as a precision allocator.**
  The trivial scalar-field feature (the Q margin) predicts which states need precision far
  better than a Morse/critical-point summary. Topology is dropped from the memory manager.

## Question 1: clustering codebook

- Hypothesis: a k-means codebook over the whitened latent lets the agent store an
  *assignment* (1–2 bytes) instead of the latent (d x 4 bytes) without losing much return.
- Falsifier: the codebook state (`esn400_vq256`, 1 byte) falls to the level of a
  memory-free baseline while the full state stays high.
- Method: `k=256` codebook fit on random-policy latents; back-end wrapped as
  `CodebookMemory`; iterated fitted-Q readout; held-out seeds. See
  `bench/memory_budget.py`.
- Result: measured return by resident state bytes (VizdoomTakeCover-v1, pretrained
  perception, 3 seeds) is in `docs/results/return_vs_bytes.png`. Headline: the codebook
  trades a ~1 MB fixed table for a 1-byte resident state and retains a large fraction of the
  full-state return on this scenario. On the pooled encoder the comparison is inconclusive:
  every back-end lands within noise of the random baseline (DefendLine random 7.18 ± 0.41),
  and the ordering is not stable across seeds, so no directional claim is made there.
- Verdict: supported as a mechanism on the pretrained arm; the trade is encoder-dependent and
  the fixed cost must be reported explicitly (it is not free).

## Question 2: Morse / persistent homology

- Hypothesis: a critical-point / Morse-signature feature of the readout-margin field
  predicts which states are "decision-critical" (need precision) better than a cheap
  spectral baseline.
- Falsifier: `max(AUC, 1-AUC)` for the topological features does not exceed the best
  baseline feature's.
- Baseline: `B1_margin_mag` (|Q1 − Q2|, no topology) and `B2_participation` (participation
  ratio of the local covariance).
- Method: `scripts/topology_probe.py`, 1000 held-out states, VizdoomTakeCover-v1,
  pretrained perception. Target = the greedy action flips under 4-bit storage of the state.
- Result (measured):

  | feature | AUC | power = max(AUC,1−AUC) | Spearman vs degradation |
  |---|---|---|---|
  | T1 criticality ‖∇margin‖ | 0.341 | 0.659 | 0.047 |
  | T2 Morse saddle (k-NN) | 0.324 | 0.676 | −0.036 |
  | B1 margin magnitude | 0.066 | **0.934** | −0.040 |
  | B2 participation ratio | 0.384 | 0.616 | −0.086 |

- Verdict: **refuted.** The baseline `B1_margin_mag` has power 0.934 versus 0.66–0.68 for
  the topological features. The readout margin is nearly a perfect ranker for
  precision-needing states; the topological summary adds nothing.

## What would change my mind

- A topological feature that beats `B1_margin_mag` (power > 0.95) on a *harder* scenario
  where the margin is ambiguous (e.g. a long-horizon memory task where two near-equal Q
  values correspond to different future returns).
- Evidence that persistent homology of the *trajectory* (not the readout field) predicts
  memory eviction — a different object than the one tested here.

## Design consequence

The memory manager should allocate precision by the **readout margin** (cheap, measured
power 0.93), not by a topological summary. This is the kind of result the program wanted:
a clean negative that removes complexity instead of adding it.

## Reproduce

```
python scripts/topology_probe.py --scenario VizdoomTakeCover-v1 --perception mobilenet --n 1000 --bits 4
python bench/memory_budget.py --scenario VizdoomTakeCover-v1 --perception mobilenet --seeds 3 --rounds 3
python bench/plot.py
```
