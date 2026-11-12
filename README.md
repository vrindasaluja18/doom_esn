<h1 align="center">doom-esn</h1>

<p align="center"><i>how little memory does a Doom agent need?</i></p>

<p align="center">
A CPU-only agent that learns to play <a href="https://github.com/Farama-Foundation/ViZDoom">ViZDoom</a>,
and a measured study of one question:
</p>

<p align="center">
<b>Given a fixed budget of <code>B</code> bytes of resident memory, how should the agent store its past so that it still plays well?</b>
</p>

<p align="center">
<a href="mailto:salujavrinda018@gmail.com">salujavrinda018@gmail.com</a> ·
<a href="https://www.linkedin.com/in/vrinda-saluja-rl/">LinkedIn</a>
</p>

<p align="center">
<img src="assets/agent_plays_takecover.gif" alt="the agent playing TakeCover" width="480">
</p>

The short answer this repo delivers: the **return-vs-bytes curve is almost flat**. On
`VizdoomTakeCover` a **48-byte** memory state matches a 1600-byte reservoir and a 9216-byte
frame window, and even a **1-byte** codeword stays in the same band. The honest caveat, stated
up front: on this scenario a **uniform-random policy is a strong baseline** (~246 return), so
the result is about the **compression curve**, not about crushing random. The maths is cheap;
memory is precious.

<details>
<summary><b>Table of contents</b> &nbsp;<sub>(click to expand)</sub></summary>

- [1. The question and why it is the right one](#1-the-question-and-why-it-is-the-right-one)
- [2. Headline result](#2-headline-result)
  - [2.1 A scenario that actually needs memory: My Way Home](#21-a-scenario-that-actually-needs-memory-my-way-home)
  - [2.2 The fair null: is it memory or just nonlinearity?](#22-the-fair-null-is-it-memory-or-just-nonlinearity)
- [3. Pipeline](#3-pipeline)
  - [3.1 The past has its own embedding — and it is the research object](#31-the-past-has-its-own-embedding--and-it-is-the-research-object)
  - [3.2 What is learned — and where the weights are](#32-what-is-learned--and-where-the-weights-are)
- [4. RAM vs storage — what lives where, and why](#4-ram-vs-storage--what-lives-where-and-why)
- [5. Where clustering fits (supported)](#5-where-clustering-fits-supported)
- [6. Where Morse theory fits (refuted, and that is a result)](#6-where-morse-theory-fits-refuted-and-that-is-a-result)
- [7. What the earlier runs established (and what changed)](#7-what-the-earlier-runs-established-and-what-changed)
- [8. Run it](#8-run-it)
- [9. Repository layout](#9-repository-layout)
- [10. Limitations (read before citing anything)](#10-limitations-read-before-citing-anything)
- [11. Engineering](#11-engineering)
- [12. Contact](#12-contact)

</details>

---

## 1. The question and why it is the right one

Most "agent plays Doom" projects optimise return with no bound on resources. That is the
wrong axis on a laptop. Here the scarce resource is **memory**: the bytes that hold the
agent's memory of the past and must sit in RAM on every control step. So the object of study
is the curve **return vs resident-state bytes**, plus an explicit decision about what lives
in RAM and what lives in storage.

Concretely, we ask three things and answer each with a measurement:

1. **Representation.** For a fixed budget `B`, which memory representation is best?
2. **Rate–distortion.** How far can the state be compressed (quantised / clustered) before
   return collapses?
3. **Allocation.** Which *states* deserve precision, and does a fancy topological criterion
   beat a cheap one?

## 2. Headline result

`VizdoomTakeCover-v1`, frozen pretrained perception, 3 seeds, 20 held-out episodes each,
paired against the random baseline on the *same* episodes
(figure: [`docs/results/return_vs_bytes.png`](docs/results/return_vs_bytes.png)):

| back-end | resident state | fixed table | readout | return | Δ vs random |
|---|---:|---:|---:|---:|---:|
| *random policy (reference)* | — | — | — | *244.2 ± 22.0* | — |
| *best fixed action* | — | — | — | *155.4 ± 5.4* | — |
| codebook k=256 on ESN | **1 B** | 8.5 MB | 588 B | 251.9 ± 12.6 | +7.6 |
| 8-bit × 24 on ESN | 48 B | 8.4 MB | 588 B | 237.1 ± 14.2 | −7.1 |
| window k=1 (memoryless) | 192 B | 1.0 MB | 588 B | 283.1 ± 17.6 | +38.9 |
| **window k=2** | **384 B** | 1.0 MB | 1.2 KB | **308.2 ± 21.0** | **+64.0** |
| window k=4 | 768 B | 1.0 MB | 2.3 KB | 294.0 ± 18.5 | +49.8 |
| window k=8 | 1536 B | 1.0 MB | 4.6 KB | 239.4 ± 12.9 | −4.8 |
| ESN 400 (float32) | 1600 B | 8.4 MB | 4812 B | 237.7 ± 12.0 | −6.5 |
| ESN 400 + tanh400 (fair null) | 1600 B | 9.0 MB | 4812 B | 238.5 ± 14.1 | −5.7 |

Reading:

- **A short window beats the reservoir.** The best arm is a **384-byte window of 2 whitened
  frames** (308.2); the 1600-byte reservoir (237.7) is *at* the random baseline. So on a
  frame-reactive scenario, recency is the whole signal and a recurrent state buys nothing —
  a clean, useful negative for the "reservoir memory" hypothesis.
- **But the win is not yet statistically clean.** The window's paired edge over random is
  +64.0 with a pooled standard error of 37.2 (≈1.7 SE), so the honest claim is
  **"equal-or-better at 4× less resident state"**, not "the window beats random".
- **The random baseline is strong.** A uniform-random policy returns 244.2 ± 22.0 on TakeCover,
  so a good fraction of every arm's return is the scenario being partly solvable by reflex.
  TakeCover is a **weak** memory test — which is why section 2.1 adds a scenario that needs memory.
- **Quantisation is cheap here.** The 1-byte codebook (251.9) and the 48-byte 8-bit state
  (237.1) are within one standard error of the float32 reservoir, so compressing the state
  ~1600× costs nothing measurable — but it also *buys* nothing, since the reservoir itself is
  at baseline. The cost is *moved*, not removed: the quantiser pays a fixed ~8 MB
  whitener/codebook table, exactly the RAM-vs-storage trade in section 4.
- **Fair null (section 2.2).** Applying the *same* fixed random nonlinearity (`tanh400`) to the
  reservoir does not help it (237.7 → 238.5), so the reservoir's null result is not an artifact
  of a linear readout.
- **The pooled (non-pretrained) arm shows a small real effect.** On `DefendLine` with the pooled
  `24x32` encoder the window k=1 (11.3 ± 0.9) and the 48-byte quantised ESN state (11.1 ± 1.0)
  both sit ~3 SE above random (7.7 ± 0.9), while the float32 reservoir (5.6 ± 0.6) is *below*
  random — a solved linear readout on a 1600-d random reservoir overfits on this little data
  (`docs/results/memory/VizdoomDefendLine-v1.json`). This is why the pretrained encoder and the
  small state matter.

> **Protocol note.** These numbers come from the **collect-once / fit-all-arms-offline**
> benchmark (`bench/memory_budget.py`): one random-policy latent dataset per seed is collected
> and cached, then *every* arm is fitted on those identical latents in pure numpy. Earlier drafts
> re-rolled the engine and the CNN for every `(arm, seed, round)`, which took 30–45 min per
> scenario, confounded arms by giving them different rollouts, and lost all results if the
> process was killed. See `AGENTS.md` for the protocol and section 7 for the history.

### 2.1 A scenario that actually needs memory: My Way Home

TakeCover rewards reacting to the current frame, so memory barely matters. `VizdoomMyWayHome-v1`
rewards *returning to where the agent started*, which is a genuine spatial-memory task. We sweep
the frame window `k ∈ {1,4,8}` (`docs/results/memory/VizdoomMyWayHome-v1_windowk.json`):

| back-end | resident state | return |
|---|---:|---:|
| window k=1 (memoryless) | 2304 B | -0.210 ± 0.000 |
| window k=4 | 9216 B | -0.068 ± 0.071 |
| window k=8 | 18432 B | -0.162 ± 0.046 |

Reading: **negative and flat.** With the frozen encoder and a linear readout the agent does not
learn the task (returns hover at ~0), so the scenario *fails to exercise* the memory effect we
hoped to isolate. This is reported as a limitation, not dressed up: the sample-factory-style
`--use_rnn` result does not reproduce with this readout/encoder, and a stronger readout or
perception would be needed before My Way Home can carry a memory claim.

### 2.2 The fair null: is it memory or just nonlinearity?

A confound: the window feeds a *linear* readout, the reservoir feeds random *nonlinear* features.
Adding the identical fixed random `tanh` projection (`RandomTanhFeatures`, 400 units, seed 0) to
the reservoir isolates the memory *mechanism* from the nonlinearity:

| pair | linear | + tanh400 | Δ |
|---|---:|---:|---:|
| reservoir 400 | 237.7 ± 12.0 | 238.5 ± 14.1 | +0.8 |

The nonlinearity does **not** lift the reservoir (Δ = +0.8, well inside one standard error), so
the reservoir's null result on TakeCover is *not* an artifact of a linear readout: adding a fixed
random nonlinearity does not recover the memory signal. The window k=2 arm (384 B, 308.2) still
leads the fair-null reservoir (1600 B, 238.5) by a wide margin, and at a quarter of the bytes.

### 2.3 Measuring memory directly: the delay-memory probe

Return is a *noisy, indirect* proxy for memory: on TakeCover the random baseline is ~244 and the
per-seed spread swamps the effect. So we also measure the memory itself, on the **same cached
latents**, with a linear-probe test: for each delay `k`, fit a ridge readout from the memory
state `s_t` to the whitened latent `k` frames earlier and report held-out `R²`. `R²(k)` is the
*delay-memory capacity* — a memoryless arm can only decode `k=1`; an arm that keeps more of the
past decodes larger `k`. It is a regression on a fixed dataset, so it has small error bars and
takes seconds per arm (`bench/memory_probe.py`, `docs/results/memory_probe.json`):

| back-end | state B | R²(k=1) | R²(k=2) | R²(k=4) | R²(k=8) | mean |
|---|---:|---:|---:|---:|---:|---:|
| window k=8 (16-bit) | 1536 | 1.00 | 1.00 | 1.00 | 0.23 | **0.605** |
| window k=8 (8-bit) | 384 | 1.00 | 1.00 | 1.00 | 0.23 | **0.605** |
| window k=4 | 768 | 1.00 | 1.00 | 0.30 | 0.06 | 0.455 |
| window k=2 | 384 | 1.00 | 0.31 | 0.15 | 0.05 | 0.297 |
| window k=1 | 192 | 0.31 | 0.21 | 0.11 | 0.06 | 0.135 |
| ESN → whitened 24 | 1600 | 0.41 | 0.29 | 0.16 | 0.06 | 0.181 |
| ESN → 8-bit × 24 | **48** | 0.41 | 0.29 | 0.16 | 0.06 | 0.180 |
| ESN → codebook k=256 | **1** | 0.23 | 0.17 | 0.09 | 0.02 | 0.098 |
| ESN 400 (float32) | 1600 | 0.41 | 0.23 | −0.01 | −0.20 | 0.011 |
| ESN 400 + tanh400 | 1600 | 0.41 | 0.20 | −0.02 | −0.19 | 0.007 |

This resolves the ambiguity the return curve could not:

- **The reservoir genuinely has *worse* memory than a short window.** The raw 1600-byte
  reservoir keeps only ~1 frame (R² falls to −0.20 by `k=8`); the 384-byte window keeps 2 frames
  perfectly. So the reservoir's null return is *not* a measurement artifact — its state simply
  does not hold the past. A random reservoir is a poor memory here; a delay line is a good one.
- **Projection is what makes the reservoir a usable memory.** Whitening the reservoir state
  (1600 B → 24 axes) *raises* its far-delay R² (mean 0.011 → 0.181), i.e. the useful memory is in
  a low-dimensional subspace a linear readout cannot find unaided. That is the real role of the
  whitener: not compression, but **extraction**.
- **Quantisation is nearly free, but codebooking is not.** 8-bit × 24 keeps the whitened
  reservoir's R² exactly (0.180 vs 0.181) at 48 B — 33× smaller. The 1-byte codebook loses about
  half the memory (0.098), so 1 byte *does* cost real capacity.
- **The two measurements agree.** The arm with the best measured memory (window k=8, mean R²
  0.605) is *not* the best return arm (window k=2, 308.2): keeping 8 frames helps a *probe* but
  adds readout noise for *control*. Memory capacity and task return are related but not identical,
  which is exactly why measuring both matters.

> The probe uses `docs/results/latents/` (regenerable), so run `bench/memory_budget.py` first.
> Pairs spanning an episode boundary are excluded, since the memory is reset there by
> construction.

## 3. Pipeline

```
ViZDoom frame (240x320x3)
  └─ perception        frozen pretrained MobileNetV3-Small, 96px, ImageNet weights
     └─ z_t           576-d latent, ~3.8 ms/frame, never trained here
        └─ memory      the research object, budget B bytes
           │           representation: window | ESN | whitened | quantised | codebook
           │           the whitener projects 576-d -> 24-d (fixed table)
           └─ readout  linear fitted-Q, solved by least squares (no CMA-ES search)
              └─ action
        record (z_t, a_t, r_t, done_t) -> storage
```

Two deliberate choices keep the experiment clean:

- **Perception is frozen and pretrained.** We are not studying "train a CNN"; we are studying
  memory. Using ImageNet weights removes perception as a confound and keeps the agent CPU-only.
- **The readout is solved, not searched.** A *linear* Q-function needs no evolution strategy;
  fitted Q-iteration is one ridge solve per iteration. This removes the search noise that the
  repo's earlier data showed was the real bottleneck (see section 7).

### 3.1 The past has its own embedding — and it is the research object

There are **two embeddings**, and only the second is what we study:

1. **Per-frame embedding (spatial).** `perception.py` maps one screen, `240x320x3`, to a
   `z_t` in `R^576`. This embeds a *single moment*, not the past.
2. **Past embedding (temporal).** The memory back-end turns the history `z_<=t` into a
   fixed-size state `s_t`. This *is* the embedding of the past, and it is the budgeted
   quantity `B`. Every back-end is a different way to embed the past:

   | back-end | how it embeds the past | resident state |
   |---|---|---|
   | `WindowMemory(k)` | stack the last `k` latents — a Takens delay embedding | `k*d` floats |
   | `ReservoirMemory` | random recurrence `x_t = f(W x_{t-1} + W_in z_t)` — a fading memory of the whole past | 400 floats |
   | `ProjectedMemory` / whitener | project the reservoir state onto 24 whitened axes | 400 floats (projection fixed) |
   | `QuantizedLatent` | the whitened past embedding, quantised to 8/16 bits per axis | 24 ints |
   | `CodebookMemory` | replace the whitened past embedding by its nearest codeword index | 1 index |

The memory state is **recurrent**: `step()` updates `s_t` in place, it is not recomputed from
scratch. In belief-state terms `s_t` is the *sufficient statistic of the past* — the belief
state — and the study asks how few bytes it needs. (`credence` asks the estimation half of the
same question: does a delay-embedded state recover the latent at all?)

### 3.2 What is learned — and where the weights are

Most of the "weights" in the pipeline are deliberately **not** learned, so the only thing that
varies across the headline table is the past embedding:

| component | learned? | how | size |
|---|---|---|---|
| perception (MobileNetV3-Small) | **no** — frozen ImageNet | downloaded | 3.7 MB (fixed) |
| reservoir matrix `W, W_in` | **no** — fixed random (seeded) | built once | ~0.99 MB (fixed) |
| whitener (`mu, V, sigma`) | fitted once on a random-policy probe, then frozen | PCA / whitening | ~1 MB (fixed) |
| codebook (`k x 24`) | fitted once on the probe, then frozen | k-means | (fixed) |
| **readout `W` (`LinearQ`)** | **yes — the only weights RL trains** | fitted Q-iteration (ridge least squares) | 300–27,660 B |
| memory state `s_t` | not weights — the *activations* | updated every step | `B` (budgeted) |

So "learning" here is a **closed-form solve, not a search**: `readout.py` fits
`Q(s,a) = r + gamma * max_a' Q(s',a')` with one ridge regression per action per iteration, and
`train_readout` alternates collect -> refit for a few rounds. No gradient descent and no CMA-ES
— which is exactly what removes the search noise that made the original run inconclusive
(section 7).

**The main point:** the study *isolates memory from learning*. Because perception is frozen and
the readout is solved in closed form, a difference in return can only come from *how the past
is embedded and how many bytes it costs* — never from optimization luck.

## 4. RAM vs storage — what lives where, and why

Decision rule: **resident (RAM) = touched on every control step and small; storage = read in
batches or rarely.** The budgeted quantity `B` is the resident state. Measured on this machine
(`docs/results/footprint.json`):

| object | where | why | size |
|---|---|---|---|
| perception weights (frozen) | **RAM** | used every frame | 3.7 MB (float32) |
| **memory state `s_t`** | **RAM** | updated every frame — **this is `B`** | **1600 B** |
| readout weights | **RAM** | tiny, every frame | 300 B |
| whitener / codebook table | **RAM** | fixed, read every step | ~1.0 MB |
| recorded latent dataset | **storage** | written once, read in batches | 96 B/step |
| checkpoints, metrics | **storage** | append-only | — |

Process RSS after building the agent is **375 MB**, dominated by the Python interpreter and
the torch runtime — *not* by the agent's memory state (1600 B). The model predicts the memory
placement; the measurement confirms it.

## 5. Where clustering fits (supported)

A codebook turns a continuous latent into a discrete **memory slot**: store the cluster index
(1 byte for `k ≤ 256`) instead of the latent (`d×4` bytes). This is a rate–distortion device,
and it is measured, not assumed: the `k=256` codebook keeps 219/257 of the full-state return
using a 1-byte resident state — *but* 219 is **below** the uniform-random baseline (~246), so
the honest verdict is "the codebook preserves most of the *full-state* return", not "it plays
well". See `src/doom_esn/memory.py` (`CodebookMemory`).

## 6. Where Morse theory fits (refuted, and that is a result)

We tested whether a **Morse/topological summary** of the readout-margin field predicts which
states need precision better than a cheap baseline. The falsifier was fixed in advance; the
answer is no (`docs/topology.md`, `scripts/topology_probe.py`, 1000 held-out states):

| feature | AUC | power = max(AUC, 1−AUC) |
|---|---:|---:|
| T1 criticality ‖∇ margin‖ (Morse) | 0.341 | 0.659 |
| T2 Morse saddle over k-NN graph | 0.324 | 0.676 |
| **B1 margin magnitude** (no topology) | 0.066 | **0.934** |
| B2 participation ratio | 0.384 | 0.616 |

The trivial **readout margin** is a near-perfect ranker (power 0.93); the topological summary
adds nothing. Design consequence: allocate precision by the margin, not by topology. A clean
negative that removes complexity is a valid deliverable, so it is reported as one.

## 7. What the earlier runs established (and what changed)

The original v0.1 design evolved a linear policy with CMA-ES on `VizdoomBasic` and found
nothing statistically distinguishable. The honest reading of that data:

- **The bottleneck was search noise.** Initial candidates differed by ~7 return points against
  ~154 of episode noise (ranking reliability 0.006). Fixing that (whitening + solved readout)
  is why this repo now gets a signal.
- **`Basic` cannot show a memory effect**: a 2-frame window ties a 400-unit reservoir there, so
  the scenario is memory-light by construction. This study therefore uses memory-requiring
  scenarios (`TakeCover`, `DefendLine`, `MyWayHome`) — never `Basic`.
- **The corrected random baseline.** The originally committed `scripts/play.py` re-seeded its
  RNG every step, so its "random" policy was a single constant action (return 164.3). Fixed to
  one generator per evaluation; the true uniform-random return is **245.9 ± 19.7**. This is the
  single most important correction in the repo: it moves the honest headline from "beats random"
  to "flat compression curve" (`docs/PLAN.md` §2 and §6).
- The detailed maths (derived / tested / measured / unverified tags) is in
  [`docs/MATHS.md`](docs/MATHS.md); the pre-registered comparison in
  [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md); the program plan in
  [`docs/RESEARCH_PROGRAM.md`](docs/RESEARCH_PROGRAM.md).

## 8. Run it

```bash
pip install -e ".[dev,doom,perception,bench]"   # CPU only (bench = matplotlib + ripser)
make check                                       # ruff + mypy + pytest (43 tests)

# 1. the headline curve (return vs resident bytes) — the full back-end set, incl. fair-null
python bench/memory_budget.py --scenario VizdoomTakeCover-v1 --perception mobilenet \
    --seeds 3 --rounds 3 --collect-episodes 40 --eval-episodes 20
python bench/plot.py                             # -> docs/results/return_vs_bytes.png

# 1b. the memory-stress scenario (window-k sweep) and the pooled (non-pretrained) arm
python bench/memory_budget.py --scenario VizdoomMyWayHome-v1 --perception mobilenet \
    --window-ks 1 4 8 --only window --seeds 2 --rounds 2 --tag _windowk
python bench/memory_budget.py --scenario VizdoomDefendLine-v1 --perception pool \
    --seeds 3 --rounds 3

# 2. train the agent and record it playing
python scripts/play.py --scenario VizdoomTakeCover-v1 --perception mobilenet \
    --rounds 4 --episodes-per-round 40 --eval-episodes 30 \
    --out-gif assets/agent_plays_takecover.gif

# 3. the topology falsifier and the memory footprint check
python scripts/topology_probe.py --scenario VizdoomTakeCover-v1 --perception mobilenet --bits 4
python scripts/memory_footprint.py --scenario VizdoomTakeCover-v1 --perception mobilenet
```

Everything runs headless on CPU. The pretrained perception arm downloads MobileNetV3-Small
weights from PyTorch once (~10 MB) and never trains them.

## 9. Repository layout

```
src/doom_esn/
  perception.py   frozen pretrained encoder (MobileNetV3-Small)
  memory.py       memory back-ends with exact byte accounting  <- the research object
  readout.py      linear fitted-Q (least squares)
  encoder.py      the original pooled 24x32 encoder (the "pool" arm)
  reservoir.py    echo-state reservoir (fixed weights)
  features.py     Whitener (PCA / whitening projection)
  calibrate.py    data-only reservoir health criteria
  dynamics.py     fixed points, saddles, stability disc
  pipeline.py, cli.py, config.py, envs.py, es.py, policy.py
bench/            memory_budget.py, plot.py
scripts/          play.py, topology_probe.py, memory_footprint.py, compare.py, data_shape.py
docs/             MATHS.md, ANALYSIS_PLAN.md, RESEARCH_PROGRAM.md, topology.md, results/
tests/            43 tests, warnings as errors
.agents/agents/   topology-scout.md  (a file-based research sub-agent)
```

## 10. Limitations (read before citing anything)

- Three seeds per cell; the 48-byte vs 1600-byte gap (265.7 vs 257.4) is within one standard
  error — the honest claim is **"equal-or-better at 33× less state"**, not "quantisation helps".
- One perception model and three scenarios so far. The pooled (non-pretrained) encoder is
  **inconclusive** on `DefendLine` — every arm sits at the random baseline within noise
  (section 2) — so the only place a compression signal appears is the pretrained arm.
- The topology result is one field on one scenario; it is reported as a negative, not a theorem.
- **The agent does not beat uniform random on TakeCover** (agent ~266, random ~246, inside one
  standard error). TakeCover is nearly solved by a 2-frame window, so it is a weak memory test;
  the point is the *memory curve*, not a leaderboard score.
- **My Way Home did not work**: the agent fails to learn it (returns ~0), so the scenario meant
  to isolate memory does not yet do so (section 2.1).
- No `Basic` numbers: by design (section 7).

## 11. Engineering

src layout, frozen dataclasses, strict YAML config (unknown keys rejected), seeded everywhere,
`ruff` + `mypy` + `pytest` (warnings as errors), pre-commit, CI matrix 3.10–3.12, `.npz`
checkpoints with `allow_pickle=False`. Every maths claim has a named test or a stored
measurement (ledger in MATHS 9). The exact permutation / Holm decision machinery is vendored
from the author's [credence](https://github.com/vrindasaluja18/credence) repo.

## 12. Contact

Questions, corrections, or a pull request you want reviewed? Reach me at
[salujavrinda018@gmail.com](mailto:salujavrinda018@gmail.com) or on
[LinkedIn](https://www.linkedin.com/in/vrinda-saluja-rl/).
