<h3 align="center">Vrinda</h3>

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

`VizdoomTakeCover-v1`, frozen pretrained perception, 3 seeds, 20 held-out episodes each
(figure: [`docs/results/return_vs_bytes.png`](docs/results/return_vs_bytes.png)):

| back-end | resident state | fixed table | readout | return | ms/frame |
|---|---:|---:|---:|---:|---:|
| window k=2 (no reservoir) | 4608 B | 0 | 13836 B | 233.6 ± 20.4 | 4.00 |
| window k=4 | 9216 B | 0 | 27660 B | 243.0 ± 26.3 | 4.29 |
| ESN 400 (float32) | 1600 B | 0.99 MB | 4812 B | 257.4 ± 29.0 | 4.43 |
| ESN 400 → whitened 24 | 1600 B | 1.03 MB | 300 B | 233.6 ± 15.1 | 4.38 |
| ESN 400 → **16-bit × 24** | **48 B** | 1.03 MB | 300 B | **265.7 ± 29.4** | 4.42 |
| ESN 400 → 8-bit × 24 | 24 B | 1.03 MB | 300 B | 244.2 ± 26.1 | 4.42 |
| ESN 400 → **codebook k=256** | **1 B** | 1.05 MB | 300 B | 219.2 ± 22.0 | 4.20 |
| ESN 400 + tanh400 (fair null) | 1600 B | 1.63 MB | 4812 B | 279.2 ± 32.2 | 4.16 |
| window k=2 + tanh400 (fair null) | 4608 B | 1.84 MB | 4812 B | 265.9 ± 27.7 | 11.0 |
| *random policy (reference)* | — | — | — | *245.9 ± 19.7* | — |

Reading:

- The curve is **flat**: the 48-byte state (265.7) sits near the top, the 1600-byte reservoir
  (257.4) and the 4608-byte window (233.6) are within one standard error of it, and even the
  best arm overall (reservoir + tanh400, 279.2) is within one standard error of the 48-byte
  state. The honest claim is **"equal-or-better at 33× less resident state"**, *not*
  "quantisation helps".
- **The random baseline is strong.** A true uniform-random policy returns **245.9 ± 19.7** on
  TakeCover. So the 48-byte state does **not** clearly beat random, and the 1-byte codebook
  (219) is *below* random. *(An earlier draft reported "random = 164.3"; that number came from a
  bug in `scripts/play.py` — the baseline RNG was re-seeded on every step, so the policy was
  constant-action. Fixed; see section 7.)* TakeCover is close to solved by a couple of frames,
  so it is a **weak** memory test — which is why section 2.1 adds a scenario that needs memory.
- The cost is *moved*, not removed: the quantiser pays a fixed ~1 MB whitener/codebook table.
  The trade is a fixed table for a tiny resident state — exactly the RAM-vs-storage decision
  in section 4.
- **Fair null (section 2.2).** Applying the *same* fixed random nonlinearity (`tanh400`) to
  both memories moves the window up (233.6 → 265.9) and the reservoir up (257.4 → 279.2): part
  of the "reservoir vs window" gap is nonlinearity, not memory. The reservoir still leads, but
  the margin shrinks.
- **The pooled (non-pretrained) arm shows no reliable effect.** On `DefendLine` with the pooled
  `24x32` encoder every back-end sits near the random baseline (7.18 ± 0.41): the 1600-byte
  reservoir returns 6.9 ± 0.8 and the 24-byte quantised state 11.1 ± 1.0, but the per-seed
  spread (e.g. esn400 = [4.5, 4.9, 11.4]) is larger than any gap, so we do **not** claim
  "compression helps" *or* "compression hurts" there. The honest statement: with a weak encoder
  the linear readout has no signal to compress, on either side
  (`docs/results/memory/VizdoomDefendLine-v1.json`). This is why the pretrained encoder matters.

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
both isolates the memory *mechanism*:

| pair | linear | + tanh400 | Δ |
|---|---:|---:|---:|
| window k=2 | 233.6 ± 20.4 | 265.9 ± 27.7 | +32.3 |
| reservoir 400 | 257.4 ± 29.0 | 279.2 ± 32.2 | +21.8 |

The nonlinearity lifts both arms, so part of the raw "reservoir > window" gap is a nonlinearity
effect, not a memory effect. After the fair null the reservoir still edges the window
(279.2 vs 265.9), but the difference is inside one standard error. The honest reading: on this
scenario **memory representation and fixed nonlinearity are entangled**, and neither is clearly
decisive.

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
