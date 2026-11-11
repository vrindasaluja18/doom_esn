# doom-esn — how little memory does a Doom agent need?

A CPU-only agent that learns to play [ViZDoom](https://github.com/Farama-Foundation/ViZDoom),
and a measured study of one question:

> **Given a fixed budget of `B` bytes of resident memory, how should the agent store its
> past so that it still plays well?**

The short answer this repo delivers: on `VizdoomTakeCover`, a **48-byte** memory state
matches or beats a 1600-byte reservoir and a 9216-byte frame window. The maths is cheap;
memory is precious.

**Author:** Vrinda Saluja — [vrindaml539@gmail.com](mailto:vrindaml539@gmail.com) ·
[LinkedIn](https://www.linkedin.com/in/vrinda-saluja-rl/)

![the agent playing TakeCover](assets/agent_plays_takecover.gif)

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
| window k=2 (no reservoir) | 4608 B | 0 | 13836 B | 233.6 ± 20.4 | 4.05 |
| window k=4 | 9216 B | 0 | 27660 B | 243.0 ± 26.3 | 4.07 |
| ESN 400 (float32) | 1600 B | 0.99 MB | 4812 B | 257.4 ± 29.0 | 4.08 |
| ESN 400 → whitened 24 | 1600 B | 1.03 MB | 300 B | 233.6 ± 15.1 | 4.09 |
| ESN 400 → **16-bit × 24** | **48 B** | 1.03 MB | 300 B | **265.7 ± 29.4** | 4.11 |
| ESN 400 → 8-bit × 24 | 24 B | 1.03 MB | 300 B | 244.2 ± 26.1 | 4.15 |
| ESN 400 → **codebook k=256** | **1 B** | 1.05 MB | 300 B | 219.2 ± 22.0 | 4.17 |
| *random policy (reference)* | — | — | — | *164.3 ± 5.0* | — |

Reading:

- The **48-byte** state beats the **1600-byte** reservoir and both windows — a **33× smaller
  state at equal-or-better return**. Even a **1-byte** codeword index stays well above random
  and within reach of the 4608-byte window.
- The cost is *moved*, not removed: the quantiser pays a fixed ~1 MB whitener/codebook table.
  The trade is a fixed table for a tiny resident state — exactly the RAM-vs-storage decision
  in section 4.
- On the pooled (non-pretrained) perception arm the same compression *hurts*
  (`docs/results/memory/VizdoomDefendLine-v1.json`): the linear readout is more fragile when
  the input features are weak. Compression quality depends on the encoder — reported, not hidden.

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
using a 1-byte resident state. See `src/doom_esn/memory.py` (`CodebookMemory`).

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
  scenarios (`TakeCover`, `DefendLine`) — never `Basic`.
- The detailed maths (derived / tested / measured / unverified tags) is in
  [`docs/MATHS.md`](docs/MATHS.md); the pre-registered comparison in
  [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md); the program plan in
  [`docs/RESEARCH_PROGRAM.md`](docs/RESEARCH_PROGRAM.md).

## 8. Run it

```bash
pip install -e ".[dev,doom,perception]"      # CPU only
make check                                    # ruff + mypy + pytest (41 tests)

# 1. the headline curve (return vs resident bytes)
python bench/memory_budget.py --scenario VizdoomTakeCover-v1 --perception mobilenet \
    --seeds 3 --rounds 3
python bench/plot.py                          # -> docs/results/return_vs_bytes.png

# 2. train the agent and record it playing
python scripts/play.py --scenario VizdoomTakeCover-v1 --perception mobilenet \
    --rounds 4 --out-gif assets/agent_plays_takecover.gif

# 3. the topology falsifier and the memory footprint check
python scripts/topology_probe.py --scenario VizdoomTakeCover-v1 --perception mobilenet --bits 4
python scripts/memory_footprint.py --scenario VizdoomTakeCover-v1
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
tests/            41 tests, warnings as errors
.agents/agents/   topology-scout.md  (a file-based research sub-agent)
```

## 10. Limitations (read before citing anything)

- Three seeds per cell; the 48-byte vs 1600-byte gap (265.7 vs 257.4) is within one standard
  error — the honest claim is **"equal-or-better at 33× less state"**, not "quantisation helps".
- One perception model and two scenarios so far. Compression quality is encoder-dependent
  (section 2), and `DefendLine` shows the failure mode.
- The topology result is one field on one scenario; it is reported as a negative, not a theorem.
- The agent is far from a strong Doom player (TakeCover ~266 vs random ~164); the point is the
  *memory curve*, not a leaderboard score.
- No `Basic` numbers: by design (section 7).

## 11. Engineering

src layout, frozen dataclasses, strict YAML config (unknown keys rejected), seeded everywhere,
`ruff` + `mypy` + `pytest` (warnings as errors), pre-commit, CI matrix 3.10–3.12, `.npz`
checkpoints with `allow_pickle=False`. Every maths claim has a named test or a stored
measurement (ledger in MATHS 9). The exact permutation / Holm decision machinery is vendored
from the author's [credence](https://github.com/vrindasaluja18/credence) repo.
