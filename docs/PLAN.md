# Plan — doom_esn

This is the working plan. It is meant to be read top to bottom: problem first, then what
is already measured, then the next experiments, then the decisions we still have to make
together. Section 7 is the part that needs your input.

---

## 1. Problem statement

**The question the project answers:**

> Given a fixed budget of `B` bytes of resident memory, how should a reinforcement-learning
> agent store its past so that it still plays well?

Everything else in the repo is in service of that one sentence. The agent has a small fixed
budget for the *state that is the memory of the past* — the thing living in RAM every control
step. We do not care how big the frozen encoder or the readout is, as long as we report them.
We care how the past is compressed, and what that costs in return.

The four sub-questions, each with the falsifier we fixed **before** running it:

| # | Claim under test | Falsifier (the result that would kill it) |
|---|---|---|
| Q1 | A learned codebook (store an index, not the latent) beats a dense reservoir at equal bytes | the 1-byte codebook does not beat the full reservoir at equal bytes |
| Q2 | Morse-theoretic features of the state space predict which states need precision | Morse/persistence features do **not** out-predict the trivial readout margin |
| Q3 | The memory/state size, not the process RAM, is what matters | resident state and total RAM are not separable — i.e. RAM is dominated by the state |
| Q4 | The reservoir's fading memory beats a plain frame window at equal bytes | a window of the same byte budget matches or beats the reservoir |

Secondary question (framing, not a claim): what is a strong, **published** number for these
scenarios, so we can say plainly where this stands rather than only "beats random"?

### Why this is a good project (no buzzwords)

- It has a **single measurable objective** (return vs. bytes) and a **random and constant
  baseline** to anchor it. A reader can reproduce a number, not a vibe.
- The maths is **real and checkable**: a fading-memory reservoir, a rate–distortion codebook,
  a least-squares readout, and one topological hypothesis that we were willing to refute.
- One of the four questions is an **honest negative** (Q2). Negatives are what make the rest
  believable.
- It runs on a CPU. The whole point is that small memory is enough; a big-GPU result would
  undercut the thesis.

---

## 2. Where we are (measured so far)

Scenario **VizDoom TakeCover** (`VizdoomTakeCover-v1`, frame-skip 4, frozen MobileNetV3-Small
perception, fitted-Q readout, 3 seeds, 20 held-out greedy episodes each):

| back-end | resident state | fixed | return |
|---|---:|---:|---:|
| window k=4 | 9216 B | 0 | 243.0 ± 26.3 |
| ESN 400 float32 | 1600 B | ~0.99 MB | 257.4 ± 29.0 |
| **16-bit × 24** | **48 B** | ~1.03 MB | **265.7 ± 29.4** |
| 8-bit × 24 | 24 B | ~1.03 MB | 244.2 ± 26.1 |
| codebook k=256 | **1 B** | ~1.05 MB | 219.2 ± 22.0 |
| window k=2 | 4608 B | 0 | 233.6 ± 20.4 |

**Corrected baseline.** True uniform-random return is **245.9 ± 19.7**; the previously
reported "164.3" was a bug (a constant-action policy — see §6). This changes the honest
headline: the 48-byte state is the best arm, but it does **not** clearly beat uniform random
within one standard error, and the 1-byte codebook (219) is **below** random. The claim that
survives is about the **compression curve**, not "our agent crushes random".

---

## 3. Architecture

```
frame (screen) ─▶ frozen encoder ─▶ memory back-end ─▶ fitted-Q readout ─▶ action
                   MobileNetV3         (the B-byte state)   least squares
                   or pooled 24×32
```

- `src/doom_esn/harness.py` — the shared plumbing (encoder choice, random-policy latents,
  reservoir config, and the collect/train/evaluate/rollout loops). One definition, used by
  the benchmark, the play script and the probe.
- `src/doom_esn/memory.py` — the back-ends, each reporting its own `state_nbytes` and
  `fixed_nbytes`: `WindowMemory`, `ReservoirMemory`, `QuantizedLatent`, `CodebookMemory`,
  `ProjectedMemory`, and `RandomTanhFeatures` (the fair-null layer).
- `src/doom_esn/perception.py` — frozen pretrained encoder (weights downloaded once, never
  trained).
- `src/doom_esn/readout.py` — `LinearQ`, fitted Q-iteration by ridge least squares (no
  CMA-ES search noise).
- `bench/memory_budget.py` — the return-vs-bytes curve. `bench/plot.py` — the figure.
- `scripts/play.py` — train one agent, record a GIF next to the random baseline.
- `scripts/topology_probe.py` — the Q2 probe. `scripts/memory_footprint.py` — the Q3 probe.
- `docs/RESEARCH_PROGRAM.md` — the program. `docs/topology.md` — the Q2 result.

---

## 4. The next experiments

### P5 — the fair null (in progress)
**Hypothesis.** "Window vs reservoir" is confounded: the window feeds a *linear* readout,
the reservoir feeds random *nonlinear* features. Apply the same fixed random `tanh`
projection to both and the difference is the memory mechanism alone.
**Method.** Arms `window_k{2,4}_tanh400` and `esn400_tanh400` (same projection), already in
the benchmark.
**Falsifier.** If the window with the same nonlinearity matches the reservoir, the
"reservoir wins" story is a nonlinearity story, not a memory story.
**Deliverable.** Extra rows in `docs/results/memory/*.json` + a paragraph in the README.

### P6 — a genuine memory scenario (My Way Home)
**Hypothesis.** On a task where the reward depends on where the agent *was* (spatial memory),
a 1-frame window must fail and the fading reservoir must win — a memory *result*, not just a
memory *story*.
**Method.** Window-k sweep `k ∈ {1,2,4,8}` and the reservoir on `VizdoomMyWayHome-v1`; report
the return-vs-bytes curve for the scenario. (Sample Factory trains this scenario with
`--use_rnn=True`, so a memoryless window is expected to underperform — we test it.)
**Falsifier.** If window k=1 matches the reservoir, the scenario does not exercise memory.
**Deliverable.** `docs/results/memory/VizdoomMyWayHome-v1_windowk.json` + figure panel.
**Caveat to state in the writeup.** My Way Home's absolute score is dominated by whether the
agent finds the goal at all; we report it as a *relative* memory comparison, not as a
leaderboard entry.

### P7 — spend bits where they matter
**Hypothesis.** The readout margin (|Q margin|) predicts which states need precision (Q2
showed power 0.93). So allocate bits by margin: coarse when the margin is large, fine when
it is small — same return at fewer bytes than uniform quantization.
**Method.** A variable-precision `QuantizedLatent` (or a two-tier codebook) driven by the
margin; compare return-at-bytes against uniform 8/16-bit.
**Falsifier.** If margin-adaptive precision matches uniform precision at equal bytes, the
margin signal is not exploitable for compression.
**Deliverable.** New back-end + a curve overlay.

### P8 — make the headline statistical
**Method.** Raise seeds from 3 to 5–10 on TakeCover; report the 48-byte vs 1600-byte gap with
proper error bars.
**Falsifier.** If the gap stays within one standard error, the claim is "equal-or-better",
not "better" — and we say so.

---

## 5. Honesty rules (non-negotiable)

- Every number in the README is produced by a committed script and a committed JSON.
- A claim is tagged **derived** (maths), **tested** (unit test), **measured** (experiment), or
  **unverified** — never left ambiguous.
- If a result is inside one standard error of its comparison, the wording is
  "equal-or-better", not "better".
- Negative results are kept and written up (Q2 is one).

---

## 6. Risks and known issues

- **Corrected baseline (fixed this session).** `scripts/play.py`'s random baseline re-seeded
  its RNG on every step, so it always chose the same action. Fixed; the README numbers will be
  updated to the true uniform-random value (~246 on TakeCover).
- **Small effect sizes.** With 20 evaluation episodes the standard errors are ~±20–30 on
  TakeCover. P8 addresses this.
- **Perception ceiling.** TakeCover is close to solved by the frozen encoder + a 2-frame
  window; the memory signal is weak there. My Way Home (P6) is where memory should separate.
- **CPU-only budget.** Multi-agent Doom (duel/deathmatch) needs many GPUs and billions of
  steps and is out of scope; noted, not attempted.

---

## 7. Decisions to confirm together

1. **Headline framing.** Agree that the honest claim is now "a 48-byte state sits at the top
   of the compression curve, equal-or-better than 1600 bytes" rather than "beats random"?
2. **Scenario priority.** Push My Way Home (P6) first, or run P7 (adaptive precision) on
   TakeCover first?
3. **Seeds.** Is 5 seeds enough for the headline, or do you want 10?
4. **Repo vs. portfolio.** Do we keep this as the research repo and write a separate,
   shorter portfolio README, or make the one README serve both?
5. **Name/identity.** Keep the repo name `doom_esn`, or rename to match the framing?
