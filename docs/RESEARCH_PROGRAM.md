# Research program: memory budgets for a CPU-only Doom agent

Status: plan for review. Scenario chosen: **not Basic** (see section 2).

## 0. Thesis

> **The maths is cheap; memory is precious.**

For a CPU-only agent the scarce resource is not FLOPs, it is the bytes that hold its
*memory of the past*. So the research question is not "which model is smarter" but:

> Given a hard per-agent memory budget **B bytes**, how should a sequential agent
> represent, store and page its history so that it still plays ViZDoom, and what is the
> smallest B that keeps it playing?

The deliverable is a measured **performance-vs-memory curve** plus the **RAM/storage
placement policy** that produces it.

## 1. Reference program structure

Modelled on GameNGen (Valevski et al., arXiv 2408.14837), which is two phases: (1) an RL
agent learns to play and its sessions are recorded, (2) a learned model predicts the next
frame. We keep that shape but make **memory** the object of study:

- **Phase 1 — Play & record.** A small CPU agent plays a scenario; every step is recorded
  as (encoded latent, action, reward, done) to **storage**.
- **Phase 2 — Memory study.** From the recording we learn memory representations under a
  byte budget and measure return vs bytes.

## 2. Scenario suite (verified runnable here, CPU, headless)

| scenario | actions | random return | episode len | why |
|---|---|---|---|---|
| `VizdoomDefendLine-v1` | 4 | +6 | ~122 | primary; dense reward, watchable aiming, needs target tracking |
| `VizdoomMyWayHome-v1` | 6 | -0.2 | ~525 | long-horizon, memory-stress (needs to remember the corridor) |
| `VizdoomDefendCenter-v1` | 4 | 0 | ~76 | secondary |
| `VizdoomDeadlyCorridor-v1` | 8 | -44 | ~43 | hard; optional |

Basic is excluded: it is memory-light (a 2-frame window ties a 400-unit reservoir), so it
cannot show a memory effect even in principle.

## 3. Pipeline

```
ViZDoom frame (240x320x3)
  -> PERCEPTION: frozen pretrained encoder  (HF weights allowed; no from-scratch training)
       -> latent z_t (d dims, e.g. 64)
  -> MEMORY MANAGER  <-- the research object, budget B bytes
       representation:  delay-window | ESN state | whitened latent | clustered code
       storage policy:  quantize / pack / evict / page
  -> READOUT (cheap, solved by least squares or fitted-Q, not evolved)
  -> action
  Record (z_t, a_t, r_t, done_t) -> storage
```

Phases, each ending in a shippable artifact:

- **P0 (done).** Fix the roast findings; make whitening reachable; correct the ANOVA.
  Branch `feat/memory-budget` @ 31bd98b.
- **P1 — Harness.** Frozen perception + byte-exact memory backends + preregistered eval.
  Deliverable: `bench/` + a first return-vs-bytes curve on DefendLine.
- **P2 — Clustered memory.** k-means/GMM codebook over latents; store assignments, not
  latents. Deliverable: bytes vs return vs reconstruction error.
- **P3 — Topological diagnostics.** Critical-point / persistent-homology summary of the
  latent landscape; test whether it predicts memory decisions better than effective rank.
  Deliverable: a falsifier report (win or honest negative).
- **P4 — Paper + demo.** RESULTS.md, recorded GIF, README.

## 4. RAM vs storage: what lives where, and why

Decision rule: **resident (RAM) = touched every control step and small; paged (storage) =
read in batches or rarely. The budgeted quantity B is the resident state.** We verify the
model by measuring process RSS against the prediction.

| object | where | why | rough size |
|---|---|---|---|
| perception encoder weights (frozen) | **RAM**, int8-quantized | used every frame; hot | ~0.25–1.25 MB |
| memory state `s_t` | **RAM** | updated every frame; **this is B** | B bytes |
| readout weights | **RAM** | tiny, used every frame | a few KB |
| cluster codebook / prototypes | **RAM** | small, read every step | k×d×dtype |
| encoded-latent dataset | **storage** | written once, read in batches for offline fitting | n×d×dtype |
| keyframes (for the mapping demo) | **storage**, paged | large, rarely needed | per-keyframe |
| metrics / checkpoints / logs | **storage** | append-only | — |

So the memory budget the curve is plotted against is the **resident state bytes**, with
encoder and readout bytes reported separately (we report total too, for honesty).

## 5. Where unsupervised clustering fits

A codebook turns a continuous latent into a **discrete memory slot**: store the cluster
index (1–2 bytes) instead of the latent (d×4 bytes). That is a direct rate–distortion
trade: bytes vs fidelity vs return. This is the "store the data properly" you asked for,
and it connects to your existing `faiss` k-means work (kmeans1d). We measure:
bytes/step, reconstruction error, and return, for k in a sweep.

## 6. Where Morse theory fits (and its falsifier)

Morse theory studies a smooth function through its critical points (grad f = 0) and their
index; sublevel sets change topology only when a critical value is crossed, and Morse
inequalities bound critical points by Betti numbers. Persistent homology is the robust,
discrete cousin that tracks which features survive across scales. Two honest slots:

- **(A) Encoder-output landscape.** Treat a scalar field over the latent (e.g. the
  readout margin or distance-to-decision) as a Morse function. Its critical points mark
  decision-critical regions (keep precision) vs flat regions (safe to quantize/evict).
- **(B) Memory-state trajectory.** The reservoir map `x_{t+1}=F(x_t,u_t)` has fixed
  points and saddles; Morse theory is the language for the topology of that state manifold
  and where trajectories get trapped. Extends the repo's `dynamics.py`.

**Falsifier (stated up front):** the topological summary must predict a memory decision
better than a cheap baseline (eigenvalue spectrum / effective rank / participation ratio).
If it does not, we report that honestly and drop it. It is a candidate, not a promise.

## 7. Paper structure (target artifact)

1. Abstract  2. Introduction (thesis)  3. Related work (GameNGen, reservoir computing,
Echo State, rate–distortion, topological data analysis)  4. Method (memory taxonomy,
RAM/storage policy)  5. Experiments (scenarios, budgets, protocol)  6. Results (curves,
tables)  7. What fails and why  8. Limitations  9. Reproducibility (one command).

## 8. Multi-agent roster

| agent | role | artifact |
|---|---|---|
| memory-backends | implement 5 backends + byte accounting | `src/doom_esn/memory/` |
| perception | frozen pretrained encoder | `src/doom_esn/perception.py` |
| benchmark | preregistered harness, exact permutation/Holm | `bench/` |
| topology-scout (deployed) | critical points / persistence; falsifier | `docs/topology.md` |
| demo | headless recorder -> GIF | `assets/` |
| writer | README + RESULTS.md | docs |

## 9. Open decisions

1. Budget definition: state-only (recommended) vs total bytes.
2. Perception source: which pretrained encoder (HF) — must run CPU-fast at 240×320.
3. Whether the readout is solved (least squares) or fitted-Q.
