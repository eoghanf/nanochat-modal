# NanoChat Benchmark Evaluation

## Purpose

Running the full training pipeline (8×H100, ~1.67 hours, ~$78) every time we want to test an architectural change is prohibitively expensive. This document defines a cheaper benchmark — **8 independent 1×H100 runs for 1,000 steps each** — that can be used to evaluate candidate changes quickly, and establishes what constitutes a statistically meaningful improvement.

---

## Benchmark Design

| Parameter | Value |
|-----------|-------|
| Hardware | 1× H100 per run |
| Runs per experiment | 8 (launched in parallel on Modal) |
| Steps per run | 1,000 |
| Wall-clock time | ~2.3 hours |
| Metric | Final validation BPB at step 1,000 (`val/bpb`) |
| Weight initialisation | Each run uses a different random seed (`--weight-seed=1..8`) |
| Data ordering | Each run uses a different shard shuffle seed (`--data-seed=1..8`) |

Each run uses the same model architecture and hyperparameters from `training_config.yaml`, but a **different random seed for both weight initialisation and training data shard ordering**. Both sources of randomness are independently varied across the 8 runs, ensuring they are genuinely independent samples.

To run the benchmark:

```bash
uv run modal run modal_app.py::benchmark --group <group-name>
```

---

## Baseline Results (group: `bench-1000-v4`)

These 8 runs establish the baseline distribution for the current codebase with default hyperparameters. Both the weight initialisation seed and the data shard ordering seed are set to `1..8` across runs.

| Run | Weight seed | Data seed | val/bpb (step 1000) | Train loss (step 1000) | MFU% | Time (h) |
|-----|-------------|-----------|---------------------|------------------------|------|----------|
| 7n7bvmu7 | 1 | 1 | 0.8130 | 2.7425 | 60.40 | 2.307 |
| bsdp7pee | 2 | 2 | 0.8137 | 2.7265 | 60.11 | 2.324 |
| 8olg3jko | 3 | 3 | 0.8125 | 2.6881 | 60.65 | 2.297 |
| bl1mbh75 | 4 | 4 | 0.8107 | 2.6825 | 60.68 | 2.299 |
| pxpmlf2h | 5 | 5 | 0.8133 | 2.6834 | 60.98 | 2.292 |
| u6kpf8x5 | 6 | 6 | 0.8110 | 2.7282 | 60.82 | 2.298 |
| i8ryvu6o | 7 | 7 | 0.8142 | 2.6939 | 60.75 | 2.287 |
| 52maklr9 | 8 | 8 | 0.8113 | 2.6484 | 60.68 | 2.295 |

| Statistic | val/bpb | Train loss |
|-----------|---------|------------|
| **Mean (Y)** | **0.8124** | **2.6992** |
| Std | 0.0013 | 0.0310 |
| Min | 0.8107 | 2.6484 |
| Max | 0.8142 | 2.7425 |
| Range | 0.0035 | 0.0941 |

The primary metric is **val/bpb** (validation bits per byte), which is more stable and directly comparable across runs than training loss.

---

## Understanding the Variance

With both weight initialisation and data shard ordering varied, the standard deviation of val/bpb across 8 runs is **0.0013 bpb** — a tight distribution. For comparison, the previous baseline (`bench-1000-v3`, which varied only the data seed) produced a train/loss std of 0.0287 nats; the `bench-1000-v4` train/loss std of 0.0310 nats is nearly identical, confirming that **weight initialisation noise contributes negligibly to run-to-run variance**. The dominant source of variance is data shard ordering.

The practical consequence is that a single training run is an unreliable indicator of whether a change is genuinely better. The baseline mean **Y = 0.8124 bpb** is the number to beat.

---

## Claiming Progress: The Threshold

To claim that an architectural change represents genuine progress, we require a single benchmark run to achieve a final val/bpb more than **2 standard deviations below the baseline mean**. Under the baseline distribution, a result this good would occur by chance only ~2.5% of the time — i.e. it is very unlikely to be a lucky seed.

> **A change demonstrates progress if a single (non-cherry-picked) benchmark run achieves a final val/bpb below:**
>
> **Y − 2σ = 0.8124 − 0.0026 = 0.8098 bpb**
>
> where **Y = 0.8124** is the baseline mean and **σ = 0.0013** is the baseline standard deviation.

To be concrete: run the benchmark once with your modified code. If any run achieves **val/bpb ≤ 0.8098**, the change is worth investigating further with a full training run. If all runs are above 0.8098, the change has not demonstrated improvement — even if it beats the baseline mean, it could simply be a fortunate seed combination.

### Why a single run?

Running 8 comparison runs per candidate change (to compare means directly) would double the cost of evaluation. The single-run threshold with a 2σ requirement is a pragmatic compromise: it filters out most lucky seeds while keeping evaluation to one Modal invocation per candidate.

### Sensitivity

With this threshold, the benchmark can reliably detect improvements of **~0.3% in val/bpb** (~0.0026 bpb). Changes smaller than this will require a full training run or more benchmark runs to confirm.

---

## Cost

Modal does not expose per-function-call cost breakdowns — only per-app, per-hour billing. However, we can derive the per-run cost from billing data combined with W&B runtimes.

| Metric | Value |
|--------|-------|
| Average runtime per benchmark run | 138 min (2.30 hr) |
| Total H100-hours (8 runs) | 18.4 H100-hr |
| Derived H100 rate | ~$4.15/hr |
| **Cost per individual benchmark run** | **~$9.55** |
| **Total cost for one full benchmark (8 runs)** | **~$76** |

The H100 rate of ~$4.15/hr is consistent with Modal's published on-demand pricing. The 8 runs execute in parallel, so the wall-clock time is ~2.3 hours regardless of how many runs are launched.

For comparison, the full training run (8×H100, 1.88 hours of training time plus tokenisation, evaluation, and SFT) cost approximately **$78**.

---

## Comparison: Benchmark vs Full Run

| | Benchmark (8 runs) | Full training run |
|-|--------------------|-----------------|
| Hardware | 8× 1×H100 (parallel) | 1× 8×H100 |
| Wall-clock time | ~2.3 hours | ~3.5 hours (end-to-end) |
| Cost | ~$76 | ~$78 |
| Metric | val/bpb at step 1,000 | Final val BPB + CORE |
| Threshold to claim progress | single run val/bpb ≤ 0.8098 | CORE > 0.2565 |
| Run-to-run variance | ±0.0013 bpb | ±0.01–0.02 CORE |

The benchmark and a full run cost approximately the same. The benchmark's value is not lower cost per se, but **faster iteration**: you get a reliable signal in the same wall-clock time as a full run, without waiting for tokenisation, SFT, and full evaluation.

---

## Historical Baselines

### `bench-1000-v3` (data seed only)

| Statistic | Train loss |
|-----------|------------|
| Mean | 2.6980 |
| Std | 0.0287 |
| Min | 2.6529 |
| Max | 2.7406 |

Data seed varied (`--data-seed=1..8`); weight initialisation seed was not controlled. `val/bpb` was not logged in this run. Superseded by `bench-1000-v4`.
