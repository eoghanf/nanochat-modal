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
| Metric | Final training loss at step 1,000 (nats/token) |
| Data ordering | Each run uses a different shard seed (`--data-seed=1..8`) |

Each run sees the same model architecture and hyperparameters from `training_config.yaml`, but a different random shuffle of the training data shards. This ensures the 8 runs are genuinely independent samples rather than near-duplicate computations.

To run the benchmark:

```bash
uv run modal run modal_app.py::benchmark --group <group-name>
```

---

## Baseline Results (group: `bench-1000-v3`)

These 8 runs establish the baseline loss distribution for the current codebase (commit `ac225bf`) with default hyperparameters.

| Run | Final loss (nats, step 1000) |
|-----|------------------------------|
| bench-1000-v3-00 | 2.7406 |
| bench-1000-v3-01 | 2.7180 |
| bench-1000-v3-02 | 2.6866 |
| bench-1000-v3-03 | 2.6875 |
| bench-1000-v3-04 | 2.6762 |
| bench-1000-v3-05 | 2.7362 |
| bench-1000-v3-06 | 2.6861 |
| bench-1000-v3-07 | 2.6529 |

| Statistic | Value |
|-----------|-------|
| **Mean (Y)** | **2.6980** |
| Std | 0.0287 |
| Min | 2.6529 |
| Max | 2.7406 |
| Range | 0.0877 |

---

## Understanding the Variance

The standard deviation of **0.029 nats** is dominated by data shard ordering, not by weight initialisation noise or CUDA non-determinism (which contribute only ~0.0004 nats, as measured in `bench-1000-v2`). This is consistent with Karpathy's observation in the leaderboard that different data shuffle seeds produce substantial spread in final model quality, even for the same architecture and hyperparameters.

The practical consequence is that a single training run is an unreliable indicator of whether a change is genuinely better. The baseline mean **Y = 2.698 nats** is the number to beat.

---

## Claiming Progress: The Threshold

To claim that an architectural change represents genuine progress, we require a single benchmark run to achieve a final loss more than **2 standard deviations below the baseline mean**. Under the baseline distribution, a result this good would occur by chance only ~2.5% of the time — i.e. it is very unlikely to be a lucky seed.

> **A change demonstrates progress if a single (non-cherry-picked) benchmark run achieves a final loss below:**
>
> **Y − X = 2.698 − 0.06 = 2.64 nats**
>
> where **Y = 2.698** is the baseline mean and **X = 0.06** is the minimum improvement required (≈ 2σ ≈ 2.1% of Y).

To be concrete: run the benchmark once with your modified code. If the result is **≤ 2.64 nats**, the change is worth investigating further with a full training run. If it is above 2.64, it has not demonstrated improvement — even if it beats the baseline mean, it could simply be a fortunate shard ordering.

### Why a single run?

Running 8 comparison runs per candidate change (to compare means directly) would double the cost of evaluation. The single-run threshold with a 2σ requirement is a pragmatic compromise: it filters out most lucky seeds while keeping evaluation to one Modal invocation per candidate.

### Sensitivity

With this threshold, the benchmark can reliably detect improvements of **~2% in loss**. Changes smaller than this will require a full training run or more benchmark runs to confirm. Note that a 2% improvement in 1,000-step loss has historically translated to meaningful improvements in final val BPB and CORE score.

---

## Comparison: Benchmark vs Full Run

| | Benchmark | Full training run |
|-|-----------|-----------------|
| Hardware | 8× 1×H100 (parallel) | 1× 8×H100 |
| Wall-clock time | ~2.3 hours | ~1.67 hours |
| Estimated cost | ~$30 | ~$40–78 |
| Metric | 1,000-step loss | Final val BPB + CORE |
| Runs needed to claim progress | 1 (if loss < 2.64) | 1 (if CORE > 0.2565) |
| Run-to-run variance | ±0.029 nats | ±0.01–0.02 CORE |
