# nanochat-modal

> **Original author: [Andrej Karpathy](https://github.com/karpathy/nanochat)**
> This is a lightly refactored fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) (commit [`a825e63`](https://github.com/karpathy/nanochat/commit/a825e63), autoresearch round 2).
> The core training code, model architecture, tokenizer, and evaluation harness are entirely Karpathy's work.
> This fork adds a Modal-based launcher (`modal_app.py`) and a central hyperparameter file (`training_config.yaml`) to make it easy to run the full pipeline on cloud GPUs without managing your own node.

---

## What this fork adds

| File | Purpose |
|------|---------|
| `modal_app.py` | Runs the full pipeline on a Modal 8×H100 node; also serves the chat UI on a single A10G |
| `training_config.yaml` | Central hyperparameter file — edit here instead of hunting through CLI flags |
| `TRAINING_REPORT.md` | Report from our training run, with results compared to Karpathy's leaderboard |

Everything else is unchanged from upstream.

## Quickstart (Modal)

```bash
# Install dependencies
uv sync --extra gpu

# Configure Modal and W&B (one-time)
uv run modal setup        # authenticate with Modal
# Add WANDB_API_KEY to .env

# Run the full pipeline (tokeniser → pretraining → SFT → eval)
uv run modal run modal_app.py

# Serve the chat web UI on a single A10G
uv run modal serve modal_app.py
```

Hyperparameters are in `training_config.yaml`. CLI arguments override YAML values if passed explicitly.

## Results

See [TRAINING_REPORT.md](TRAINING_REPORT.md) for full results. Summary:

| Metric | Our run | Karpathy Run 6 (same codebase) | GPT-2 (2019) |
|--------|---------|-------------------------------|--------------|
| Val BPB | **0.7180** | 0.7180 | — |
| CORE score | **0.2736** | 0.2626 (avg of 5 runs) | 0.2565 |
| Training time | ~1.65 hr | 1.65 hr | 168 hr |
| Cost | ~$78 (on-demand) | ~$48 (spot) | ~$43,000 |

---

## Original nanochat description

nanochat is the simplest experimental harness for training LLMs from scratch. It covers the full stack: tokenization, pretraining, supervised fine-tuning, evaluation, inference, and a ChatGPT-style web UI. The goal is to train a GPT-2 capability model (~$43,000 in 2019) for well under $100 on modern hardware.

The single complexity dial is `--depth` (number of transformer layers). All other hyperparameters — width, heads, learning rates, training horizon, weight decay — are derived automatically for compute-optimal training.

For more detail, see the [original repo](https://github.com/karpathy/nanochat) and Karpathy's writeup: [Beating GPT-2 for <<$100](https://github.com/karpathy/nanochat/discussions/481).

## Time-to-GPT-2 Leaderboard (upstream)

| # | Time | Val BPB | CORE | Description | Date | Commit |
|---|------|---------|------|-------------|------|--------|
| 0 | 168 hr | — | 0.2565 | Original OpenAI GPT-2 | 2019 | — |
| 1 | 3.04 hr | 0.74833 | 0.2585 | d24 baseline | Jan 29 2026 | `348fbb3` |
| 2 | 2.91 hr | 0.74504 | 0.2578 | d26 + fp8 | Feb 2 2026 | `a67eba3` |
| 3 | 2.76 hr | 0.74645 | 0.2602 | 1M token batch | Feb 5 2026 | `2c062aa` |
| 4 | 2.02 hr | 0.71854 | 0.2571 | ClimbMix dataset | Mar 4 2026 | `324e69c` |
| 5 | 1.80 hr | 0.71808 | 0.2690 | autoresearch round 1 | Mar 9 2026 | `6ed7d1d` |
| **6** | **1.65 hr** | **0.71800** | **0.2626** | **autoresearch round 2** | **Mar 14 2026** | **`a825e63`** |

This fork is based on entry 6 (the current state of the art as of March 2026).

## License

MIT — see [LICENSE](LICENSE). Original work by Andrej Karpathy.
