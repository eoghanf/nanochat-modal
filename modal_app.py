"""
Modal app — runs the full nanochat training pipeline on a single 8×H100 node,
and serves the chat web UI on a single A10G.

Usage:
    uv run modal run modal_app.py          # full training run
    uv run modal serve modal_app.py        # serve chat web UI

Hyperparameters come from training_config.yaml in the project root.
Checkpoints, datasets, and tokenizer artifacts are stored in a persistent
Modal Volume ("nanochat-cache") and survive across runs.
"""

import os
import subprocess
import sys

import modal

APP_NAME = "nanochat-training"
CACHE_PATH = "/cache/nanochat"

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Image — deps baked into lower layers, source code in the top layer.
# Modal caches each layer by content hash, so editing Python files only
# rebuilds the final add_local_dir layer, not the slow pip/apt layers.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "git", "build-essential", "libgomp1")
    # PyTorch from the CUDA 12.8 index
    .pip_install("torch==2.9.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    # Everything else
    .pip_install(
        "datasets>=4.0.0",
        "fastapi>=0.117.1",
        "filelock>=3.0",
        "kernels>=0.11.7",
        "matplotlib>=3.10.8",
        "psutil>=7.1.0",
        "python-dotenv>=1.2.1",
        "pyyaml>=6.0",
        "regex>=2025.9.1",
        "rustbpe>=0.1.0",
        "scipy>=1.15.3",
        "setuptools>=80.9.0",
        "tabulate>=0.9.0",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "transformers>=4.57.3",
        "uvicorn>=0.36.0",
        "wandb>=0.21.3",
        "zstandard>=0.25.0",
    )
    # Source code — top layer so code edits don't invalidate the dep cache
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=[".venv", ".git", "__pycache__", ".idea", ".pytest_cache", "*.pyc"],
    )
)

# ---------------------------------------------------------------------------
# Persistent volume — datasets, tokenizer, and checkpoints live here
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("nanochat-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], **kwargs) -> None:
    """Print and run a command, raising on non-zero exit."""
    print(f"\n+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100:8",
    volumes={CACHE_PATH: volume},
    timeout=4 * 3600,  # 4 hours — adjust if your run is longer
    secrets=[modal.Secret.from_dotenv()],
)
def train() -> None:
    os.chdir("/app")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "NANOCHAT_BASE_DIR": CACHE_PATH,
        "PYTHONPATH": "/app",
    }

    # Reset the markdown training report
    _run([sys.executable, "-m", "nanochat.report", "reset"], env=env)

    # Download 8 shards (~800 MB) needed to train the tokenizer
    _run([sys.executable, "-m", "nanochat.dataset", "-n", "8"], env=env)

    # Kick off downloading the remaining ~150 shards in the background
    dataset_proc = subprocess.Popen(
        [sys.executable, "-m", "nanochat.dataset", "-n", "170"],
        env=env,
    )

    # Train and evaluate the tokenizer
    _run([sys.executable, "-m", "scripts.tok_train"], env=env)
    _run([sys.executable, "-m", "scripts.tok_eval"], env=env)

    # Wait for the full dataset before starting pretraining
    print("\nWaiting for dataset download to complete...", flush=True)
    dataset_proc.wait()
    if dataset_proc.returncode != 0:
        raise RuntimeError("Dataset download failed")

    # -------------------------------------------------------------------
    # Pretraining — all hyperparameters come from training_config.yaml
    # -------------------------------------------------------------------
    _run(
        ["torchrun", "--standalone", "--nproc_per_node=8", "-m", "scripts.base_train"],
        env=env,
    )

    # Base-model evaluation (CORE metric, BPB, samples)
    _run(
        [
            "torchrun", "--standalone", "--nproc_per_node=8",
            "-m", "scripts.base_eval",
            "--", "--device-batch-size=16",
        ],
        env=env,
    )

    # -------------------------------------------------------------------
    # SFT — download identity conversations, then fine-tune
    # -------------------------------------------------------------------
    identity_path = os.path.join(CACHE_PATH, "identity_conversations.jsonl")
    if not os.path.exists(identity_path):
        import urllib.request
        print("\nDownloading identity conversations...", flush=True)
        urllib.request.urlretrieve(
            "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
            identity_path,
        )

    _run(
        [
            "torchrun", "--standalone", "--nproc_per_node=8",
            "-m", "scripts.chat_sft",
            "--", "--device-batch-size=16",
        ],
        env=env,
    )

    # SFT evaluation
    _run(
        [
            "torchrun", "--standalone", "--nproc_per_node=8",
            "-m", "scripts.chat_eval",
            "--", "-i", "sft",
        ],
        env=env,
    )

    # Generate the final markdown report
    _run([sys.executable, "-m", "nanochat.report", "generate"], env=env)

    # Flush all writes to the persistent volume
    volume.commit()


# ---------------------------------------------------------------------------
# Chat web UI — served on a single A10G (inference only, no training)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_PATH: volume},
    timeout=2 * 3600,  # 2 hours max session
    secrets=[modal.Secret.from_dotenv()],
)
@modal.web_server(port=8000, startup_timeout=120)
def serve() -> None:
    import subprocess, sys, os
    os.chdir("/app")
    env = {
        **os.environ,
        "NANOCHAT_BASE_DIR": CACHE_PATH,
        "PYTHONPATH": "/app",
    }
    subprocess.Popen(
        [sys.executable, "-m", "scripts.chat_web", "--num-gpus", "1", "--source", "sft"],
        env=env,
    )


# ---------------------------------------------------------------------------
# Benchmark — 8 parallel 1×H100 runs for 1000 steps each, grouped in W&B
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100:1",
    volumes={CACHE_PATH: volume},
    timeout=1 * 3600,
    secrets=[modal.Secret.from_dotenv()],
)
def bench_run(run_index: int, group: str) -> None:
    import os, sys
    os.chdir("/app")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "NANOCHAT_BASE_DIR": CACHE_PATH,
        "PYTHONPATH": "/app",
    }
    _run(
        [
            sys.executable, "-m", "scripts.base_train",
            f"--run={group}-{run_index:02d}",
            f"--wandb-group={group}",
            "--num-iterations=1000",
            "--core-metric-every=-1",
            "--sample-every=-1",
            "--save-every=-1",
        ],
        env=env,
    )


# ---------------------------------------------------------------------------
# Local entry-point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    train.remote()


@app.local_entrypoint()
def benchmark(group: str = "bench-1000", n: int = 8) -> None:
    """Run N parallel 1xH100 benchmark runs for 1000 steps each.

    Usage:
        uv run modal run modal_app.py::benchmark
        uv run modal run modal_app.py::benchmark --group bench-1000-v2 --n 4
    """
    list(bench_run.map(range(n), kwargs={"group": group}))
