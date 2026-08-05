# Environment setup

Notes for reproducing the environment used in this replication. The awkward parts are all about
keeping **torch, vLLM and the CUDA toolkit on the same version**; vLLM links against a specific
torch ABI, so a mismatch surfaces as an import-time symbol error rather than anything readable.

## Recorded versions

The numbers below come from the W&B metadata of the generation runs.

| Component | Version |
| --- | --- |
| OS | Linux 5.14.0 (RHEL 9.6), glibc 2.34 |
| Python | 3.10.18 (CPython) |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| vllm | 0.8.5 |
| xformers | 0.0.29.post2 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| datasets | 4.8.5 |
| huggingface_hub | 0.36.2 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| wandb | 0.27.0 |
| nvidia-cuda-runtime-cu12 | 12.4.127 |
| GPU | 4 × NVIDIA L40S (48 GB, Ada) |
| CPU / RAM | 64 cores / 503 GiB |

## Install with uv

```bash
uv sync                      # resolves everything in pyproject.toml / uv.lock

uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

uv pip install vllm==0.8.5 \
  --extra-index-url https://wheels.vllm.ai/0.8.5/cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
```

`--index-strategy unsafe-best-match` is required: without it uv resolves every package from the
first index that has *any* matching version, and the CUDA-suffixed wheels never win.

The remaining dependencies come from `pyproject.toml`:

```bash
uv add transformers==4.57.6 datasets huggingface_hub wandb jupyterlab ipykernel google-genai
```

**Check the pins each time you change one.** Bumping vLLM means bumping torch to whatever that vLLM
release was built against, and pointing `--extra-index-url` at the matching
`https://wheels.vllm.ai/<version>/cu124`.

## CUDA on the cluster

The cluster exposes CUDA as environment modules. The wheels above are cu124 builds; the 12.8 module
is backwards compatible and is what the SLURM job loads.

```bash
module unload cuda/12.4
module load cuda/12.8
nvcc --version        # verify
```

Environment variables expected by the vLLM build:

```bash
export CUDA_HOME=/apps/pkg/cuda/12.8
export CUDA_ROOT=/apps/pkg/cuda/12.8
export LD_LIBRARY_PATH=/apps/pkg/cuda/12.8/lib64
export TORCH_CUDA_ARCH_LIST=8.0
export VLLM_CUDA_COMPATIBILITY_PATH=/apps/pkg/cuda/12.8/lib64
```

Sanity check from Python:

```python
import torch
print("CUDA available :", torch.cuda.is_available())
print("CUDA version   :", torch.version.cuda)
print("Device 0       :", torch.cuda.get_device_name(0))
```

## Credentials

The pipeline reads these from a `.env` at the repository root (loaded with `python-dotenv`):

| Variable | Used for |
| --- | --- |
| `GEMINI_API_KEY` | Gemini 2.5 Pro claim-verification oracle |
| `WANDB_API_KEY` | Run logging |
| `HF_TOKEN` | LongFact++ dataset and Qwen 3 8B weights |

`.env` is gitignored. Never commit keys.

## Running on SLURM

`scripts/batch_test.slurm` requests one node with 4 × L40S, 200 GB RAM and a 48-hour walltime, then
runs `scripts/generate_model_response.py` from the project virtualenv. Both files contain absolute
paths under `/users/<user>/reinforcement-learning-from-feature-rewards-replication/` — update them
for your own filesystem before submitting.

```bash
sbatch scripts/batch_test.slurm
```

`scripts/batch_size_sweep.py` sweeps batch sizes from 128 to 4096 and logs throughput to W&B; use it
to pick `BATCH_SIZE` in the generation script for a given GPU count.
