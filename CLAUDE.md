# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a replication of **Reinforcement Learning from Feature Rewards (RLFR)** — a pipeline for reducing hallucination in LLMs by training a student model with probe-guided RL rewards derived from factual correctness signals. The target model is `Qwen/Qwen3-8B` trained on the `obalcells/longfact-augmented-prompts` dataset.

## Environment Setup

This project uses `uv` for dependency management with a local `.venv`.

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv sync

# Install PyTorch with CUDA 12.4 support (must be done separately)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM with CUDA 12.4
uv pip install vllm==0.8.5 --extra-index-url https://wheels.vllm.ai/0.8.5/cu124 --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
```

Python version is pinned in `.python-version`.

## Running Tests

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/unit/probes/test_localization.py

# Run a specific test
pytest tests/unit/probes/test_localization.py::TestLocalizationProbe::test_output_shape
```

## Running the Pipeline Stages

```bash
# Stage 1 — Generate student model responses (uses vLLM, requires GPU)
python src/compliation/response_generator.py

# The scripts/ directory (per layout) will hold top-level runners:
python scripts/run_compilation.py
python scripts/run_probe_training.py
python scripts/run_training.py
python scripts/run_evaluation.py
```

Configs are loaded from `configs/base.yaml` merged with stage-specific configs (e.g., `configs/compilation/response_generation.yaml`). The `configs/test_run.yaml` has reduced scale values for local development.

## Architecture

The pipeline has three major phases:

### Phase 1: Dataset Compilation (`src/compliation/`)

A sequential pipeline that converts raw questions into labeled training records:

1. **ResponseGenerator** — Uses the student model (via vLLM) to generate `N_COMPLETIONS` candidate responses per question. Writes JSONL to `outputs/`.
2. **ClaimExtractor** — Calls the teacher model with structured output to decompose each response into atomic factual claims and named entities.
3. **ClaimValidator** — Queries an oracle (web search / RAG) per claim, then the teacher model assigns a `ClaimLabel` (SUPPORTED / REFUTED / UNCERTAIN) and confidence score.
4. **ResponseCorrector** — Prompts the student model to emit a self-corrected response with inline `<retract>/<correct>/<confirm>/<uncertain>` tag tokens.
5. **ResponseEvaluator** — Teacher model assigns a token-level quality score (INCORRECT / PARTIALLY_CORRECT / CORRECT).
6. **CompilationPipeline** — Orchestrates the above chain per question; checkpoints every 100 records.

> **Note:** `src/compliation/` is misspelled (missing an `i`). The canonical layout (in the `layout` file) uses `src/compilation/`. Most stub files are still empty — implementation is in progress.

### Phase 2: Probe Training (`src/probes/`)

Three linear probes trained on transformer hidden states extracted by `ActivationExtractor` (PyTorch forward hooks):

| Probe | Input | Task |
|---|---|---|
| `LocalizationProbe` | Question activations, single layer | Binary: will the model hallucinate? |
| `ValidationProbe` | Response activations, multiple layers | Multi-class: claim-level factual support |
| `InterventionProbe` | Correction activations, multiple layers | Multi-class: quality after self-correction |

`ProbeTrainer` uses cross-entropy loss with early stopping; checkpoints go to `outputs/checkpoints/probes/`.

### Phase 3: RL Fine-Tuning (`src/training/`)

The student model is fine-tuned using probe-weighted composite rewards:

```
r = 0.2·f(localization) + 0.5·f(validation) + 0.3·f(intervention)
  + 0.5 * |correct_retractions|
  - 2.0 * |uncorrected_refuted_claims|
```

Two optimisers are implemented:
- **`ScaleRLTrainer`** — PPO-style with KL penalty against a frozen reference copy of the student model.
- **`CISPOTrainer`** — Lagrangian-constrained importance sampling variant with tighter KL bounds.

### Data Schemas (`src/data/schemas.py`)

Pydantic v2 models are the contracts between pipeline stages. Key types:
`Claim` → `ValidatedClaim` → `ClaimExtractionOutput` → `ClaimValidationOutput` → `CorrectionOutput` → `ResponseEvaluationOutput` → `DatasetRecord`

`DatasetRecord` is the unit that gets written to JSONL and fed into probe/RL training.

### Prompt Templates (`prompts/`)

Plain-text files versioned in the repo. Each compilation stage loads its prompt by name from a `prompts: dict[str, str]` passed to the pipeline at construction. Modifying prompts does not require code changes.

## Key Configuration

`configs/base.yaml` sets global defaults (model name, dataset source, output paths, dataset size). Stage configs override specific keys. The merge pattern in `response_generator.py` is the reference:

```python
cfg = {**yaml.safe_load(base), **yaml.safe_load(stage_specific)}
```

WandB is used for experiment tracking; the `WANDB_API_KEY` must be set in `.env`.

## Implementation Status

Most `src/` files contain only stub skeletons (`...` bodies). The `scafold.py` file at the root is the consolidated API design reference — it documents the intended interfaces for all classes before they are implemented in their target files.
