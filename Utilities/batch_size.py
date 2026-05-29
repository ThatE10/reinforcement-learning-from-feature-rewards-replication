import time
import json
from pathlib import Path

import torch
import wandb

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B"

BATCH_SIZES = [128, 256, 512, 768, 1024, 1516, 2048, 2560, 3072, 4096]

N_SAMPLES = 4
MAX_TOKENS = 4608

DUMMY_PROMPT = (
    "Who was Julius Caesar? Provide as many specific details and examples "
    "as possible (such as names of people, numbers, events, locations, dates, times, etc.)."
)

# ── W&B ───────────────────────────────────────────────────────────────────────

wandb.init(
    project="vllm-batch-benchmark",
    config={
        "model": MODEL_NAME,
        "batch_sizes": BATCH_SIZES,
        "max_tokens": MAX_TOKENS,
        "n_samples": N_SAMPLES,
    },
)

# ── Model ─────────────────────────────────────────────────────────────────────

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=2,
    dtype="float16",
    gpu_memory_utilization=0.9,
    max_seq_len_to_capture=4708,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=1.0,
    top_p=0.95,
    n=N_SAMPLES,
)

# ── Prompt ────────────────────────────────────────────────────────────────────

def format_prompt(tokenizer) -> str:
    messages = [{"role": "user", "content": DUMMY_PROMPT}]

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_batch(batch_size: int, prompt: str) -> dict:

    prompts = [prompt] * batch_size

    # warmup
    llm.generate([prompt], SamplingParams(max_tokens=16, n=1))
    torch.cuda.synchronize()

    start = time.perf_counter()

    outputs = llm.generate(prompts, sampling_params)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    total_output_tokens = sum(
        len(completion.token_ids)
        for output in outputs
        for completion in output.outputs
    )

    result = {
        "batch_size": batch_size,
        "elapsed_seconds": round(elapsed, 3),
        "sequences_generated": batch_size * N_SAMPLES,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": round(total_output_tokens / elapsed, 1),
        "prompts_per_second": round(batch_size / elapsed, 3),
        "tokens_per_second_per_gpu": round(
            total_output_tokens / elapsed / torch.cuda.device_count(),
            1,
        ),
    }

    return result

# ── Run Benchmark ─────────────────────────────────────────────────────────────

results = []

prompt = format_prompt(tokenizer)

results_path = Path(__file__).parent / "batch_size_results.jsonl"

print(
    f"{'Batch':>6} | "
    f"{'Tok/s':>10} | "
    f"{'Prompt/s':>10} | "
    f"{'Elapsed':>10}"
)

print("-" * 60)

with open(results_path, "w", encoding="utf-8") as f:

    for bs in BATCH_SIZES:

        result = benchmark_batch(bs, prompt)

        results.append(result)

        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

        wandb.log(result)

        print(
            f"{result['batch_size']:>6} | "
            f"{result['tokens_per_second']:>10.1f} | "
            f"{result['prompts_per_second']:>10.3f} | "
            f"{result['elapsed_seconds']:>9.2f}s"
        )

print(f"\nResults written to: {results_path.resolve()}")

best = max(results, key=lambda r: r["tokens_per_second"])

print(
    f"\nHighest throughput at batch_size={best['batch_size']} "
    f"({best['tokens_per_second']} tok/s)"
)

wandb.summary["best_batch_size"] = best["batch_size"]
wandb.summary["best_tokens_per_second"] = best["tokens_per_second"]

wandb.finish()