import time
from datetime import datetime
from pathlib import Path

import wandb
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



with open("configs/compilation/default.yaml") as f:
    default_cfg = yaml.safe_load(f)

with open("configs/compilation/response_generation.yaml") as f:
    generation_cfg = yaml.safe_load(f)

cfg = {
    **default_cfg,
    **generation_cfg,
}


PROJECT_NAME = cfg["PROJECT_NAME"]

MODEL_NAME = cfg["MODEL_NAME"]
QUESTION_SRC = cfg["QUESTION_SRC"]

DATASET_OUTPUT_PATH = cfg["DATASET_OUTPUT_PATH"]

BATCH_SIZE = cfg["BATCH_SIZE"]

MAX_TOKENS = cfg["max_tokens"]
TEMPERATURE = cfg["temperature"]
TOP_P = cfg["top_p"]
N_COMPLETIONS = cfg["n_completions"]

TENSOR_PARALLEL_SIZE = cfg["tensor_parallel_size"]
DTYPE = cfg["dtype"]
GPU_MEMORY_UTILIZATION = cfg["gpu_memory_utilization"]

DATASET_LEN = cfg["DATASET_LEN"]


# ── Output Paths ──────────────────────────────────────────────────────────────

output_path = Path(DATASET_OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Dataset ───────────────────────────────────────────────────────────────────

ds = load_dataset(
    QUESTION_SRC,
    split="train",
    streaming=True,
)

dataset_name = QUESTION_SRC.split("/")[-1]
effective_dataset_len = DATASET_LEN * N_COMPLETIONS

#used in the logging of the model 
wandb.init(
    project=PROJECT_NAME,
    name=f"{MODEL_NAME.split('/')[-1]}_{dataset_name}_{timestamp}",
    config=cfg,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Optional: estimate from dataset beforehand
max_prompt_tokens = 700 #could cause problems in the future but shrug

max_seq_len_to_capture = (
    max_prompt_tokens
    + MAX_TOKENS
)


llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    dtype=DTYPE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_seq_len_to_capture=max_seq_len_to_capture,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    n=N_COMPLETIONS,
)


# ── Global counters ───────────────────────────────────────────────────────────
records_written = 0
examples_processed = 0
start_time = time.time()

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_prompt(example: dict) -> str:
    messages = [{
        "role": "user",
        "content": (
            f"{example['question']} Provide as many specific details and "
            "examples as possible (such as names of people, numbers, events, "
            "locations, dates, times, etc.)."
        )
    }]

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def flush_batch(
    batch_examples: list,
    batch_prompts: list,
    f,
    pbar_examples: tqdm,
):
    global records_written
    global examples_processed



    # ── WandB logging ────────────────────────────────────────────────────────
    wandb.log({
        "examples_processed": examples_processed,
        "records_written": records_written,
        "batch_size_actual": len(batch_examples),
        "batch_time_sec": batch_time,
        "examples_per_sec": examples_per_sec,
        "records_per_sec": records_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "output_tokens_batch": total_output_tokens,
        "avg_tokens_per_completion": (
            total_output_tokens / batch_records
            if batch_records > 0 else 0
        ),
        "runtime_sec": total_runtime,
    })

    # ── tqdm updates ─────────────────────────────────────────────────────────
    pbar_examples.update(len(batch_examples))

    pbar_examples.set_postfix({
        "batch": len(batch_examples),
        "records": records_written,
        "tok/s": f"{tokens_per_sec:.0f}",
        "ex/s": f"{examples_per_sec:.2f}",
    })


# ── Main generation loop ──────────────────────────────────────────────────────
batch_examples, batch_prompts = [], []

print(f"Writing to: {output_path.resolve()}\n")

with open(output_path, "a", encoding="utf-8") as f, \
     tqdm(
         total=DATASET_LEN,
         desc="Examples",
         unit="ex",
         dynamic_ncols=True,
     ) as pbar_examples:

    for example in ds["train"]:
        batch_examples.append(example)
        batch_prompts.append(format_prompt(example))

        if len(batch_prompts) == BATCH_SIZE:
            flush_batch(batch_examples, batch_prompts, f, pbar_examples)
            batch_examples, batch_prompts = [], []

    # ── Remainder batch ───────────────────────────────────────────────────────
    if batch_prompts:
        flush_batch(batch_examples, batch_prompts, f, pbar_examples)

# ── Final summary logging ─────────────────────────────────────────────────────
total_runtime = time.time() - start_time

wandb.summary["final_records_written"] = records_written
wandb.summary["final_examples_processed"] = examples_processed
wandb.summary["total_runtime_sec"] = total_runtime
wandb.summary["avg_records_per_sec"] = records_written / total_runtime

wandb.finish()

print(
    f"\nGeneration complete — "
    f"{records_written} records written to {output_path.resolve()}"
)
