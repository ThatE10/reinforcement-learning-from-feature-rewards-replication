import json
import time
import wandb
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen3-8B"
BATCH_SIZE  = 512
DATASET_LEN = 20000

output_path = Path(
    "/users/enguye17/reinforcement-learning-from-feature-rewards-replication/Data/response.jsonl"
)

# ── Weights & Biases ──────────────────────────────────────────────────────────
wandb.init(
    project="longfact-generation",
    name=f"qwen3-8b-bs{BATCH_SIZE}",
    config={
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "dataset_len": DATASET_LEN,
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "n_completions": 4,
        "tensor_parallel_size": 2,
        "dtype": "float16",
    },
)

# ── Model initialisation ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=2,
    dtype="float16",
    gpu_memory_utilization=0.9,
    max_seq_len_to_capture=4708,
)

sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=0.95,
    n=4,
)

ds = load_dataset(
    "obalcells/longfact-augmented-prompts",
    streaming=True,
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

    batch_start = time.time()

    outputs = llm.generate(batch_prompts, sampling_params)

    batch_records = 0
    total_output_tokens = 0

    for example, output in zip(batch_examples, outputs):
        for completion in output.outputs:
            total_output_tokens += len(completion.token_ids)

            record = {
                "prompt": output.prompt,
                "concept": example.get("topic"),
                "question_type": None,
                "student_response": completion.text,
                "extracted_entities": [],
                "entity_context": [],
                "entity_labels": [],
                "entity_confidence_labels": [],
                "entity_verification_notes": [],
                "intervention": [],
                "intervention_label": [],
                "retraction_reward": [],
                "retraction_reward_notes": [],
                "correction_reward": [],
                "correct_reward_notes": [],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            batch_records += 1

    f.flush()

    batch_time = time.time() - batch_start

    records_written += batch_records
    examples_processed += len(batch_examples)

    total_runtime = time.time() - start_time

    examples_per_sec = len(batch_examples) / batch_time
    records_per_sec = batch_records / batch_time
    tokens_per_sec = total_output_tokens / batch_time

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