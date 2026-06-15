"""Stage 1 of the compilation pipeline: generate student model responses.

Run standalone:
    python src/compliation/response_generator.py

Or import and call run(db, cfg) from pipeline.py.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import wandb
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from src.data.writers import CompilationDB


# ── Model / sampler init ──────────────────────────────────────────────────────

def _build_model(cfg: dict) -> tuple[LLM, AutoTokenizer, SamplingParams]:
    model_name = cfg["MODEL_NAME"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=cfg["tensor_parallel_size"],
        dtype=cfg["dtype"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_seq_len_to_capture=cfg.get("max_prompt_tokens", 700) + cfg["max_tokens"],
    )
    sampling_params = SamplingParams(
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        n=cfg["n_completions"],
    )
    return llm, tokenizer, sampling_params


def _format_prompt(example: dict, tokenizer: AutoTokenizer) -> str:
    messages = [{
        "role": "user",
        "content": (
            f"{example['question']} Provide as many specific details and "
            "examples as possible (such as names of people, numbers, events, "
            "locations, dates, times, etc.)."
        ),
    }]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


# ── Batch processing ──────────────────────────────────────────────────────────

def _flush_batch(
    batch_examples: list[dict],
    batch_prompts: list[str],
    llm: LLM,
    sampling_params: SamplingParams,
    db: "CompilationDB",
    run_state: dict,
    pbar: tqdm,
) -> None:
    """Generate responses for a batch, write to DB, update WandB + tqdm."""
    batch_start = time.time()
    outputs = llm.generate(batch_prompts, sampling_params)

    records: list[dict] = []
    total_tokens = 0

    for example, output in zip(batch_examples, outputs):
        for completion in output.outputs:
            total_tokens += len(completion.token_ids)
            records.append({
                "prompt": output.prompt,
                "question": example.get("question"),
                "concept": example.get("topic"),
                "question_type": None,
                "student_response": completion.text,
            })

    db.insert_responses(records)

    batch_time = time.time() - batch_start
    run_state["records_written"] += len(records)
    run_state["examples_processed"] += len(batch_examples)
    total_runtime = time.time() - run_state["start_time"]

    examples_per_sec = len(batch_examples) / batch_time
    records_per_sec = len(records) / batch_time
    tokens_per_sec = total_tokens / batch_time

    wandb.log({
        "examples_processed": run_state["examples_processed"],
        "records_written": run_state["records_written"],
        "batch_size_actual": len(batch_examples),
        "batch_time_sec": batch_time,
        "examples_per_sec": examples_per_sec,
        "records_per_sec": records_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "output_tokens_batch": total_tokens,
        "avg_tokens_per_completion": total_tokens / len(records) if records else 0,
        "runtime_sec": total_runtime,
    })

    pbar.update(len(batch_examples))
    pbar.set_postfix({
        "records": run_state["records_written"],
        "tok/s": f"{tokens_per_sec:.0f}",
        "ex/s": f"{examples_per_sec:.2f}",
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def run(db: "CompilationDB", cfg: dict) -> None:
    """Generate student responses for DATASET_LEN questions and write to db.

    Resumes automatically: skips the first db.count() examples so a crashed
    run can be restarted without duplicating records.
    """
    dataset_len = cfg["DATASET_LEN"]
    batch_size = cfg["BATCH_SIZE"]
    n_completions = cfg["n_completions"]
    already_done = db.count()

    if already_done >= dataset_len * n_completions:
        print(
            f"[response_generator] DB already has {already_done} records "
            f"(target {dataset_len * n_completions}). Skipping."
        )
        return

    examples_to_skip = already_done // n_completions
    target_examples = dataset_len - examples_to_skip
    print(
        f"[response_generator] {already_done} records in DB; "
        f"skipping first {examples_to_skip} examples, "
        f"generating {target_examples} more."
    )

    llm, tokenizer, sampling_params = _build_model(cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = cfg["MODEL_NAME"].split("/")[-1]
    dataset_tag = cfg["QUESTION_SRC"].split("/")[-1]
    wandb.init(
        project=cfg["PROJECT_NAME"],
        name=f"{model_tag}_{dataset_tag}_{timestamp}",
        config=cfg,
        resume="allow",
    )

    ds = load_dataset(cfg["QUESTION_SRC"], streaming=True)

    run_state = {
        "records_written": already_done,
        "examples_processed": examples_to_skip,
        "start_time": time.time(),
    }

    batch_examples: list[dict] = []
    batch_prompts: list[str] = []
    examples_seen = 0

    with tqdm(total=dataset_len, initial=examples_to_skip, desc="Examples", unit="ex",
              dynamic_ncols=True) as pbar:
        for example in ds["train"]:
            # Skip examples already processed in a previous run
            if examples_seen < examples_to_skip:
                examples_seen += 1
                continue
            if run_state["examples_processed"] >= dataset_len:
                break

            batch_examples.append(example)
            batch_prompts.append(_format_prompt(example, tokenizer))
            examples_seen += 1

            if len(batch_prompts) == batch_size:
                _flush_batch(
                    batch_examples, batch_prompts,
                    llm, sampling_params, db, run_state, pbar,
                )
                batch_examples, batch_prompts = [], []

        if batch_prompts:
            _flush_batch(
                batch_examples, batch_prompts,
                llm, sampling_params, db, run_state, pbar,
            )

    total_runtime = time.time() - run_state["start_time"]
    wandb.summary.update({
        "final_records_written": run_state["records_written"],
        "final_examples_processed": run_state["examples_processed"],
        "total_runtime_sec": total_runtime,
        "avg_records_per_sec": run_state["records_written"] / total_runtime,
    })
    wandb.finish()

    print(
        f"\n[response_generator] Done — "
        f"{run_state['records_written']} records in DB at {db.db_path.resolve()}"
    )


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.writers import CompilationDB

    with open("configs/base.yaml") as f:
        base_cfg = yaml.safe_load(f)
    with open("configs/compilation/response_generation.yaml") as f:
        gen_cfg = yaml.safe_load(f)
    cfg = {**base_cfg, **gen_cfg}

    db_path = cfg.get("DB_PATH", "outputs/compilation.db")
    with CompilationDB(db_path) as db:
        run(db, cfg)
