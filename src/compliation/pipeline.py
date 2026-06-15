"""Compilation pipeline orchestrator.

Opens the SQLite DB, then runs each compilation stage in sequence.
Stages that are not yet implemented are left as commented stubs.

Usage:
    python src/compliation/pipeline.py
    python src/compliation/pipeline.py --config configs/test_run.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.writers import CompilationDB
from src.compliation import response_generator, cliaim_extractor


def _load_cfg(base_path: str = "configs/base.yaml") -> dict:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    # Merge stage-specific config
    gen_cfg_path = Path("configs/compilation/response_generation.yaml")
    if gen_cfg_path.exists():
        with open(gen_cfg_path) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def run(cfg: dict) -> None:
    db_path = Path(cfg.get("DB_PATH", "outputs/compilation.db"))

    with CompilationDB(db_path) as db:
        print(f"[pipeline] DB: {db_path.resolve()}")
        print(f"[pipeline] Records already in DB: {db.count()}")

        # ── Stage 1: Response generation ──────────────────────────────────
        print("[pipeline] Stage 1 — response generation")
        response_generator.run(db, cfg)

        # ── Stage 2: Claim extraction ──────────────────────────────────────
        print("[pipeline] Stage 2 — claim extraction")
        cliaim_extractor.run(db, cfg)

        # ── Stage 3: Claim validation ──────────────────────────────────────
        # from src.compliation import claim_validator
        # print("[pipeline] Stage 3 — claim validation")
        # claim_validator.run(db, cfg)

        # ── Stage 4: Correction generation ────────────────────────────────
        # from src.compliation import correction_generator
        # print("[pipeline] Stage 4 — correction generation")
        # correction_generator.run(db, cfg)

        # ── Stage 5: Response evaluation ──────────────────────────────────
        # from src.compliation import evaluator
        # print("[pipeline] Stage 5 — evaluation")
        # evaluator.run(db, cfg)

        print(f"[pipeline] Complete. Total records: {db.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to base config YAML (default: configs/base.yaml)",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    run(cfg)
