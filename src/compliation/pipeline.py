# =============================================================================
# TARGET: src/compilation/pipeline.py
# =============================================================================
from dataclasses import field


@dataclass
class CompilationPipelineConfig:
    generator:        GeneratorConfig        = field(default_factory=GeneratorConfig)
    extractor:        ClaimExtractorConfig   = field(default_factory=ClaimExtractorConfig)
    validator:        ClaimValidatorConfig   = field(default_factory=ClaimValidatorConfig)
    corrector:        CorrectorConfig        = field(default_factory=CorrectorConfig)
    evaluator:        EvaluatorConfig        = field(default_factory=EvaluatorConfig)
    output_dir:       Path                   = Path("outputs/datasets")
    checkpoint_every: int                    = 100


class CompilationPipeline:
    """Orchestrates the full generator→extractor→validator→corrector→evaluator chain."""

    def __init__(
        self,
        student_model: BaseLanguageModel,
        teacher_model: BaseLanguageModel,
        oracle:        BaseOracle,
        config:        CompilationPipelineConfig,
        prompts:       dict[str, str],          # keyed by stage name
    ) -> None: ...

    def run_record(
        self,
        question_id: str,
        question:    str,
    ) -> DatasetRecord:
        """Execute the full pipeline for a single question-answer episode."""
        ...

    def run_dataset(
        self,
        dataset_path: Path,
        split:        Literal["train", "val", "test"] = "train",
    ) -> list[DatasetRecord]:
        """Execute the pipeline over a full JSONL dataset split with checkpointing."""
        ...
