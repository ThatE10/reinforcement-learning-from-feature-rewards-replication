# =============================================================================
# scaffold.py
#
# Consolidated API scaffold for the hallucination_probe project.
# Each section is annotated with its target file path.
# Function bodies are intentionally stubbed with `...`.
# =============================================================================



# =============================================================================
# TARGET: src/models/base.py
# Abstract interfaces for all model and oracle backends.
# Concrete implementations register themselves in the ModelRegistry so that
# config-driven instantiation requires no import-time coupling.
# =============================================================================
import torch
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class GenerationConfig(Protocol):
    max_new_tokens: int
    temperature:    float
    top_p:          float
    do_sample:      bool


class BaseLanguageModel(ABC):
    """Abstract interface for any language model (student or teacher role)."""

    model_id: str
    device:   str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt:  Full formatted prompt string.
            config:  Generation hyperparameters; falls back to instance defaults.
            **kwargs: Backend-specific arguments.

        Returns:
            Generated text, excluding the prompt itself.
        """
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        config: Optional[GenerationConfig] = None,
    ) -> BaseModel:
        """Generate a structured response conforming to a Pydantic schema.

        Args:
            prompt:  Full formatted prompt string.
            schema:  Pydantic model class defining the expected output structure.
            config:  Generation hyperparameters.

        Returns:
            An instantiated and validated schema object.
        """
        ...

    @abstractmethod
    def get_activations(
        self,
        input_ids: torch.Tensor,
        layers: list[int],
        aggregation: Literal["mean", "last", "cls"] = "mean",
    ) -> dict[int, torch.Tensor]:
        """Extract hidden-state activations at specified transformer layers.

        Args:
            input_ids:   Tokenised input tensor of shape (batch, seq_len).
            layers:      Zero-indexed layer indices to extract from.
            aggregation: Sequence-dimension reduction strategy.

        Returns:
            Mapping from layer index to activation tensor (batch, hidden_dim).
        """
        ...


class BaseOracle(ABC):
    """Abstract interface for the factual grounding oracle (web search, RAG, etc.)."""

    @abstractmethod
    def query(self, claim: str, context: Optional[str] = None) -> str:
        """Retrieve external evidence relevant to a factual claim.

        Args:
            claim:   Claim string to ground.
            context: Optional surrounding context to disambiguate the query.

        Returns:
            Evidence text retrieved from the oracle.
        """
        ...

    @abstractmethod
    def batch_query(self, claims: list[str]) -> list[str]:
        """Query the oracle for multiple claims, potentially in parallel.

        Args:
            claims: List of claim strings.

        Returns:
            Evidence strings in input order.
        """
        ...


class ModelRegistry:
    """Central registry enabling config-driven model and oracle instantiation."""

    _student: dict[str, type[BaseLanguageModel]] = {}
    _teacher: dict[str, type[BaseLanguageModel]] = {}
    _oracle:  dict[str, type[BaseOracle]]         = {}

    @classmethod
    def register_student(cls, name: str):
        """Class decorator that registers a student model implementation."""
        def decorator(klass: type[BaseLanguageModel]) -> type[BaseLanguageModel]:
            cls._student[name] = klass
            return klass
        return decorator

    @classmethod
    def register_teacher(cls, name: str):
        """Class decorator that registers a teacher model implementation."""
        def decorator(klass: type[BaseLanguageModel]) -> type[BaseLanguageModel]:
            cls._teacher[name] = klass
            return klass
        return decorator

    @classmethod
    def register_oracle(cls, name: str):
        """Class decorator that registers an oracle implementation."""
        def decorator(klass: type[BaseOracle]) -> type[BaseOracle]:
            cls._oracle[name] = klass
            return klass
        return decorator

    @classmethod
    def build_student(cls, name: str, **kwargs: Any) -> BaseLanguageModel:
        """Instantiate a registered student model by name."""
        ...

    @classmethod
    def build_teacher(cls, name: str, **kwargs: Any) -> BaseLanguageModel:
        """Instantiate a registered teacher model by name."""
        ...

    @classmethod
    def build_oracle(cls, name: str, **kwargs: Any) -> BaseOracle:
        """Instantiate a registered oracle by name."""
        ...




# =============================================================================
# TARGET: src/compilation/claim_extractor.py
# =============================================================================

@dataclass
class ClaimExtractorConfig:
    max_claims_per_response: int  = 20
    min_claim_length:        int  = 10
    include_named_entities:  bool = True


class ClaimExtractor:
    """Extracts atomic factual claims and named entities from student responses.

    The teacher model is invoked with constrained generation to produce a
    ClaimExtractionOutput conforming to the JSONL schema.
    """

    def __init__(
        self,
        teacher_model:   BaseLanguageModel,
        config:          ClaimExtractorConfig,
        prompt_template: str,
    ) -> None: ...

    def extract(
        self,
        question_id:   str,
        question:      str,
        response_text: str,
    ) -> ClaimExtractionOutput:
        """Extract claims from a single question-response pair."""
        ...

    def extract_batch(
        self,
        records: list[tuple[str, str, str]],  # (question_id, question, response)
    ) -> list[ClaimExtractionOutput]:
        """Extract claims from a batch; preserves input order."""
        ...

    def _parse_structured_output(
        self,
        raw_output:  str,
        question_id: str,
    ) -> ClaimExtractionOutput:
        """Parse and validate the teacher model's raw JSON output.

        Raises:
            ValueError: If the output cannot be coerced into ClaimExtractionOutput.
        """
        ...


# =============================================================================
# TARGET: src/compilation/claim_validator.py
# =============================================================================

@dataclass
class ClaimValidatorConfig:
    oracle_timeout_seconds:  float = 10.0
    max_retries:             int   = 3
    confidence_threshold:    float = 0.5
    parallel_oracle_queries: bool  = True


class ClaimValidator:
    """Validates extracted claims using a teacher model augmented with an oracle.

    The oracle retrieves grounding evidence per claim; the teacher model then
    predicts a ClaimLabel and confidence score from the (claim, evidence) pair.
    """

    def __init__(
        self,
        teacher_model:   BaseLanguageModel,
        oracle:          BaseOracle,
        config:          ClaimValidatorConfig,
        prompt_template: str,
    ) -> None: ...

    def validate_claim(self, claim: Claim, question: str) -> ValidatedClaim:
        """Validate a single claim against oracle evidence."""
        ...

    def validate_extraction(
        self,
        extraction: ClaimExtractionOutput,
    ) -> ClaimValidationOutput:
        """Validate all claims in a ClaimExtractionOutput."""
        ...

    def _query_oracle_with_retry(
        self,
        claim_text: str,
        context:    Optional[str] = None,
    ) -> str:
        """Query the oracle with exponential-backoff retries on transient failure."""
        ...


# =============================================================================
# TARGET: src/compilation/corrector.py
# =============================================================================

@dataclass
class CorrectorConfig:
    max_new_tokens:    int            = 768
    temperature:       float          = 0.3
    retraction_tokens: tuple[str, ...] = (
        "<retract>", "<correct>", "<confirm>", "<uncertain>"
    )


class ResponseCorrector:
    """Prompts the student model to emit self-corrected responses with retraction tags.

    Validated claims are serialised into a natural-language correction context
    that the student model conditions on; tag tokens provide soft structural
    supervision for the retraction behaviour.
    """

    def __init__(
        self,
        student_model:   BaseLanguageModel,
        config:          CorrectorConfig,
        prompt_template: str,
    ) -> None: ...

    def correct(
        self,
        question_id:       str,
        question:          str,
        original_response: str,
        validation:        ClaimValidationOutput,
    ) -> CorrectionOutput:
        """Generate a corrected response with inline retraction tags."""
        ...

    def _format_validation_context(
        self,
        validation: ClaimValidationOutput,
    ) -> str:
        """Serialise validated claims into the correction-prompt context block."""
        ...

    def _parse_retraction_tags(
        self,
        corrected_response: str,
    ) -> tuple[list[RetractionTag], list[str]]:
        """Extract all retraction tags and their enclosed text spans.

        Returns:
            Tuple of (tag_enums, retracted_text_spans).
        """
        ...


# =============================================================================
# TARGET: src/compilation/evaluator.py
# =============================================================================

@dataclass
class EvaluatorConfig:
    token_label_map: Optional[dict[int, EvaluationLabel]] = None

    def __post_init__(self) -> None:
        if self.token_label_map is None:
            self.token_label_map = {
                0: EvaluationLabel.INCORRECT,
                1: EvaluationLabel.PARTIALLY_CORRECT,
                2: EvaluationLabel.CORRECT,
            }


class ResponseEvaluator:
    """Teacher-model evaluator assigning a token-based quality score per response."""

    def __init__(
        self,
        teacher_model:   BaseLanguageModel,
        config:          EvaluatorConfig,
        prompt_template: str,
    ) -> None: ...

    def evaluate(
        self,
        question_id:        str,
        question:           str,
        corrected_response: str,
        validation:         ClaimValidationOutput,
    ) -> ResponseEvaluationOutput:
        """Assign a token score to a corrected response given ground-truth validation."""
        ...





# =============================================================================
# TARGET: src/probes/activation_extractor.py
# =============================================================================

@dataclass
class ActivationExtractorConfig:
    layers:      list[int]
    aggregation: Literal["mean", "last", "cls"] = "mean"
    batch_size:  int  = 16
    dtype:       str  = "float32"   # float16, bfloat16, or float32


class ActivationExtractor:
    """Hooks into a transformer's forward pass to capture layer-wise hidden states.

    Registers `register_forward_hook` handles at construction and exposes them
    via explicit `_register_hooks` / `_remove_hooks` to prevent accumulation
    across calls.
    """

    def __init__(
        self,
        model:  BaseLanguageModel,
        config: ActivationExtractorConfig,
    ) -> None: ...

    def extract(
        self,
        texts:     list[str],
        return_as: Literal["dict", "tensor", "numpy"] = "tensor",
    ) -> dict[int, torch.Tensor]:
        """Extract activations for a list of raw text strings.

        Returns:
            Layer index → activation tensor of shape (n_samples, hidden_dim).
        """
        ...

    def extract_for_record(
        self,
        record: DatasetRecord,
    ) -> dict[str, dict[int, torch.Tensor]]:
        """Extract activations for all three probe-relevant segments of a DatasetRecord.

        Returns:
            Dict with keys "question", "response", "intervention", each mapping
            layer index → activation tensor.
        """
        ...

    def _register_hooks(self) -> list[Any]:
        """Register forward hooks; returns handles required for subsequent cleanup."""
        ...

    def _remove_hooks(self, handles: list[Any]) -> None:
        """Remove all registered hooks to prevent memory leaks between calls."""
        ...













# TARGET: tests/conftest.py
# Root-level pytest conftest: session- and function-scoped fixtures shared
# across both the unit and integration test suites.
# =============================================================================
import pytest
from unittest.mock import MagicMock
import torch


# ─── Model Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_student_model() -> BaseLanguageModel:
    """Deterministic mock student model — no GPU, no API calls."""
    model = MagicMock(spec=BaseLanguageModel)
    model.model_id = "mock-student-v0"
    model.generate.return_value = "Paris is the capital of France."
    model.get_activations.side_effect = (
        lambda input_ids, layers, **kw: {l: torch.randn(input_ids.shape[0], 768) for l in layers}
    )
    return model


@pytest.fixture(scope="session")
def mock_teacher_model() -> BaseLanguageModel:
    model = MagicMock(spec=BaseLanguageModel)
    model.model_id = "mock-teacher-v0"
    model.generate.return_value = '{"claims": []}'
    return model


@pytest.fixture(scope="session")
def mock_oracle() -> BaseOracle:
    oracle = MagicMock(spec=BaseOracle)
    oracle.query.return_value = "France is in Western Europe; its capital is Paris."
    oracle.batch_query.side_effect = lambda claims: [oracle.query.return_value] * len(claims)
    return oracle


# ─── Canonical Data Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def sample_claim() -> Claim:
    return Claim(
        claim_id="c_001",
        text="Paris is the capital of France.",
        named_entities=["Paris", "France"],
        source_sentence="Paris is the capital of France.",
    )


@pytest.fixture
def sample_validated_claim(sample_claim) -> ValidatedClaim:
    return ValidatedClaim(
        **sample_claim.model_dump(),
        label=ClaimLabel.SUPPORTED,
        evidence="Paris is the capital and largest city of France.",
        confidence=0.97,
    )


@pytest.fixture
def sample_extraction(sample_claim) -> ClaimExtractionOutput:
    return ClaimExtractionOutput(
        question_id="q_001",
        model_id="mock-student-v0",
        question="What is the capital of France?",
        response_text="Paris is the capital of France.",
        claims=[sample_claim],
    )


@pytest.fixture
def sample_validation(sample_validated_claim) -> ClaimValidationOutput:
    return ClaimValidationOutput(
        question_id="q_001",
        validated_claims=[sample_validated_claim],
    )


@pytest.fixture
def sample_correction() -> CorrectionOutput:
    return CorrectionOutput(
        question_id="q_001",
        original_response="Paris is the capital of France.",
        corrected_response="Paris is the capital of France.",
        retraction_tags=[],
        retracted_claims=[],
    )


@pytest.fixture
def sample_evaluation() -> ResponseEvaluationOutput:
    return ResponseEvaluationOutput(
        question_id="q_001",
        label=EvaluationLabel.CORRECT,
        token_score=2,
    )


@pytest.fixture
def sample_record(
    sample_extraction,
    sample_validation,
    sample_correction,
    sample_evaluation,
) -> DatasetRecord:
    return DatasetRecord(
        question_id="q_001",
        question="What is the capital of France?",
        student_response="Paris is the capital of France.",
        extraction=sample_extraction,
        validation=sample_validation,
        correction=sample_correction,
        evaluation=sample_evaluation,
    )


# ─── Activation Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def synthetic_activations() -> dict[int, torch.Tensor]:
    """Synthetic activations for layers 12, 16, 20 with hidden_dim=768, batch=4."""
    torch.manual_seed(42)
    return {l: torch.randn(4, 768) for l in [12, 16, 20]}


@pytest.fixture
def localization_probe_instance() -> LocalizationProbe:
    return LocalizationProbe(hidden_dim=768, layer=12)


@pytest.fixture
def validation_probe_instance() -> ValidationProbe:
    return ValidationProbe(hidden_dim=768, layers=[12, 16])


@pytest.fixture
def intervention_probe_instance() -> InterventionProbe:
    return InterventionProbe(hidden_dim=768, layers=[12, 16])


# =============================================================================
# TARGET: tests/unit/probes/test_localization.py
# Representative unit test patterns — replicate structure for all probe tests.
# =============================================================================

class TestLocalizationProbe:

    def test_output_shape(self, localization_probe_instance, synthetic_activations):
        """Forward pass must emit logits of shape (batch, 2)."""
        acts = synthetic_activations[12]           # (4, 768)
        logits = localization_probe_instance(acts)
        assert logits.shape == (4, 2)

    def test_predict_returns_probe_output(self, localization_probe_instance, synthetic_activations):
        acts = synthetic_activations[12]
        out = localization_probe_instance.predict(acts)
        assert isinstance(out, ProbeOutput)
        assert out.probe_type == "localization"
        assert len(out.probabilities) == 2
        assert abs(sum(out.probabilities) - 1.0) < 1e-5  # softmax sums to 1

    def test_save_load_roundtrip(self, localization_probe_instance, tmp_path):
        """Serialised and reloaded probe must produce identical logits."""
        acts = torch.randn(2, 768)
        logits_before = localization_probe_instance(acts).detach()
        localization_probe_instance.save(tmp_path)
        reloaded = LocalizationProbe.load(tmp_path)
        logits_after = reloaded(acts).detach()
        assert torch.allclose(logits_before, logits_after, atol=1e-6)


# =============================================================================
# TARGET: tests/unit/training/test_reward.py
# =============================================================================

class TestRewardFunction:

    @pytest.fixture
    def reward_fn(self):
        return RewardFunction(RewardConfig(normalize=False))

    def test_correctness_bonus_applied(self, reward_fn, sample_correction, sample_evaluation):
        """A fully correct evaluation with no retractions should yield the correctness bonus."""
        loc = ProbeOutput(probe_type="localization", question_id="q_001",
                          logits=[0.1, 0.9], probabilities=[0.1, 0.9],
                          predicted_class=1, confidence=0.9)
        val = ProbeOutput(probe_type="validation", question_id="q_001",
                          logits=[0.0, 0.1, 2.0], probabilities=[0.05, 0.1, 0.85],
                          predicted_class=2, confidence=0.85)
        ivn = ProbeOutput(probe_type="intervention", question_id="q_001",
                          logits=[0.0, 0.1, 2.0], probabilities=[0.05, 0.1, 0.85],
                          predicted_class=2, confidence=0.85)
        reward = reward_fn.compute(loc, val, ivn, sample_correction, sample_evaluation)
        assert isinstance(reward, float)
        assert reward > 0.0

    def test_hallucination_penalty_reduces_reward(self, reward_fn):
        """Uncorrected refuted claims must reduce the reward relative to the clean case."""
        ...   # construct a CorrectionOutput with retracted_claims=[] but a refuted ValidatedClaim