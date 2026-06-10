# =============================================================================
# TARGET: src/evaluation/sampling.py
# =============================================================================

@dataclass
class BestOfNConfig:
    n_samples:        int                                      = 8
    temperature:      float                                    = 0.9
    scoring_strategy: Literal["probe", "reward", "length_penalty"] = "probe"


class BestOfNSampler:
    """Generates N candidate responses and selects the best by a scoring strategy.

    When scoring_strategy="probe", the validation probe's predicted class
    probability over the CORRECT label serves as the selection criterion.
    """

    def __init__(
        self,
        student_model:    BaseLanguageModel,
        validation_probe: ValidationProbe,
        extractor:        ActivationExtractor,
        config:           BestOfNConfig,
    ) -> None: ...

    def sample(self, question: str) -> tuple[str, list[str]]:
        """Sample N responses; return (best_response, all_candidates)."""
        ...

    def _score_responses(
        self,
        question:  str,
        responses: list[str],
    ) -> list[float]:
        """Score each candidate response under the configured scoring strategy."""
        ...

