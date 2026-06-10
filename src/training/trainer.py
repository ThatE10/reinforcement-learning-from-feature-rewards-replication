# =============================================================================
# TARGET: src/training/reward.py
# =============================================================================

@dataclass
class RewardConfig:
    localization_weight:  float = 0.2
    validation_weight:    float = 0.5
    intervention_weight:  float = 0.3
    correctness_bonus:    float = 1.0
    retraction_bonus:     float = 0.5
    hallucination_penalty: float = -2.0
    normalize:            bool  = True


class RewardFunction:
    """Composite reward combining probe outputs, retraction bonuses, and hallucination penalties.

    The signal structure is:
        r = w_loc * f(loc) + w_val * f(val) + w_int * f(int)
            + retraction_bonus * |correct_retractions|
            + hallucination_penalty * |uncorrected_refuted_claims|
    """

    def __init__(self, config: RewardConfig) -> None: ...

    def compute(
        self,
        localization_output:  ProbeOutput,
        validation_output:    ProbeOutput,
        intervention_output:  ProbeOutput,
        correction:           CorrectionOutput,
        evaluation:           ResponseEvaluationOutput,
    ) -> float:
        """Compute the scalar reward for a single generation episode."""
        ...

    def compute_batch(
        self,
        localization_outputs:  list[ProbeOutput],
        validation_outputs:    list[ProbeOutput],
        intervention_outputs:  list[ProbeOutput],
        corrections:           list[CorrectionOutput],
        evaluations:           list[ResponseEvaluationOutput],
    ) -> torch.Tensor:
        """Compute rewards for a batch; returns a tensor of shape (batch,)."""
        ...

    def _retraction_bonus(self, correction: CorrectionOutput) -> float:
        """Per-retraction bonus contribution from the correction's tag annotations."""
        ...

    def _hallucination_penalty(
        self,
        correction:  CorrectionOutput,
        validation:  ClaimValidationOutput,
    ) -> float:
        """Penalty for refuted claims that remain uncorrected in the final response."""
        ...


