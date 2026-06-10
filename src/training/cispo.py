@dataclass
class CISPOConfig:
    importance_sampling_clip:    float = 0.3
    constraint_threshold:        float = 0.05
    constraint_learning_rate:    float = 1e-4
    learning_rate:               float = 1e-5


class CISPOTrainer:
    """CISPO policy optimiser with Lagrangian-constrained importance sampling.

    Extends standard policy gradient with a constraint mechanism that bounds
    KL divergence from the reference policy more tightly than clipped
    surrogate objectives alone.
    """

    def __init__(
        self,
        student_model: BaseLanguageModel,
        reward_fn:     RewardFunction,
        probes:        dict[str, BaseProbe],
        config:        CISPOConfig,
    ) -> None: ...

    def step(
        self,
        trajectories: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Perform one CISPO update over a batch of pre-collected trajectories."""
        ...

    def _compute_importance_weights(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clipped importance-sampling weights.

        Args:
            old_log_probs: Log-probabilities under the behaviour policy.
            new_log_probs: Log-probabilities under the current policy.
        """
        ...

