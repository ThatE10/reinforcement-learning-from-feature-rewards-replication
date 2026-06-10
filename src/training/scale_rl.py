# =============================================================================
# TARGET: src/training/scale_rl.py
# =============================================================================

@dataclass
class ScaleRLConfig:
    clip_ratio:       float = 0.2
    value_loss_coef:  float = 0.5
    entropy_coef:     float = 0.01
    max_grad_norm:    float = 1.0
    learning_rate:    float = 1e-5
    kl_penalty_coef:  float = 0.1


class ScaleRLTrainer:
    """ScaleRL policy optimiser for student model fine-tuning.

    Maintains a frozen reference copy of the student model to compute the
    KL-divergence penalty preventing excessive policy drift.
    """

    def __init__(
        self,
        student_model: BaseLanguageModel,
        reward_fn:     RewardFunction,
        probes:        dict[str, BaseProbe],
        config:        ScaleRLConfig,
    ) -> None: ...

    def step(
        self,
        question:           str,
        generated_response: str,
        correction:         CorrectionOutput,
        reward:             float,
    ) -> dict[str, float]:
        """Perform one policy update step.

        Returns:
            Loss-component dict: "policy_loss", "value_loss", "entropy",
            "kl_divergence", "total_loss".
        """
        ...

    def rollout(
        self,
        questions:  list[str],
        extractor:  ActivationExtractor,
    ) -> list[dict[str, Any]]:
        """Generate trajectories, extract activations, and compute rewards.

        Returns:
            List of trajectory dicts with keys: question, response, activations,
            probe_outputs, correction, reward.
        """
        ...
