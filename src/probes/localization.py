# =============================================================================
# TARGET: src/probes/localization.py
# =============================================================================

class LocalizationProbe(BaseProbe):
    """Binary logistic-regression classifier on question-level activations.

    Predicts the probability that the student model will hallucinate in
    response to a given question, based solely on question hidden states.
    """

    probe_type  = "localization"
    num_classes = 2

    def __init__(
        self,
        hidden_dim: int,
        layer:      int,
        dropout:    float = 0.1,
    ) -> None:
        """
        Args:
            layer:   Single layer index whose activations this probe consumes.
            dropout: Pre-classifier dropout probability.
        """
        ...

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute binary logits from question activations of shape (batch, hidden_dim)."""
        ...


