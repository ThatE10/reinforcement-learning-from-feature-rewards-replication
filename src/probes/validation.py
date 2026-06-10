# =============================================================================
# TARGET: src/probes/validation.py
# =============================================================================

class ValidationProbe(BaseProbe):
    """Softmax multi-class classifier on concatenated question + response activations.

    Predicts factual accuracy (correct / partial / incorrect) for
    question-response pairs drawn from the compiled dataset.
    """

    probe_type  = "validation"
    num_classes = 3

    def __init__(
        self,
        hidden_dim: int,
        layers:     list[int],
        mlp_hidden: int   = 256,
        dropout:    float = 0.1,
    ) -> None:
        """
        Args:
            layers:     Layer indices; question and response activations from
                        each layer are concatenated before the MLP head.
            mlp_hidden: Width of the single hidden layer in the MLP classifier.
        """
        ...

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            activations: Shape (batch, 2 * len(layers) * hidden_dim).
        """
        ...
