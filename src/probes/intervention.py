# =============================================================================
# TARGET: src/probes/intervention.py
# =============================================================================

class InterventionProbe(BaseProbe):
    """Quality classifier on three-segment activations: question, response, intervention.

    Evaluates correction quality to provide a reward signal for the RL
    training stage, labelling each correction episode as high/partial/low quality.
    """

    probe_type  = "intervention"
    num_classes = 3

    def __init__(
        self,
        hidden_dim:              int,
        layers:                  list[int],
        mlp_hidden:              int   = 256,
        dropout:                 float = 0.1,
        attention_aggregation:   bool  = False,
    ) -> None:
        """
        Args:
            attention_aggregation: If True, a learnable attention head aggregates
                the three activation segments rather than naive concatenation.
        """
        ...

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute quality logits.

        Args:
            activations: Shape (batch, 3 * len(layers) * hidden_dim).
        """
        ...