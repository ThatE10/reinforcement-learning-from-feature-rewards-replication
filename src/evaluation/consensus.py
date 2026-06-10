# =============================================================================
# TARGET: src/evaluation/consensus.py
# =============================================================================

class MajorityConsensus:
    """Selects the response whose factual content receives majority probe support
    across N sampled candidates.
    """

    def __init__(self, validation_probe: ValidationProbe) -> None: ...

    def select(
        self,
        question:    str,
        candidates:  list[str],
        activations: list[dict[int, torch.Tensor]],
    ) -> str:
        """Select the consensus-optimal response from a set of candidates."""
        ...