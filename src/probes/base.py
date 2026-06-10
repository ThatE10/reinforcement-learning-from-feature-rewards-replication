# =============================================================================
# TARGET: src/probes/base.py
# =============================================================================
import torch.nn as nn


class BaseProbe(nn.Module, ABC):
    """Abstract base class for all three probing classifiers.

    Subclasses must specify `probe_type`, `hidden_dim`, and `num_classes`
    as class attributes, and implement `forward`.
    """

    probe_type:  str
    hidden_dim:  int
    num_classes: int

    @abstractmethod
    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute class logits from activation inputs.

        Args:
            activations: Shape (batch, D) where D depends on the probe's
                         concatenation strategy across layers and segments.

        Returns:
            Logit tensor of shape (batch, num_classes).
        """
        ...

    def predict(self, activations: torch.Tensor) -> ProbeOutput:
        """Run inference and return a structured ProbeOutput (no_grad context)."""
        ...

    def save(self, path: Path) -> None:
        """Serialise probe weights and metadata (hidden_dim, num_classes, layer config)."""
        ...

    @classmethod
    def load(cls, path: Path) -> "BaseProbe":
        """Deserialise a probe from a checkpoint written by .save()."""
        ...

