# =============================================================================
# TARGET: src/probes/trainer.py
# =============================================================================
from torch.utils.data import DataLoader, Dataset


@dataclass
class ProbeTrainerConfig:
    learning_rate:           float = 1e-3
    weight_decay:            float = 1e-4
    epochs:                  int   = 10
    batch_size:              int   = 64
    early_stopping_patience: int   = 3
    checkpoint_dir:          Path  = Path("outputs/checkpoints/probes")
    device:                  str   = "cuda"


class ActivationDataset(Dataset):
    """PyTorch Dataset wrapping pre-extracted activation tensors and integer labels."""

    def __init__(
        self,
        activations: dict[int, torch.Tensor],   # layer → (n, hidden_dim)
        labels:      torch.Tensor,
        probe_type:  Literal["localization", "validation", "intervention"],
    ) -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class ProbeTrainer:
    """Trains any BaseProbe subclass with cross-entropy loss and early stopping."""

    def __init__(
        self,
        probe:  BaseProbe,
        config: ProbeTrainerConfig,
    ) -> None: ...

    def train(
        self,
        train_dataset: ActivationDataset,
        val_dataset:   ActivationDataset,
    ) -> dict[str, list[float]]:
        """Train the probe; returns history dict with train/val loss and accuracy.

        Returns:
            Keys: "train_loss", "val_loss", "train_acc", "val_acc".
        """
        ...

    def evaluate(
        self,
        dataset: ActivationDataset,
    ) -> dict[str, float]:
        """Compute loss, accuracy, precision, recall, and F1 on a held-out dataset."""
        ...


