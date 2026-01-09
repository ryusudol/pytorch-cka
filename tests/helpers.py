"""Shared test helper functions for pytorch_cka tests."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# SHARED CONSTANTS
# ============================================================================

TEST_BATCH_SIZE = 16  # Must be > 3 for unbiased HSIC; 16 improves numerical stability
TEST_NUM_SAMPLES = 32  # Small for CI (2 batches)
TEST_IMAGE_SIZE = 224  # Standard ImageNet input size
TEST_SEQ_LENGTH = 32  # Short sequence for speed


# ============================================================================
# SHARED HELPER FUNCTIONS
# ============================================================================


def get_sample_layers(model: nn.Module, max_layers: int = 5) -> list[str]:
    """Get a sample of hookable layers from a model.

    Selects layers strategically: evenly distributed across network depth.
    """
    all_layers = [name for name, _ in model.named_modules() if name]

    if len(all_layers) <= max_layers:
        return all_layers

    # Select evenly distributed indices up to max_layers
    step = len(all_layers) // max_layers
    indices = [i * step for i in range(max_layers)]
    return [all_layers[i] for i in indices]


def get_layers_by_type(model: nn.Module, layer_type: type) -> list[str]:
    """Get all layer names of a specific type."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, layer_type) and name
    ]


def create_image_dataloader(
    batch_size: int = TEST_BATCH_SIZE,
    num_samples: int = TEST_NUM_SAMPLES,
    image_size: int = TEST_IMAGE_SIZE,
    channels: int = 3,
) -> DataLoader:
    """Create a lightweight dataloader with random image-like tensors."""
    data = torch.randn(num_samples, channels, image_size, image_size)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_text_dataloader(
    batch_size: int = TEST_BATCH_SIZE,
    num_samples: int = TEST_NUM_SAMPLES,
    seq_length: int = TEST_SEQ_LENGTH,
    vocab_size: int = 1000,
) -> DataLoader:
    """Create a dataloader with random tokenized text (dict format)."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)

    class DictDataset:
        """Simple dataset returning dict batches with input_ids."""

        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }

    dataset = DictDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================================
# SHARED ASSERTION HELPERS
# ============================================================================


def assert_valid_cka_matrix(
    matrix: torch.Tensor, expected_shape: tuple = None
) -> None:
    """Assert CKA matrix has valid numerical properties."""
    assert not torch.isnan(matrix).any(), "Matrix contains NaN values"
    assert not torch.isinf(matrix).any(), "Matrix contains Inf values"
    if expected_shape is not None:
        assert (
            matrix.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {matrix.shape}"


def assert_self_similarity_diagonal(matrix: torch.Tensor, atol: float = 0.05) -> None:
    """Assert diagonal values are ~1.0 for self-comparison."""
    diagonal = torch.diag(matrix)
    assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=atol)


def count_hooks(model: nn.Module) -> int:
    """Count total forward hooks registered on a model."""
    return sum(len(m._forward_hooks) for m in model.modules())
