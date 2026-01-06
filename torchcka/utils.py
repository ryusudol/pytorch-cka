"""Utility functions for CKA computation."""

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


def validate_batch_size(n: int, unbiased: bool = True) -> None:
    """Validate that batch size is sufficient for HSIC computation.

    Args:
        n: Batch size (number of samples).
        unbiased: Whether using unbiased estimator.

    Raises:
        ValueError: If n <= 3 for unbiased, or n <= 1 for biased.
    """
    if unbiased and n <= 3:
        raise ValueError(
            f"Unbiased HSIC requires batch size > 3, got {n}. "
            "Use unbiased=False or increase batch size."
        )
    if not unbiased and n <= 1:
        raise ValueError(f"HSIC requires batch size > 1, got {n}")


def validate_layers(
    model: nn.Module,
    layers: List[str],
) -> Tuple[List[str], List[str]]:
    """Validate layer names exist in model.

    Args:
        model: PyTorch model.
        layers: List of layer names to validate.

    Returns:
        Tuple of (valid_layers, invalid_layers).
    """
    model_layers = {name for name, _ in model.named_modules()}
    valid = [layer for layer in layers if layer in model_layers]
    invalid = [layer for layer in layers if layer not in model_layers]
    return valid, invalid


def get_all_layer_names(model: nn.Module, include_root: bool = False) -> List[str]:
    """Get all layer names from a model.

    Args:
        model: PyTorch model.
        include_root: Whether to include the root module (empty string name).

    Returns:
        List of layer names.
    """
    names = [name for name, _ in model.named_modules()]
    if not include_root and names and names[0] == "":
        names = names[1:]
    return names


def auto_select_layers(
    model: nn.Module,
    max_layers: int = 50,
    model_name: str = "Model",
) -> Tuple[List[str], bool]:
    """Auto-select layers from a model with optional truncation.

    Args:
        model: PyTorch model to extract layers from.
        max_layers: Maximum number of layers to select.
        model_name: Name to use in warning message.

    Returns:
        Tuple of (selected_layers, truncated).
    """
    import warnings

    all_layers = get_all_layer_names(model)
    truncated = len(all_layers) > max_layers

    if truncated:
        selected = all_layers[:max_layers]
        warnings.warn(
            f"{model_name} has {len(all_layers)} layers. Auto-selected first {max_layers}. "
            "Consider specifying layers explicitly for better control."
        )
    else:
        selected = all_layers

    return selected, truncated


def get_device(
    model: nn.Module,
    fallback: Optional[torch.device] = None,
) -> torch.device:
    """Get device from model parameters.

    Args:
        model: PyTorch model.
        fallback: Device to use if model has no parameters.

    Returns:
        Device the model is on.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback or torch.device("cpu")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel/DistributedDataParallel wrapper.

    Args:
        model: Potentially wrapped model.

    Returns:
        Unwrapped model (accesses .module attribute if wrapped).
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


class FeatureCache:
    """Feature cache for hook outputs.

    Stores layer outputs with optional detaching for memory efficiency.

    Attributes:
        detach: Whether to detach tensors from computation graph.
    """

    def __init__(self, detach: bool = True) -> None:
        """Initialize feature cache.

        Args:
            detach: Whether to detach tensors from computation graph.
        """
        self._features: Dict[str, torch.Tensor] = {}
        self._detach = detach

    def store(self, name: str, tensor: torch.Tensor) -> None:
        """Store a feature tensor.

        Args:
            name: Layer name.
            tensor: Feature tensor to store.
        """
        if self._detach:
            tensor = tensor.detach()
        self._features[name] = tensor

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Get a stored feature tensor.

        Args:
            name: Layer name.

        Returns:
            Stored tensor or None if not found.
        """
        return self._features.get(name)

    def clear(self) -> None:
        """Clear all stored features."""
        self._features.clear()

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over stored features.

        Yields:
            Tuples of (layer_name, tensor).
        """
        return iter(self._features.items())

    def keys(self) -> Iterator[str]:
        """Iterate over layer names.

        Yields:
            Layer names.
        """
        return iter(self._features.keys())

    def __len__(self) -> int:
        """Return number of stored features."""
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        """Check if a layer name is in cache."""
        return name in self._features


@contextmanager
def eval_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Context manager to temporarily set model to eval mode.

    Restores original training state on exit.

    Args:
        model: PyTorch model.

    Yields:
        The model in eval mode.
    """
    was_training = model.training
    try:
        model.eval()
        yield model
    finally:
        model.train(was_training)
