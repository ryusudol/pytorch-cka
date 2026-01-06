"""Shared test fixtures for torchcka tests."""

import pytest

from .helpers import create_image_dataloader, create_text_dataloader


# ============================================================================
# SHARED FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def image_dataloader():
    """Standard 224x224 image dataloader."""
    return create_image_dataloader()


@pytest.fixture(scope="module")
def text_dataloader():
    """Standard text dataloader with dict batches."""
    return create_text_dataloader()
