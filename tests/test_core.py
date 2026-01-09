"""Tests for pytorch_cka.core module."""

import pytest
import torch

from pytorch_cka.core import (
    compute_gram_matrix,
    hsic,
)


class TestComputeGramMatrix:
    """Tests for compute_gram_matrix function."""

    def test_invalid_dims(self):
        """compute_gram_matrix should raise ValueError for non-2D tensors."""
        x_1d = torch.randn(10)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            compute_gram_matrix(x_1d)

        x_3d = torch.randn(5, 10, 5)
        with pytest.raises(ValueError, match="requires 2D tensor"):
            compute_gram_matrix(x_3d)

    def test_shape(self):
        """compute_gram_matrix should produce (n, n) gram matrix."""
        x = torch.randn(10, 5)
        K = compute_gram_matrix(x)
        assert K.shape == (10, 10)

    def test_symmetry(self):
        """Gram matrix should be symmetric."""
        x = torch.randn(10, 5)
        K = compute_gram_matrix(x)
        assert torch.allclose(K, K.T)

    def test_positive_semidefinite(self):
        """Gram matrix should be positive semi-definite."""
        x = torch.randn(10, 5, dtype=torch.float64)
        K = compute_gram_matrix(x)
        eigenvalues = torch.linalg.eigvalsh(K)
        # Allow small numerical errors
        assert (eigenvalues >= -1e-6).all()

    def test_computation(self):
        """compute_gram_matrix should compute K = X @ X^T."""
        x = torch.randn(5, 3)
        K = compute_gram_matrix(x)
        expected = x @ x.T
        assert torch.allclose(K, expected)


class TestHSIC:
    """Tests for HSIC function."""

    def test_invalid_dims(self):
        """hsic should raise for non-2D gram matrices."""
        gram_valid = torch.randn(10, 10)
        gram_1d = torch.randn(10)

        with pytest.raises(ValueError, match="requires 2D tensors"):
            hsic(gram_1d, gram_valid)

        with pytest.raises(ValueError, match="requires 2D tensors"):
            hsic(gram_valid, gram_1d)

    def test_shape_mismatch(self):
        """hsic should raise when gram matrix shapes don't match."""
        gram_10x10 = torch.randn(10, 10)
        gram_8x8 = torch.randn(8, 8)

        with pytest.raises(ValueError, match="requires matching shapes"):
            hsic(gram_10x10, gram_8x8)

    def test_non_square(self):
        """hsic should raise for non-square gram matrices."""
        gram_rect = torch.randn(10, 15)

        with pytest.raises(ValueError, match="requires square matrices"):
            hsic(gram_rect, gram_rect)

    def test_requires_n_gt_3(self):
        """HSIC should raise error for n <= 3."""
        K = torch.randn(3, 3)
        K = K @ K.T  # Make symmetric
        with pytest.raises(ValueError, match="n > 3"):
            hsic(K, K)

    def test_hsic_identical_positive(self):
        """HSIC(K, K) should be positive for non-trivial K."""
        x = torch.randn(10, 5, dtype=torch.float64)
        K = compute_gram_matrix(x)
        hsic_val = hsic(K, K)
        assert hsic_val > 0
