import torch


def hsic(
    gram_x: torch.Tensor,
    gram_y: torch.Tensor,
) -> torch.Tensor:
    if gram_x.dim() != 3 or gram_y.dim() != 3:
        raise ValueError(
            f"hsic requires 3D tensors, got shapes {gram_x.shape} and {gram_y.shape}"
        )

    if gram_x.shape != gram_y.shape:
        raise ValueError(
            f"hsic requires matching shapes, got {gram_x.shape} and {gram_y.shape}"
        )

    _, n, m = gram_x.shape
    if n != m:
        raise ValueError(f"hsic requires square matrices, got shape {gram_x.shape}")

    if n <= 3:
        raise ValueError(f"hsic requires n > 3, got n={n}")

    diag_x = torch.diagonal(gram_x, dim1=-2, dim2=-1)  # (batch, n)
    diag_y = torch.diagonal(gram_y, dim1=-2, dim2=-1)  # (batch, n)

    # Term 1: tr(K @ L) where K, L have zero diagonals
    trace_KL = (gram_x * gram_y).sum(dim=(-2, -1)) - (diag_x * diag_y).sum(dim=-1)

    # Term 2: (1^T K 1)(1^T L 1) / ((n-1)(n-2))
    sum_K = gram_x.sum(dim=(-2, -1)) - diag_x.sum(dim=-1)
    sum_L = gram_y.sum(dim=(-2, -1)) - diag_y.sum(dim=-1)
    term2 = (sum_K * sum_L) / ((n - 1) * (n - 2))

    # Term 3: 2 * (col_sum_K @ col_sum_L) / (n-2)
    col_sum_K = gram_x.sum(dim=-2) - diag_x  # (batch, n)
    col_sum_L = gram_y.sum(dim=-2) - diag_y  # (batch, n)
    term3 = 2 * (col_sum_K * col_sum_L).sum(dim=-1) / (n - 2)

    main_term = trace_KL + term2 - term3
    denominator = n * (n - 3)
    return main_term / denominator


def hsic_outer(
    grams_x: torch.Tensor,
    grams_y: torch.Tensor,
) -> torch.Tensor:
    if grams_x.dim() != 3 or grams_y.dim() != 3:
        raise ValueError(
            f"hsic_outer requires 3D tensors, got shapes {grams_x.shape} and {grams_y.shape}"
        )

    _, n, m = grams_x.shape
    _, n_y, m_y = grams_y.shape

    if n != m or n_y != m_y:
        raise ValueError("hsic_outer requires square matrices")

    if n != n_y:
        raise ValueError(
            f"Gram matrices must have same sample dimension, got {n} and {n_y}"
        )

    if n <= 3:
        raise ValueError(f"hsic_outer requires n > 3, got n={n}")

    diag_x = torch.diagonal(grams_x, dim1=-2, dim2=-1)  # (n1, n)
    diag_y = torch.diagonal(grams_y, dim1=-2, dim2=-1)  # (n2, n)

    # Term 1: trace_KL[i,j] = sum(grams_x[i] * grams_y[j]) - sum(diag_x[i] * diag_y[j])
    # Use einsum for pairwise element-wise product sum
    trace_full = torch.einsum("aij,bij->ab", grams_x, grams_y)  # (n1, n2)
    trace_diag = torch.einsum("ai,bi->ab", diag_x, diag_y)  # (n1, n2)
    trace_KL = trace_full - trace_diag

    # Term 2: (sum_K[i] * sum_L[j]) / ((n-1)(n-2))
    sum_K = grams_x.sum(dim=(-2, -1)) - diag_x.sum(dim=-1)  # (n1,)
    sum_L = grams_y.sum(dim=(-2, -1)) - diag_y.sum(dim=-1)  # (n2,)
    term2 = torch.outer(sum_K, sum_L) / ((n - 1) * (n - 2))  # (n1, n2)

    # Term 3: 2 * (col_sum_K[i] @ col_sum_L[j]) / (n-2)
    col_sum_K = grams_x.sum(dim=-2) - diag_x  # (n1, n)
    col_sum_L = grams_y.sum(dim=-2) - diag_y  # (n2, n)
    term3 = 2 * torch.mm(col_sum_K, col_sum_L.T) / (n - 2)  # (n1, n2)

    main_term = trace_KL + term2 - term3
    denominator = n * (n - 3)
    return main_term / denominator
