import torch


def hsic(
    grams_x: torch.Tensor,
    grams_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grams_x.dim() != 3 or grams_y.dim() != 3:
        raise ValueError(
            f"hsic_all requires 3D tensors, got shapes {grams_x.shape} and {grams_y.shape}"
        )

    _, n, m = grams_x.shape
    _, n_y, m_y = grams_y.shape

    if n != m or n_y != m_y:
        raise ValueError("hsic_all requires square matrices")

    if n != n_y:
        raise ValueError(
            f"Gram matrices must have same sample dimension, got {n} and {n_y}"
        )

    if n <= 3:
        raise ValueError(f"hsic_all requires n > 3, got n={n}")

    # Shared intermediates for grams_x
    diag_x = torch.diagonal(grams_x, dim1=-2, dim2=-1)  # (n1, n)
    sum_x = grams_x.sum(dim=(-2, -1)) - diag_x.sum(dim=-1)  # (n1,)
    col_sum_x = grams_x.sum(dim=-2) - diag_x  # (n1, n)

    # Shared intermediates for grams_y
    diag_y = torch.diagonal(grams_y, dim1=-2, dim2=-1)  # (n2, n)
    sum_y = grams_y.sum(dim=(-2, -1)) - diag_y.sum(dim=-1)  # (n2,)
    col_sum_y = grams_y.sum(dim=-2) - diag_y  # (n2, n)

    denominator = n * (n - 3)
    term2_denom = (n - 1) * (n - 2)
    term3_coeff = 2 / (n - 2)

    # Cross HSIC (hsic_xy)
    trace_xy = torch.einsum("aij,bij->ab", grams_x, grams_y) - torch.einsum(
        "ai,bi->ab", diag_x, diag_y
    )
    term2_xy = torch.outer(sum_x, sum_y) / term2_denom
    term3_xy = term3_coeff * torch.mm(col_sum_x, col_sum_y.T)
    hsic_xy = (trace_xy + term2_xy - term3_xy) / denominator

    # Self HSIC for grams_x (hsic_xx)
    trace_xx = (grams_x * grams_x).sum(dim=(-2, -1)) - (diag_x * diag_x).sum(dim=-1)
    term2_xx = (sum_x * sum_x) / term2_denom
    term3_xx = term3_coeff * (col_sum_x * col_sum_x).sum(dim=-1)
    hsic_xx = (trace_xx + term2_xx - term3_xx) / denominator

    # Self HSIC for grams_y (hsic_yy)
    trace_yy = (grams_y * grams_y).sum(dim=(-2, -1)) - (diag_y * diag_y).sum(dim=-1)
    term2_yy = (sum_y * sum_y) / term2_denom
    term3_yy = term3_coeff * (col_sum_y * col_sum_y).sum(dim=-1)
    hsic_yy = (trace_yy + term2_yy - term3_yy) / denominator

    return hsic_xy, hsic_xx, hsic_yy


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
