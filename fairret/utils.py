import torch


def safe_div(num: torch.Tensor, denom: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """
    Safely perform elementwise division of two tensors, where an error is raised for division by zero or NaN values if
    the numerator is non-zero. If the numerator is zero, the result is zero.

    Args:
        num: Numerator of shape (N, *) to be divided.
        denom: Denominator of shape (N, *) to divide by.
        eps: Tolerance with which to consider a value's magnitude as zero.

    Returns:
        Elementwise division in shape (N, *).
    """

    if num.isnan().any() or denom.isnan().any():
        raise ValueError("Cannot safely divide due to NaN values in numerator or denominator.")

    zero_num_idx = num.abs() < eps
    zero_denom_idx = denom.abs() < eps
    if zero_denom_idx.any() and (~zero_num_idx[zero_denom_idx]).any():
        raise ZeroDivisionError(f"Division by zero denominator {denom} despite non-zero numerator ({num}).")

    res = torch.zeros_like(num)
    res[~zero_num_idx] = num[~zero_num_idx] / denom[~zero_num_idx]
    return res
