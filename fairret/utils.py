import torch


def safe_div(num, denom, eps=1e-20):
    if num.isnan().any() or denom.isnan().any():
        raise ValueError("Cannot safely divide due to NaN values in numerator or denominator.")

    zero_num_idx = num.abs() < eps
    zero_denom_idx = denom.abs() < eps
    if zero_denom_idx.any() and (~zero_num_idx[zero_denom_idx]).any():
        raise ZeroDivisionError(f"Division by zero denominator {denom} despite non-zero numerator ({num}).")

    res = torch.zeros_like(num)
    res[~zero_num_idx] = num[~zero_num_idx] / denom[~zero_num_idx]
    return res
