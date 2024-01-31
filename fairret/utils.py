import torch


def safe_div(nom, denom, eps=1e-20):
    if nom.isnan().any() or denom.isnan().any():
        raise ValueError("Cannot safely divide due to NaN values in numerator or denominator.")

    zero_nom_idx = nom.abs() < eps
    zero_denom_idx = denom.abs() < eps
    if zero_denom_idx.any() and (~zero_nom_idx[zero_denom_idx]).any():
        raise ZeroDivisionError(f"Division by zero denominator {denom} despite non-zero numerator ({nom}).")

    res = torch.zeros_like(nom)
    res[~zero_nom_idx] = nom[~zero_nom_idx] / denom[~zero_nom_idx]
    return res
