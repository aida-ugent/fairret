import torch
import tensorflow as tf


def safe_div_tf(num, denom, eps=1e-20):
    if tf.math.reduce_any(tf.math.is_nan(num)) or tf.math.reduce_any(tf.math.is_nan(denom)):
        raise ValueError("Cannot safely divide due to NaN values in numerator or denominator.")

    zero_num_idx = tf.math.abs(num) < eps
    zero_denom_idx = tf.math.abs(denom) < eps
    if tf.math.reduce_any(zero_denom_idx) and tf.math.reduce_any(~zero_num_idx[zero_denom_idx]):
        raise ZeroDivisionError(f"Division by zero denominator {denom} despite non-zero numerator ({num}).")

    return tf.math.divide_no_nan(num, denom)

def safe_div_torch(num, denom, eps=1e-20):
    if num.isnan().any() or denom.isnan().any():
        raise ValueError("Cannot safely divide due to NaN values in numerator or denominator.")

    zero_num_idx = num.abs() < eps
    zero_denom_idx = denom.abs() < eps
    if zero_denom_idx.any() and (~zero_num_idx[zero_denom_idx]).any():
        raise ZeroDivisionError(f"Division by zero denominator {denom} despite non-zero numerator ({num}).")

    res = torch.zeros_like(num)
    res[~zero_num_idx] = num[~zero_num_idx] / denom[~zero_num_idx]
    return res


def safe_div(num, denom, eps=1e-20, torch=True):
    if torch:
        return safe_div_torch(num, denom, eps)
    return safe_div_tf(num, denom, eps)
