import pytest
import torch

from fairret.statistic import (PositiveRate, TruePositiveRate, FalsePositiveRate, PositivePredictiveValue,
                               FalseOmissionRate, Accuracy, FalseNegativeFalsePositiveFraction)


@pytest.fixture
def easy_data():
    return (
        torch.tensor([[0.2], [0.6], [0.9], [0.8]]),  # pred (4,)
        torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),  # sens (4, 2)
        torch.tensor([[0.], [1.], [0.], [1.]])  # label (4,)
    )


@pytest.mark.parametrize("stat_vals", [
    (PositiveRate, [0.4, 0.85]),
    (TruePositiveRate, [0.6, 0.8]),
    (FalsePositiveRate, [0.2, 0.9]),
    (PositivePredictiveValue, [0.75, 0.47]),
    (FalseOmissionRate, [0.33, 0.67]),
    (Accuracy, [0.7, 0.45]),
    (FalseNegativeFalsePositiveFraction, [2, 0.22])
])
# TODO add different data test cases.
def test_statistic(easy_data, stat_vals):
    statistic, expected = stat_vals
    stat = statistic()
    pred, sens, label = easy_data

    if statistic == PositiveRate:
        computed_stat = stat(pred, sens)
    else:
        computed_stat = stat(pred, sens, label)
    assert torch.allclose(computed_stat, torch.tensor(expected), atol=1e-2)
