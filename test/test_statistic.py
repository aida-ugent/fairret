import pytest
import torch

from fairret.statistic import (PositiveRate, TruePositiveRate, FalsePositiveRate, PositivePredictiveValue,
                               FalseOmissionRate, Accuracy, FalseNegativeFalsePositiveFraction, FScore,
                               StackedLinearFractionalStatistic)


@pytest.fixture
def easy_data():
    return (
        torch.tensor([[0.2], [0.6], [0.9], [0.8]]),  # pred (4,)
        torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),  # sens (4, 2)
        torch.tensor([[0.], [1.], [0.], [1.]])  # label (4,)
    )


@pytest.mark.parametrize("stat_vals", [
    (PositiveRate(), [0.4, 0.85]),
    (TruePositiveRate(), [0.6, 0.8]),
    (FalsePositiveRate(), [0.2, 0.9]),
    (PositivePredictiveValue(), [0.75, 0.47]),
    (FalseOmissionRate(), [0.33, 0.67]),
    (Accuracy(), [0.7, 0.45]),
    (FalseNegativeFalsePositiveFraction(), [2, 0.22]),
    (FScore(), [0.67, 0.59]),
    (FScore(beta=2), [0.63, 0.70])
])
# TODO add different data test cases.
def test_statistic(easy_data, stat_vals):
    statistic, expected = stat_vals
    pred, sens, label = easy_data

    if isinstance(statistic, PositiveRate):
        computed_stat = statistic(pred, sens)
    else:
        computed_stat = statistic(pred, sens, label)
    assert torch.allclose(computed_stat, torch.tensor(expected), atol=1e-2)


def test_statistic_tensor(easy_data):
    equalised_odds = StackedLinearFractionalStatistic(TruePositiveRate(), FalsePositiveRate())

    pred, sens, label = easy_data
    combined_stat = equalised_odds(pred, sens, label)
    tpr = TruePositiveRate()(pred, sens, label)
    fpr = FalsePositiveRate()(pred, sens, label)

    assert torch.allclose(combined_stat, torch.stack([tpr, fpr]), atol=1e-2)
