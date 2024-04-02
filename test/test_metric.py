import pytest
import torch
import numpy as np

from fairret.statistic import (PositivePredictiveValue, StackedLinearFractionalStatistic, TruePositiveRate,
                               FalsePositiveRate)
from fairret.metric import LinearFractionalParity


@pytest.fixture
def easy_data():
    return (
        torch.tensor([[0.2], [0.6], [0.9], [0.8]]),  # pred (4,)
        torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),  # sens (4, 2)
        torch.tensor([[0.], [1.], [0.], [1.]])  # label (4,)
    )


@pytest.mark.parametrize("stat_vals", [
    (PositivePredictiveValue, 0.339286),
    (StackedLinearFractionalStatistic, [0.142857, 0.636364])
])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_metric(easy_data, stat_vals, batch_size):
    pred, sens, label = easy_data
    statistic_cls, expected_metric_vals = stat_vals
    sens_dim = sens.shape[1]

    if statistic_cls == StackedLinearFractionalStatistic:
        statistic = statistic_cls(TruePositiveRate(), FalsePositiveRate())
        stat_shape = (2, sens_dim)
    else:
        statistic = statistic_cls()
        stat_shape = sens_dim
    metric = LinearFractionalParity(statistic, stat_shape)

    for i in range(0, pred.shape[0], batch_size):
        metric.update(pred[i:i + batch_size],
                      sens[i:i + batch_size],
                      label[i:i + batch_size])

    assert np.allclose(metric.compute(), np.array(expected_metric_vals), atol=1e-4)
