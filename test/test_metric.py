import pytest
import torch
import numpy as np

from fairret.statistic import PositivePredictiveValue
from fairret.metric import LinearFractionalParity


@pytest.fixture
def easy_data():
    return (
        torch.tensor([[0.2], [0.6], [0.9], [0.8]]),  # pred (4,)
        torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),  # sens (4, 2)
        torch.tensor([[0.], [1.], [0.], [1.]])  # label (4,)
    )


@pytest.fixture
def expected_predictive_parity():
    return 0.339286


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_predictive_parity(easy_data, expected_predictive_parity, batch_size):
    pred, sens, label = easy_data
    sens_dim = sens.shape[1]
    statistic = PositivePredictiveValue()
    metric = LinearFractionalParity(statistic, sens_dim)

    for i in range(0, pred.shape[0], batch_size):
        metric.update(pred[i:i + batch_size],
                      sens[i:i + batch_size],
                      label[i:i + batch_size])

    assert np.allclose(metric.compute().item(), expected_predictive_parity, atol=1e-4)
