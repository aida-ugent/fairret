import pytest
import torch

from fairret.statistic import Accuracy
from fairret.loss import (NormLoss, LSELoss, KLProjectionLoss, JensenShannonProjectionLoss,
                          TotalVariationProjectionLoss, SquaredEuclideanProjectionLoss)


@pytest.fixture
def easy_data():
    return (
        torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]),  # feat (4, 2)
        torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]),  # sens (4, 2)
        torch.tensor([[0.], [1.], [0.], [1.]])  # label (4,)
    )


@pytest.fixture
def net():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1)
    )


@pytest.fixture
def stat():
    return Accuracy()


@pytest.mark.parametrize("loss_cls", [NormLoss, LSELoss, KLProjectionLoss, JensenShannonProjectionLoss,
                                      TotalVariationProjectionLoss, SquaredEuclideanProjectionLoss])
def test_loss(loss_cls, easy_data, net, stat):
    feat, sens, label = easy_data
    feat_backup = feat.clone()
    sens_backup = sens.clone()
    label_backup = label.clone()

    fairret = loss_cls(stat)
    nb_tries = 5

    # Calculate loss multiple times as some losses are stateful
    for _ in range(nb_tries):
        logit = net(feat)
        loss = fairret(logit, feat, sens, label)
        assert loss.item() >= 0  # fairret should be nonnegative
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None
            assert torch.all(torch.isfinite(p.grad))
            p.grad.zero_()

    # Check that the data was not modified
    assert torch.all(feat == feat_backup)
    assert torch.all(sens == sens_backup)
    assert torch.all(label == label_backup)
