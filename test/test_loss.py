import pytest
import torch

from fairret.statistic import (PositiveRate, Accuracy, StackedLinearFractionalStatistic, TruePositiveRate,
                               FalsePositiveRate)
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


@pytest.mark.parametrize("loss_cls", [NormLoss, LSELoss, KLProjectionLoss, JensenShannonProjectionLoss,
                                      TotalVariationProjectionLoss, SquaredEuclideanProjectionLoss])
def test_loss_positive_rate(loss_cls, easy_data, net):
    feat, sens, _ = easy_data
    feat_backup = feat.clone()
    sens_backup = sens.clone()

    fairret = loss_cls(PositiveRate())
    nb_tries = 5

    # Calculate loss multiple times as some losses are stateful
    for attempt in range(nb_tries):
        if attempt % 2 == 1:
            # Vary the amount of data to check if the loss can handle it
            batch_feat = feat[:2]
            batch_sens = sens[:2]
        else:
            batch_feat = feat
            batch_sens = sens

        logit = net(batch_feat)
        loss = fairret(logit, batch_sens)
        assert loss.item() >= 0  # fairret should be nonnegative
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None
            assert torch.all(torch.isfinite(p.grad))
            p.grad.zero_()

    # Check that the data was not modified
    assert torch.all(feat == feat_backup)
    assert torch.all(sens == sens_backup)


@pytest.mark.parametrize("loss_cls", [NormLoss, LSELoss, KLProjectionLoss, JensenShannonProjectionLoss,
                                      TotalVariationProjectionLoss, SquaredEuclideanProjectionLoss])
def test_loss_accuracy(loss_cls, easy_data, net):
    feat, sens, label = easy_data
    feat_backup = feat.clone()
    sens_backup = sens.clone()
    label_backup = label.clone()

    fairret = loss_cls(Accuracy())
    nb_tries = 5

    # Calculate loss multiple times as some losses are stateful
    for attempt in range(nb_tries):
        if attempt % 2 == 1:
            # Vary the amount of data to check if the loss can handle it
            batch_feat = feat[:2]
            batch_sens = sens[:2]
            batch_label = label[:2]
        else:
            batch_feat = feat
            batch_sens = sens
            batch_label = label
        logit = net(batch_feat)
        loss = fairret(logit, batch_sens, batch_label)
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


@pytest.mark.parametrize("loss_cls", [NormLoss, LSELoss, KLProjectionLoss, JensenShannonProjectionLoss,
                                      TotalVariationProjectionLoss, SquaredEuclideanProjectionLoss])
def test_loss_equalised_odds(loss_cls, easy_data, net):
    feat, sens, label = easy_data
    feat_backup = feat.clone()
    sens_backup = sens.clone()
    label_backup = label.clone()

    stat = StackedLinearFractionalStatistic(TruePositiveRate(), FalsePositiveRate())
    fairret = loss_cls(stat)
    nb_tries = 5

    # Calculate loss multiple times as some losses are stateful
    for _ in range(nb_tries):
        logit = net(feat)
        loss = fairret(logit, sens, label)
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
