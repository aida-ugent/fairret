import torch

from .base import FairnessLoss
from ..statistic import Statistic


class ViolationLoss(FairnessLoss):
    def __init__(self, stat: Statistic):
        super().__init__()
        self.stat = stat

    def quantify_violation(self, violation):
        raise NotImplementedError

    def forward(self, logit, feat, sens, label, as_logit=True, **kwargs):
        if as_logit:
            pred = torch.sigmoid(logit)
        else:
            pred = logit

        c = self.stat.overall_statistic(pred, feat, label)

        stats = self.stat(pred, feat, sens, label)
        if c.item() == 0.:
            loss = stats.sum()
            return loss
        violation = torch.abs(stats / c - 1)

        loss = self.quantify_violation(violation)
        return loss


class NormLoss(ViolationLoss):
    def __init__(self, stat: Statistic, p=1):
        super().__init__(stat)
        self.p = p

    def quantify_violation(self, violation):
        return torch.linalg.vector_norm(violation, ord=self.p, dim=-1).sum()


class LSELoss(ViolationLoss):
    def quantify_violation(self, violation):
        loss = torch.logsumexp(violation, dim=-1) - torch.log(torch.tensor(violation.shape[-1]))
        return loss.sum()
