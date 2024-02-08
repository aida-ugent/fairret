import torch
import torchmetrics

from .statistic import Statistic, PositiveRate, TruePositiveRate, FalsePositiveRate, PositivePredictiveValue, \
    FalseOmissionRate, Accuracy, FalseNegativeFalsePositiveFraction
from .utils import safe_div


class FairMetric(torchmetrics.Metric):
    def update(self, pred, feat, sens, label):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class LinearFractionalParity(FairMetric):
    # TODO: write test
    def __init__(self, statistic: Statistic, sens_dim=None, normalized=True, **kwargs):
        super().__init__(**kwargs)
        self.stat = statistic
        self.normalized = normalized

        self.add_state('nom', default=torch.zeros(sens_dim, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('denom', default=torch.zeros(sens_dim, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('overall_nom', default=torch.zeros(1, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('overall_denom', default=torch.zeros(1, dtype=torch.float), dist_reduce_fx='sum')

    def update(self, pred, feat, sens, label):
        if self.simple_sens_cols:
            sens = sens[:, self.simple_sens_cols]

        self.nom += self.stat.nom(pred, feat, sens, label)
        self.denom += self.stat.denom(pred, feat, sens, label)
        self.overall_nom += self.stat.nom(pred, feat, 1., label)
        self.overall_denom += self.stat.denom(pred, feat, 1., label)

    def compute(self):
        stats = safe_div(self.nom, self.denom)
        overall_stat = safe_div(self.overall_nom, self.overall_denom)

        if self.normalized:
            return (torch.abs(stats / overall_stat - 1)).max()
        else:
            return (torch.abs(stats - overall_stat)).max()


class DemographicParity(LinearFractionalParity):
    def __init__(self, **kwargs):
        super().__init__(PositiveRate(), **kwargs)


class EqualizedOpportunity(LinearFractionalParity):
    def __init__(self, **kwargs):
        super().__init__(TruePositiveRate(), **kwargs)


class PredictiveEquality(LinearFractionalParity):
    def __init__(self, **kwargs):
        super().__init__(FalsePositiveRate(), **kwargs)


class PredictiveParity(LinearFractionalParity):
    def __init__(self, **kwargs):
        super().__init__(PositivePredictiveValue(), **kwargs)


class FalseOmissionParity(LinearFractionalParity):
    def __init__(self, **kwargs):
        super().__init__(FalseOmissionRate(), **kwargs)


class OverallAccuracyEquality(LinearFractionalParity):
    def __init__(self, **kwargs):
        super().__init__(Accuracy(), **kwargs)


class TreatmentEquality(LinearFractionalParity):
    def __init__(self, **kwargs):
        super().__init__(FalseNegativeFalsePositiveFraction(), **kwargs)
