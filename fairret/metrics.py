import torch
import torchmetrics

from .statistic import *
from .utils import safe_div


class FairMetric(torchmetrics.Metric):
    def update(self, pred, feat, sens, label):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class LinearFractionalParity(FairMetric):
    def __init__(self, statistic: Statistic, sens_dim=None, simple_sens_cols=None, normalized=True, **kwargs):
        super().__init__(**kwargs)
        self.stat = statistic
        self.simple_sens_cols = simple_sens_cols
        self.normalized = normalized

        if simple_sens_cols is not None:
            sens_dim = len(simple_sens_cols)

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


class FairMetricCollection(torchmetrics.MetricCollection):
    __metric_by_name = {
        'dp': DemographicParity,
        'eo': EqualizedOpportunity,
        'pe': PredictiveEquality,
        'pp': PredictiveParity,
        'fop': FalseOmissionParity,
        'oa': OverallAccuracyEquality,
        'te': TreatmentEquality
    }

    def __init__(self, metrics=None, prefix=None, simple_sens_cols=None, **metric_kwargs):
        if metrics is None:
            metrics = self.__metric_by_name.keys()
        metrics_dict = {name: self.__metric_by_name[name](**metric_kwargs) for name in metrics}
        if simple_sens_cols is not None:
            metrics_dict |= {f"{name}_simp": self.__metric_by_name[name](
                **metric_kwargs, simple_sens_cols=simple_sens_cols
            ) for name in metrics}

        # Very weird bugs happen for compute_groups=True
        super().__init__(metrics_dict, prefix=prefix, compute_groups=False)
