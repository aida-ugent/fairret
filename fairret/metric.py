import torch
import torchmetrics

from .statistic import LinearFractionalStatistic
from .utils import safe_div


def abs_diff(stats, overall_stat):
    return (torch.abs(stats - overall_stat)).max()


def abs_diff_norm(stats, overall_stat):
    return (torch.abs(stats / overall_stat - 1)).max()


class Parity(torchmetrics.Metric):
    def update(self, pred, sens, stat_args):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class LinearFractionalParity(Parity):
    # TODO: warn about compute groups.
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, statistic: LinearFractionalStatistic, sens_dim: int, gap_fn: callable = abs_diff_norm, **kwargs):
        super().__init__(**kwargs)
        self.stat = statistic
        self.sens_dim = sens_dim
        self.gap_fn = gap_fn

        self.add_state('nom', default=torch.zeros(self.sens_dim, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('denom', default=torch.zeros(self.sens_dim, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('overall_nom', default=torch.zeros(1, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('overall_denom', default=torch.zeros(1, dtype=torch.float), dist_reduce_fx='sum')

    def update(self, pred, sens, *stat_args, **stat_kwargs):
        if sens.shape[1] != self.sens_dim:
            raise ValueError(f"Expected sens to have shape (N, {self.sens_dim}), got {sens.shape}")

        self.nom += self.stat.nom(pred, sens, *stat_args, **stat_kwargs)
        self.denom += self.stat.denom(pred, sens, *stat_args, **stat_kwargs)
        self.overall_nom += self.stat.nom(pred, 1., *stat_args, **stat_kwargs)
        self.overall_denom += self.stat.denom(pred, 1., *stat_args, **stat_kwargs)

    def compute(self):
        stats = safe_div(self.nom, self.denom)
        overall_stat = safe_div(self.overall_nom, self.overall_denom)
        return self.gap_fn(stats, overall_stat)
