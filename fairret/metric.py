from typing import Any, Callable
import torch

try:
    import torchmetrics
except ImportError:
    raise ImportError("The 'torchmetrics' package must be installed to use the LinearFractionalParity metric.")

from .statistic import LinearFractionalStatistic
from .utils import safe_div


def gap_abs_max(vals: torch.Tensor, target_val: float) -> float:
    """
    Compute the gaps between the vals and the target_val, and return the absolute maximum (the :math:`\\Vert \\cdot
    \\Vert_\\infty` norm).

    Args:
        vals (torch.Tensor): The values to compare.
        target_val (float): The target value.

    Returns:
        float: The maximal gap.
    """

    return (torch.abs(vals - target_val)).max()


def gap_relative_abs_max(vals: torch.Tensor, target_val: float) -> float:
    """
    Compute the relative gaps between the vals and the target_val, and return the absolute maximum (the :math:`\\Vert
    \\cdot \\Vert_\\infty` norm).

    Args:
        vals (torch.Tensor): The values to compare.
        target_val (float): The target value.

    Returns:
        float: The maximal gap.
    """
    return (torch.abs(vals / target_val - 1)).max()


class LinearFractionalParity(torchmetrics.Metric):
    """
    Metric that assesses the fairness of a model's predictions by comparing the gaps between the provided
    LinearFractionalStatistic for every sensitive feature.

    The metric maintains two pairs of running sums: one for the statistic for every sensitive feature, and one for the
    overall statistic. Each pair of running sums consists of the nominator and the denominator for those statistics.
    Observations are added to these sums by calling the `update` method. The final fairness gap is computed by calling
    the `compute` method, which also resets the internal state of the metric.

    The class is implemented as a subclass of torchmetrics.Metric, so the `torchmetrics` package is required.

    Warning:
        It is advised not to mix LinearFractionalParity metrics with different statistics in a single
        torchmetrics.MetricCollection with `compute_groups=True`, as this can lead to hard-to-debug errors.
    """

    # TODO: warn about compute groups.
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self,
                 statistic: LinearFractionalStatistic,
                 sens_dim: int,
                 gap_fn: Callable[[torch.Tensor, float], float] = gap_relative_abs_max,
                 **torchmetrics_kwargs: Any):
        """
        Args:
            statistic (LinearFractionalStatistic): the LinearFractionalStatistic that should be evaluated.
            sens_dim (int): the number of sensitive features.
            gap_fn (Callable[[torch.Tensor, float], float]): the function that computes the gaps between the statistic
                for every sensitive feature and the overall statistic. The default is the absolute maximum of the
                relative gaps.
            **torchmetrics_kwargs: Any additional keyword arguments that should be passed to torchmetrics.Metric.
        """

        super().__init__(**torchmetrics_kwargs)
        self.stat = statistic
        self.sens_dim = sens_dim
        self.gap_fn = gap_fn

        self.add_state('nom', default=torch.zeros(self.sens_dim, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('denom', default=torch.zeros(self.sens_dim, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('overall_nom', default=torch.zeros(1, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('overall_denom', default=torch.zeros(1, dtype=torch.float), dist_reduce_fx='sum')

    def update(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> None:
        """
        Update the running sums for the nominator and denominator of the groupwise and overall statistics with a new
        batch of predictions and sensitive features.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, C)` with `C` the number of classes. For binary
                classification or regression, it can be :math:`C = 1`.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: Any further arguments used to compute the statistic.
            **stat_kwargs: Any keyword arguments used to compute the statistic.
        """

        if sens.shape[1] != self.sens_dim:
            raise ValueError(f"Expected sens to have shape (N, {self.sens_dim}), got {sens.shape}")

        self.nom += self.stat.nom(pred, sens, *stat_args, **stat_kwargs)
        self.denom += self.stat.denom(pred, sens, *stat_args, **stat_kwargs)
        self.overall_nom += self.stat.nom(pred, 1., *stat_args, **stat_kwargs)
        self.overall_denom += self.stat.denom(pred, 1., *stat_args, **stat_kwargs)

    def compute(self) -> float:
        """
        Divide the running sums of the nominator and denominator of the groupwise and overall statistics and compute the
        final gaps between the groupwise and overall statistics, according to the `gap_fn`.

        The internal state of the metric is reset after calling this method.

        Returns:
            float: The final fairness gap.
        """

        stats = safe_div(self.nom, self.denom)
        overall_stat = safe_div(self.overall_nom, self.overall_denom)
        return self.gap_fn(stats, overall_stat)
