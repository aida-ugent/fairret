from typing import Any, Callable, Tuple, Union
import torch

try:
    import torchmetrics
except ImportError:
    raise ImportError("The 'torchmetrics' package must be installed to use the LinearFractionalParity metric. Please "
                      "install it using `pip install torchmetrics`.")

from fairret.statistic import LinearFractionalStatistic
from fairret.utils import safe_div


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

    return torch.amax(torch.abs(vals - target_val), dim=-1)


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
    return torch.amax(torch.abs(vals / target_val - 1), dim=-1)


class LinearFractionalParity(torchmetrics.Metric):
    """
    Metric that assesses the fairness of a model's predictions by comparing the gaps between the provided
    :py:class:`~fairret.statistic.linear_fractional.LinearFractionalStatistic` for every sensitive feature.

    The metric maintains two pairs of running sums: one for the statistic for every sensitive feature, and one for the
    overall statistic. Each pair of running sums consists of the numerator and the denominator for those statistics.
    Observations are added to these sums by calling the :py:meth:`~fairret.metric.LinearFractionalParity.update` method.
    The final fairness gap is computed by calling the :py:meth:`~fairret.metric.LinearFractionalParity.compute` method.

    The class is implemented as a subclass of :py:class:`torchmetrics.Metric`, so the :py:mod:`torchmetrics` package is
    required.

    Warning:
        A separate :py:meth:`~torchmetrics.Metric.reset()` call is required to reset the internal state of the metric
        between epochs.
    Warning:
        It is advised not to mix metrics of this class with different statistics in a single
        :py:class:`torchmetrics.MetricCollection` with `compute_groups=True`, as this can lead to hard-to-debug errors.
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self,
                 statistic: LinearFractionalStatistic,
                 stat_shape: Union[int, Tuple[int]],
                 gap_fn: Callable[[torch.Tensor, float], float] = gap_relative_abs_max,
                 **torchmetrics_kwargs: Any):
        """
        Args:
            statistic (LinearFractionalStatistic): the LinearFractionalStatistic that should be evaluated.
            stat_shape (Union[int, Tuple[int]]): the shape of the statistic, excluding the batch dimension. For example,
                a single statistic computed for every sensitive feature would have a shape of `(S,)` with `S` the number
                of sensitive features. If the statistic is a stacked statistic, the shape should be `(K, S)` with `K`
                the number of statistics in the stack.
            gap_fn (Callable[[torch.Tensor, float], float]): the function that computes the gaps between the statistic
                for every sensitive feature and the overall statistic. The default is the absolute maximum of the
                relative gaps.
            **torchmetrics_kwargs: Any additional keyword arguments that should be passed to torchmetrics.Metric.
        """

        super().__init__(**torchmetrics_kwargs)
        self.stat = statistic
        self.stat_shape = (stat_shape,) if isinstance(stat_shape, int) else stat_shape
        self.gap_fn = gap_fn

        self.add_state('num', default=torch.zeros(self.stat_shape, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('denom', default=torch.zeros(self.stat_shape, dtype=torch.float), dist_reduce_fx='sum')

        overall_shape = (*(self.stat_shape[1:]), 1)
        self.add_state('overall_num', default=torch.zeros(overall_shape, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('overall_denom', default=torch.zeros(overall_shape, dtype=torch.float), dist_reduce_fx='sum')

    def update(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> None:
        """
        Update the running sums for the numerator and denominator of the groupwise and overall statistics with a new
        batch of predictions and sensitive features.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: All arguments used by the statistic that this metric computes.
            **stat_kwargs: All keyword arguments used by the statistic that this metric computes.
        """

        if sens.shape[1] != self.stat_shape[-1]:
            raise ValueError(f"Expected sens to have shape (N, {self.stat_shape[-1]}, got {sens.shape}")

        self.num += self.stat.num(pred, sens, *stat_args, **stat_kwargs)
        self.denom += self.stat.denom(pred, sens, *stat_args, **stat_kwargs)
        self.overall_num += self.stat.num(pred, None, *stat_args, **stat_kwargs)
        self.overall_denom += self.stat.denom(pred, None, *stat_args, **stat_kwargs)

    def compute(self) -> float:
        """
        Divide the running sums of the numerator and denominator of the groupwise and overall statistics and compute the
        final gaps between the groupwise and overall statistics, according to the `gap_fn`.

        Warning:
            This does NOT reset the internal state of the metric. A separate `.reset()` call is required to do so.

        Returns:
            float: The final fairness gap.
        """

        stats = safe_div(self.num, self.denom)
        overall_stat = safe_div(self.overall_num, self.overall_denom)
        return self.gap_fn(stats, overall_stat)
