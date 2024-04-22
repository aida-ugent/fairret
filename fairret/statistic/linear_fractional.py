import abc
from typing import Any, Tuple, Optional
import torch

from fairret.utils import safe_div
from .base import Statistic


class LinearFractionalStatistic(Statistic):
    """
    Absract base class for a linear-fractional Statistic. This is a Statistic that is computed as the ratio between two
    linear functions over the predictions.

    A LinearFractionalStatistic is constructed in a canonical form by defining the intercept and slope of the
    numerator and denominator linear functions, i.e. the functions
    :func:`~linear_fractional.LinearFractionalStatistic.num_intercept`,
    :func:`~linear_fractional.LinearFractionalStatistic.num_slope`,
    :func:`~linear_fractional.LinearFractionalStatistic.denom_intercept`,
    and :func:`~linear_fractional.LinearFractionalStatistic.denom_slope`. Each subclass must implement these
    functions (using any signature).

    The statistic is then computed as :math:`\\frac{num\\_intercept + num\\_slope * pred}{denom\\_intercept +
    denom\\_slope * pred}`.
    """

    # The following methods violate the Liskov Substitution Principle (LSP) because they are implemented more generally
    # in this base class than is expected in the subclasses. However, it is kept for practical reasons. See 
    # https://github.com/python/mypy/issues/5876

    @abc.abstractmethod
    def num_intercept(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the intercept of the numerator in the canonical form of the linear-fractional statistic.

        Args:
            *args: Any arguments.
            **kwargs: Any keyword arguments.

        Returns:
            torch.Tensor: Shape of the predictions tensor or a shape suitable for broadcasting (see 
            https://pytorch.org/docs/stable/notes/broadcasting.html).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def num_slope(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the slope of the numerator in the canonical form of the linear-fractional statistic. This will be
        multiplied by the prediction tensor.

        Args:
            *args: Any arguments.
            **kwargs: Any keyword arguments.

        Returns:
            torch.Tensor: Shape of the predictions tensor or a shape suitable for broadcasting (see 
            https://pytorch.org/docs/stable/notes/broadcasting.html).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def denom_intercept(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the intercept of the denominator in the canonical form of the linear-fractional statistic.

        Args:
            *args: Any arguments.
            **kwargs: Any keyword arguments.

        Returns:
            torch.Tensor: Shape of the predictions tensor or a shape suitable for broadcasting (see 
            https://pytorch.org/docs/stable/notes/broadcasting.html).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def denom_slope(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the slope of the denominator in the canonical form of the linear-fractional statistic. This will be
        multiplied by the prediction tensor.

        Args:
            *args: Any arguments.
            **kwargs: Any keyword arguments.

        Returns:
            torch.Tensor: Shape of the predictions tensor or a shape suitable for broadcasting (see 
            https://pytorch.org/docs/stable/notes/broadcasting.html).
        """
        raise NotImplementedError

    def num(self, pred: torch.Tensor, sens: Optional[torch.Tensor], *stat_args: Any, **stat_kwargs: Any
            ) -> torch.Tensor:
        """
        Intermediate function to compute the numerator of the linear-fractional statistic.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
                If None, the statistic is computed over all samples, ignoring the sensitive feature.
            *stat_args: Any further arguments used to compute the statistic.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: Shape :math:`(S)`.
        """
        intercept = self.num_intercept(*stat_args, **stat_kwargs)
        slope = self.num_slope(*stat_args, **stat_kwargs)
        return self.__compute_num_or_denom(pred, sens, intercept, slope)

    def denom(self, pred: torch.Tensor, sens: Optional[torch.Tensor], *stat_args: Any, **stat_kwargs: Any
              ) -> torch.Tensor:
        """
        Intermediate function to compute the denominator of the linear-fractional statistic.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
                If None, the statistic is computed over all samples, ignoring the sensitive feature.
            *stat_args: Any further arguments used to compute the statistic.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: Shape :math:`(S)`.
        """

        intercept = self.denom_intercept(*stat_args, **stat_kwargs)
        slope = self.denom_slope(*stat_args, **stat_kwargs)
        return self.__compute_num_or_denom(pred, sens, intercept, slope)

    @staticmethod
    def __compute_num_or_denom(pred: torch.Tensor, sens: Optional[torch.Tensor], intercept: torch.Tensor,
                               slope: torch.Tensor) -> torch.Tensor:
        """
        Shorthand function to deduplicate the num and denom functions, given the intercept and slope functions.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
                If None, the statistic is computed over all samples, ignoring the sensitive feature.
            intercept (torch.Tensor): The intercept of the linear function.
            slope (torch.Tensor): The slope of the linear function.
        """
        if isinstance(slope, float) or len(slope.shape) == 0:
            linear_expression = intercept + slope * pred
        elif len(slope.shape) >= 2:
            linear_expression = intercept + torch.einsum('n...y,ny->n...y', slope, pred)
        else:
            raise ValueError("The slope should be a scalar or a tensor with at least as many dimensions as the 'pred' "
                             "tensor.")

        if sens is None:
            return linear_expression.sum(dim=0)
        else:
            return torch.einsum('n...y,ns->n...s', linear_expression, sens).sum(dim=0)

    def forward(self, pred: torch.Tensor, sens: Optional[torch.Tensor], *stat_args: Any, **stat_kwargs: Any
                ) -> torch.Tensor:
        num = self.num(pred, sens, *stat_args, **stat_kwargs)
        denom = self.denom(pred, sens, *stat_args, **stat_kwargs)
        return safe_div(num, denom)

    def overall_statistic(self, pred: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> torch.Tensor:
        """
        Compute the overall statistic for the predictions. This is the value of the statistic when the sensitive feature
        is ignored, i.e. None. The predictions are typically considered fair (with respect to the statistic) if and only
        if the statistic computed for every sensitive feature equals this value.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            *stat_args: Any further arguments used to compute the statistic.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: The overall statistic as a scalar tensor.
        """
        return self.forward(pred, None, *stat_args, **stat_kwargs)

    def fixed_constraint(self, fix_value: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reformulate the fairness definition for this LinearFractionalStatistic as a linear constraint.

        This is done by fixing the intended values of the statistic for every sensitive feature to a given value
        `fix_value`. The linear expressions that make up the numerator and denominator are then combined into a single
        linear constraint.

        Args:
            fix_value (torch.Tensor): The intended value of the statistic for every sensitive feature. Typically, this
                is the overall statistic.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: Any further arguments used to compute the statistic.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The intercept and slope of the linear constraint defined as
            :math:`intercept + slope * pred = 0`.
        """
        num_intercept = self.num_intercept(*stat_args, **stat_kwargs)
        denom_intercept = self.denom_intercept(*stat_args, **stat_kwargs)
        if isinstance(denom_intercept, float) or len(denom_intercept.shape) == 0:
            intercept = num_intercept - denom_intercept * fix_value
        else:
            intercept = num_intercept - torch.einsum('n...y,...y->n...y', denom_intercept, fix_value)
        intercept = torch.einsum('n...y,ns->n...s', intercept, sens).flatten(start_dim=1)

        num_slope = self.num_slope(*stat_args, **stat_kwargs)
        denom_slope = self.denom_slope(*stat_args, **stat_kwargs)
        if isinstance(denom_slope, float) or len(denom_slope.shape) == 0:
            slope = num_slope - denom_slope * fix_value
        else:
            slope = num_slope - torch.einsum('n...y,...y->...', denom_slope, fix_value)
        slope = torch.einsum('n...y,ns->n...s', slope, sens).flatten(start_dim=1)
        return intercept, slope


class PositiveRate(LinearFractionalStatistic):
    """
    PositiveRate is a LinearFractionalStatistic that computes the average rate at which positive predictions are made in
    binary classification.

    Formulated as a probability, it computes :math:`P(\\hat{Y} = 1 | S)` for categorical sensitive features :math:`S`,
    with the predicted label :math:`\\hat{Y}` sampled according to the model's (probabilistic) prediction.

    The functions of its canonical form take no arguments.
    """

    def num_intercept(self) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.

    def num_slope(self) -> torch.Tensor:
        """
        :math:`= 1`
        """
        return 1.

    def denom_intercept(self) -> torch.Tensor:
        """
        :math:`= 1`
        """
        return 1.

    def denom_slope(self) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.


class TruePositiveRate(LinearFractionalStatistic):
    """
    TruePositiveRate is a LinearFractionalStatistic that computes the average rate at which positives are actually
    predicted as positives, also known as the recall.

    Formulated as a probability, it computes :math:`P(\\hat{Y} = 1 | Y = 1, S)` for categorical sensitive features
    :math:`S` and only for samples where the target label :math:`Y` is positive, with the predicted label
    :math:`\\hat{Y}` sampled according to the model's (probabilistic) prediction.

    The functions of its canonical form require that the tensor of target labels is provided with the same shape as the
    predictions.
    """

    def num_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.

    def num_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= Y`
        """
        return label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= Y`
        """
        return label

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.


class FalsePositiveRate(LinearFractionalStatistic):
    """
    FalsePositiveRate is a LinearFractionalStatistic that computes the average rate at which negatives are actually
    predicted as positives.

    Formulated as a probability, it computes :math:`P(\\hat{Y} = 1 | Y = 0, S)` for categorical sensitive features
    :math:`S` and only for samples where the target label :math:`Y` is negative, with the predicted label
    :math:`\\hat{Y}` sampled according to the model's (probabilistic) prediction.

    The functions of its canonical form require that the tensor of target labels is provided with the same shape as the
    predictions.
    """

    def num_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.

    def num_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1 - Y`
        """
        return 1 - label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1 - Y`
        """
        return 1 - label

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.


class PositivePredictiveValue(LinearFractionalStatistic):
    """
    PositivePredictiveValue is a LinearFractionalStatistic that computes the average rate at which the predicted
    positives were actually labeled positive, also known as the precision.

    Formulated as a probability, it computes :math:`P(Y = 1 | \\hat{Y} = 1, S)` for categorical sensitive features
    :math:`S` and only for samples where the predicted label (sampled according to the model's (probabilistic)
    prediction) is positive, with :math:`Y` the target label.

    The functions of its canonical form require that the tensor of target labels is provided with the same shape as the
    predictions.
    """

    def num_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.

    def num_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= Y`
        """
        return label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1`
        """
        return 1.


class FalseOmissionRate(LinearFractionalStatistic):
    """
    FalseOmissionRate is a LinearFractionalStatistic that computes the average rate at which the predicted
    negatives were actually labeled positive, also known as the precision.

    Formulated as a probability, it computes :math:`P(Y = 1 | \\hat{Y} = 0, S)` for categorical sensitive features
    :math:`S` and only for samples where the predicted label (sampled according to the model's (probabilistic)
    prediction) is negative, with :math:`Y` the target label.

    The functions of its canonical form require that the tensor of target labels is provided with the same shape as the
    predictions.
    """

    def num_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= Y`
        """
        return label

    def num_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1 - Y`
        """
        return -label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1`
        """
        return 1.

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= -1`
        """
        return -1.


class Accuracy(LinearFractionalStatistic):
    """
    Accuracy is a LinearFractionalStatistic that computes the average rate at which predictions match the actual target
    labels.

    Formulated as a probability, it computes :math:`P(\\hat{Y} = Y | S)` for categorical sensitive features :math:`S`,
    with the predicted label :math:`\\hat{Y}` sampled according to the model's (probabilistic) prediction and with
    :math:`Y` the target label.

    The functions of its canonical form require that the tensor of target labels is provided with the same shape as the
    predictions.
    """

    def num_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1 - Y`
        """
        return 1 - label

    def num_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 2Y - 1`
        """
        return 2 * label - 1

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1`
        """
        return 1.

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.


class FalseNegativeFalsePositiveFraction(LinearFractionalStatistic):
    """
    FalseNegativeFalsePositiveFraction is a LinearFractionalStatistic that computes the ratio between false negatives
    and false positives.

    The statistic cannot be formulated as a single probability.

    The functions of its canonical form require that the tensor of target labels is provided with the same shape as the
    predictions.
    """

    def num_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= Y`
        """
        return label

    def num_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= -Y`
        """
        return -label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0.

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1 - Y`
        """
        return 1 - label


class FScore(LinearFractionalStatistic):
    """
    FScore is a LinearFractionalStatistic that computes the :math:`F_β`-score, the weighted mean of precision and
    recall. The :math:`F_1`-score is the harmonic mean of precision and recall.

    The statistic cannot be formulated as a single probability.

    The functions of its canonical form require that the tensor of target labels is provided with the same shape as the
    predictions.
    """

    def __init__(self, beta: float = 1.):
        """
        Args:
            beta (float): The weight of recall in the harmonic mean. Default is 1.
        """
        super().__init__()
        self.beta = beta

    def num_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 0`
        """
        return 0

    def num_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1 + β^2`
        """
        return (1 + self.beta ** 2) * label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= β^2Y`
        """
        return (self.beta ** 2) * label

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        """
        :math:`= 1`
        """
        return 1.


class StackedLinearFractionalStatistic(LinearFractionalStatistic):
    """
    A vector-valued LinearFractionalStatistic that combines the outputs of K LinearFractionalStatistics into a single
    statistic with output (K, S) by stacking all outputs in the second-to-last dimension (`dim=-2`).
    """

    def __init__(self, *statistics: LinearFractionalStatistic):
        """
        Args:
            *statistics: The LinearFractionalStatistics to be stacked.
        """

        super().__init__()

        if len(statistics) == 0:
            raise ValueError("At least one statistic should be provided.")
        if any(not isinstance(stat, LinearFractionalStatistic) for stat in statistics):
            raise ValueError("All statistics should be instances of LinearFractionalStatistic.")

        self.statistics = torch.nn.ModuleList(statistics)

    def num_intercept(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Stack the numerator intercepts of all `statistics`.
        """
        return self.__stack('num_intercept', *args, **kwargs)

    def num_slope(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Stack the numerator slopes of all `statistics`.
        """
        return self.__stack('num_slope', *args, **kwargs)

    def denom_intercept(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Stack the denominator intercepts of all `statistics`.
        """
        return self.__stack('denom_intercept', *args, **kwargs)

    def denom_slope(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Stack the denominator slope of all `statistics`.
        """
        return self.__stack('denom_slope', *args, **kwargs)

    @staticmethod
    def __ensure_shape(val: Any) -> torch.Tensor:
        """
        Ensure `val` is a tensor that is not a singleton (0-dimensional). If it is, make it have two dimensions
        (a batch dimension followed by a dimension of sensitive features).
        Args:
            val (torch.Tensor): The tensor to ensure the shape of.
        """
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if len(val.shape) == 0:
            val = val.view(1, 1)
        return val

    def __stack(self, method: str, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Stack the outputs of a method of all `statistics`.
        Args:
            method (str): The method to call on all `statistics`.
            *args (Any): Any arguments to pass to the method.
            **kwargs (Any): Any keyword arguments to pass to the method.

        Returns:
            torch.Tensor: The stacked outputs.
        """
        return torch.stack([
            self.__ensure_shape(getattr(stat, method)(*args, **kwargs))
            for stat in self.statistics
        ], dim=-2)
