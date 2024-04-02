import abc
from typing import Any, Tuple, Optional
import torch

from utils import safe_div
from .base import Statistic


class LinearFractionalStatistic(Statistic):
    """
    Absract base class for a linear-fractional Statistic. This is a Statistic that is computed as the ratio between two
    linear functions over the predictions.

    A LinearFractionalStatistic is constructed in a canonical form by defining the intercept and slope of the numerator
    and denominator linear functions, i.e. the functions nom_intercept, nom_slope, denom_intercept, and denom_slope.
    Each subclass must implement these functions (using any signature).

    The statistic is then computed as :math:`\\frac{nom\\_intercept + nom\\_slope * pred}{denom\\_intercept +
    denom\\_slope * pred}`.
    """

    # The following methods violate the Liskov Substitution Principle (LSP) because they are implemented more generally
    # in this base class than is expected in the subclasses. However, it is kept for practical reasons. See 
    # https://github.com/python/mypy/issues/5876

    @abc.abstractmethod
    def nom_intercept(self, *args: Any, **kwargs: Any) -> torch.Tensor:
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
    def nom_slope(self, *args: Any, **kwargs: Any) -> torch.Tensor:
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

    def nom(self, pred: torch.Tensor, sens: Optional[torch.Tensor], *stat_args: Any, **stat_kwargs: Any
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
        intercept = self.nom_intercept(*stat_args, **stat_kwargs)
        slope = self.nom_slope(*stat_args, **stat_kwargs)
        return self.__compute_nom_or_denom(pred, sens, intercept, slope)

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
        return self.__compute_nom_or_denom(pred, sens, intercept, slope)

    @staticmethod
    def __compute_nom_or_denom(pred: torch.Tensor, sens: Optional[torch.Tensor], intercept: torch.Tensor,
                               slope: torch.Tensor) -> torch.Tensor:
        """
        Shorthand function to deduplicate the nom and denom functions, given the intercept and slope functions.

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
        nom = self.nom(pred, sens, *stat_args, **stat_kwargs)
        denom = self.denom(pred, sens, *stat_args, **stat_kwargs)
        return safe_div(nom, denom)

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
        `fix_value`. The linear expressions that make up the nominator and denominator are then combined into a single
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
        nom_intercept = self.nom_intercept(*stat_args, **stat_kwargs)
        denom_intercept = self.denom_intercept(*stat_args, **stat_kwargs)
        if isinstance(denom_intercept, float) or len(denom_intercept.shape) == 0:
            intercept = nom_intercept - denom_intercept * fix_value
        else:
            intercept = nom_intercept - torch.einsum('n...y,...y->n...y', denom_intercept, fix_value)
        intercept = torch.einsum('n...y,ns->n...s', intercept, sens).flatten(start_dim=1)

        nom_slope = self.nom_slope(*stat_args, **stat_kwargs)
        denom_slope = self.denom_slope(*stat_args, **stat_kwargs)
        if isinstance(denom_slope, float) or len(denom_slope.shape) == 0:
            slope = nom_slope - denom_slope * fix_value
        else:
            slope = nom_slope - torch.einsum('n...y,...y->...', denom_slope, fix_value)
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

    def nom_intercept(self) -> torch.Tensor:
        return 0.

    def nom_slope(self) -> torch.Tensor:
        return 1.

    def denom_intercept(self) -> torch.Tensor:
        return 1.

    def denom_slope(self) -> torch.Tensor:
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

    def nom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return 0.

    def nom_slope(self, label: torch.Tensor) -> torch.Tensor:
        return label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return label

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
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

    def nom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return 0.

    def nom_slope(self, label: torch.Tensor) -> torch.Tensor:
        return 1 - label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return 1 - label

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
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

    def nom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return 0.

    def nom_slope(self, label: torch.Tensor) -> torch.Tensor:
        return label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return 0.

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
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

    def nom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return label

    def nom_slope(self, label: torch.Tensor) -> torch.Tensor:
        return -label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return 1.

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
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

    def nom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return 1 - label

    def nom_slope(self, label: torch.Tensor) -> torch.Tensor:
        return 2 * label - 1

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return 1.

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        return 0.


class FalseNegativeFalsePositiveFraction(LinearFractionalStatistic):
    """
    FalseNegativeFalsePositiveFraction is a LinearFractionalStatistic that computes the ratio between false negatives
    and false positives.

    The statistic cannot be formulated as a single probability.

    The functions of its canonical form require that the tensor of target labels is provided with the same shape as the
    predictions.
    """

    def nom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        return label

    def nom_slope(self, label: torch.Tensor) -> torch.Tensor:
        return -label

    def denom_intercept(self, label: torch.Tensor) -> torch.Tensor:
        # Small value to avoid numerical issues in initial experiments. Now handled more gracefully.
        # return 1. / label.shape[0]
        return 0.

    def denom_slope(self, label: torch.Tensor) -> torch.Tensor:
        return 1 - label


class StackedLinearFractionalStatistic(LinearFractionalStatistic):
    """
    A vector-valued LinearFractionalStatistic that combines the outputs of K LinearFractionalStatistics into a single
    statistic with output (K, S) by stacking all outputs in the first dimension.
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

    def nom_intercept(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.__stack('nom_intercept', *args, **kwargs)

    def nom_slope(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.__stack('nom_slope', *args, **kwargs)

    def denom_intercept(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.__stack('denom_intercept', *args, **kwargs)

    def denom_slope(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.__stack('denom_slope', *args, **kwargs)

    @staticmethod
    def __ensure_shape(val: torch.Tensor) -> torch.Tensor:
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if len(val.shape) == 0:
            val = val.view(1, 1)
        return val

    def __stack(self, method: str, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.stack([
            self.__ensure_shape(getattr(stat, method)(*args, **kwargs))
            for stat in self.statistics
        ], dim=1)
