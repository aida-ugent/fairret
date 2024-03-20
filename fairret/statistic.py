import abc
from typing import Any, Tuple
import torch

from .utils import safe_div


class Statistic(abc.ABC, torch.nn.Module):
    """
    Abstract base class for a statistic.

    As a subclass of torch.nn.Module, it should implement the forward method with the
    'forward(self, pred, sens, *stat_args, **stat_kwargs)' signature.
    """

    @abc.abstractmethod
    def forward(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> torch.Tensor:
        """
        Compute the statistic for a batch of `N` samples for each sensitive feature.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, C)` with `C` the number of classes. For binary
                classification or regression, it can be :math:`C = 1`.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            stat_args: Any further arguments used to compute the statistic.
            stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: Shape :math:`(S)`.
        """
        raise NotImplementedError


class LinearFractionalStatistic(Statistic):
    """
    Absract base class for a linear-fractional Statistic. This is a Statistic that is computed as the ratio between two
    linear functions over the predictions.

    A LinearFractionalStatistic is constructed in a canonical form by defining the intercept and slope of the numerator
    and denominator linear functions, i.e. the functions nom_intercept, nom_slope, denom_intercept, and denom_slope.
    Each subclass must implement these functions (using any signature).

    The statistic is then computed as :math:`\\frac{nom\_intercept + nom\_slope * pred}{denom\_intercept + denom\_slope
    * pred}`.
    """

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

    def nom(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> torch.Tensor:
        """
        Intermediate function to compute the numerator of the linear-fractional statistic.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, C)` with `C` the number of classes. For binary
                classification or regression, it can be :math:`C = 1`.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            stat_args: Any further arguments used to compute the statistic.
            stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: Shape :math:`(S)`.
        """
        
        nom = self.nom_intercept(*stat_args, **stat_kwargs) + self.nom_slope(*stat_args, **stat_kwargs) * pred
        nom = sens * nom
        return nom.sum(dim=0)

    def denom(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> torch.Tensor:
        """
        Intermediate function to compute the denominator of the linear-fractional statistic.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, C)` with `C` the number of classes. For binary
                classification or regression, it can be :math:`C = 1`.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            stat_args: Any further arguments used to compute the statistic.
            stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: Shape :math:`(S)`.
        """

        denom = self.denom_intercept(*stat_args, **stat_kwargs) + self.denom_slope(*stat_args, **stat_kwargs) * pred
        denom = sens * denom
        return denom.sum(dim=0)

    def forward(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> torch.Tensor:
        nom = self.nom(pred, sens, *stat_args, **stat_kwargs)
        denom = self.denom(pred, sens, *stat_args, **stat_kwargs)
        return safe_div(nom, denom)

    def overall_statistic(self, pred: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> float:
        """
        Compute the overall statistic for the predictions. This is the value of the statistic when the sensitive feature
        is ignored, i.e. set to 1. The predictions are typically considered fair (with respect to the statistic) if and
        only if the statistic computed for every sensitive feature equals this value.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, C)` with `C` the number of classes. For binary
                classification or regression, it can be :math:`C = 1`.
            stat_args: Any further arguments used to compute the statistic.
            stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            float: The overall statistic.
        """
        return self.forward(pred, 1., *stat_args, **stat_kwargs)

    def fixed_constraint(self, fix_value: float, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reformulate the fairness definition for this LinearFractionalStatistic as a linear constraint.

        This is done by fixing the intended values of the statistic for every sensitive feature to a given value
        `fix_value`. The linear expressions that make up the nominator and denominator are then combined into a single
        linear constraint.

        Args:
            fix_value (float): The intended value of the statistic for every sensitive feature. Typically, this is the
            overall statistic.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: Any further arguments used to compute the statistic.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The intercept and slope of the linear constraint defined as
            :math:`intercept + slope * pred = 0`.
        """

        intercept = self.nom_intercept(*stat_args, **stat_kwargs) - self.denom_intercept(*stat_args, **stat_kwargs) * fix_value
        intercept = sens * intercept
        slope = self.nom_slope(*stat_args, **stat_kwargs) - self.denom_slope(*stat_args, **stat_kwargs) * fix_value
        slope = sens * slope
        return intercept, slope


class PositiveRate(LinearFractionalStatistic):
    """
    PositiveRate is a LinearFractionalStatistic that computes the average rate at which positive predictions are made in
    binary classification.

    Formulated as a probability, it computes :math:`P(\hat{Y} = 1 | S)` for categorical sensitive features :math:`S`,
    with the predicted label :math:`\hat{Y}` sampled according to the model's (probabilistic) prediction.

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

    Formulated as a probability, it computes :math:`P(\hat{Y} = 1 | Y = 1, S)` for categorical sensitive features
    :math:`S` and only for samples where the target label :math:`Y` is positive, with the predicted label
    :math:`\hat{Y}` sampled according to the model's (probabilistic) prediction.

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

    Formulated as a probability, it computes :math:`P(\hat{Y} = 1 | Y = 0, S)` for categorical sensitive features
    :math:`S` and only for samples where the target label :math:`Y` is negative, with the predicted label
    :math:`\hat{Y}` sampled according to the model's (probabilistic) prediction.

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

    Formulated as a probability, it computes :math:`P(Y = 1 | \hat{Y} = 1, S)` for categorical sensitive features
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

    Formulated as a probability, it computes :math:`P(Y = 1 | \hat{Y} = 0, S)` for categorical sensitive features
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

    Formulated as a probability, it computes :math:`P(\hat{Y} = Y | S)` for categorical sensitive features :math:`S`,
    with the predicted label :math:`\hat{Y}` sampled according to the model's (probabilistic) prediction and with
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
