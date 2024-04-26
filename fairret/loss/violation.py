import abc
from typing import Any, Optional
import torch

from .base import FairnessLoss
from ..statistic import Statistic, LinearFractionalStatistic


class ViolationLoss(FairnessLoss):
    """
    Abstract base class for fairness losses that penalize the violation vector of a fairness constraint. The violation
    vector is computed as the gap between the statistics per sensitive feature and a target statistic.

    Each subclass must implement the `penalize_violation` method.
    """

    def __init__(self, statistic: Statistic):
        """
        Args:
            statistic (Statistic): The statistic that should be used to calculate the violation vector. Preferably, a
                LinearFractionalStatistic is provided, as this allows for a straightforward calculation of the target
                statistic as the overall statistic.
        """

        super().__init__()
        self.statistic = statistic

    @abc.abstractmethod
    def penalize_violation(self, violation) -> torch.Tensor:
        """
        Penalize the fairness violation.

        Args:
            violation (torch.Tensor): The violation vector, i.e. the vector of gaps between the statistics per sensitive
            feature and the target statistic.

        Returns:
            torch.Tensor: A scalar tensor.
        """
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args, pred_as_logit=True,
                target_statistic: Optional[torch.Tensor] = None, **stat_kwargs: Any) -> torch.Tensor:
        """
        Calculate the violation vector in relation to the `target_statistic` and penalize this violation using the
        :py:meth:`~fairret.loss.violation.ViolationLoss.penalize_violation` method implemented by the subclass.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: All arguments used by the statistic that this loss minimizes.
            pred_as_logit (bool): Whether the `pred` tensor should be interpreted as logits. Though most losses are
                will simply take the sigmoid of `pred` if `pred_as_logit` is `True`, some losses benefit from improved
                numerical stability if they handle the conversion themselves.
            target_statistic (Optional[torch.Tensor]): The target statistic as a scalar tensor. If not provided for a
                LinearFractionalStatistic, the overall statistic will be used by default.
            **stat_kwargs: All keyword arguments used by the statistic that this loss computes.

        Returns:
            torch.Tensor: The calculated loss as a scalar tensor.
        """

        if pred_as_logit:
            pred = torch.sigmoid(pred)

        if target_statistic is None:
            if isinstance(self.statistic, LinearFractionalStatistic):
                if target_statistic is None:
                    # TODO check if we should detach.
                    target_statistic = self.statistic.overall_statistic(pred, *stat_args, **stat_kwargs)
            else:
                raise ValueError(f"Was initialized with a statistic of type {self.statistic.__class__}, but no 'c' was "
                                 f"given. Either set the statistic to a subclass of LinearFractionalStatistic or "
                                 f"provide a value for 'c'.")

        stats = self.statistic(pred, sens, *stat_args, **stat_kwargs)

        # TODO should be more formal about penalization when target statistics are zero.
        if any(target_statistic == 0.):
            raise NotImplementedError("Target statistic is zero. Penalization for this edge case is not implemented.")

        violation = torch.abs(stats / target_statistic - 1)

        loss = self.penalize_violation(violation)
        return loss


class NormLoss(ViolationLoss):
    """
    Fairness loss that penalizes the p-norm of the violation vector.
    """

    def __init__(self, statistic: Statistic, p=1):
        """
        Args:
            statistic (Statistic): The statistic that should be used to calculate the violation vector. Preferably, a
                LinearFractionalStatistic is provided, as this allows for a straightforward calculation of the target
                statistic as the overall statistic.
            p (int): The order of the norm. Default is 1.
        """

        super().__init__(statistic)
        self.p = p

    def penalize_violation(self, violation: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(violation, ord=self.p, dim=-1).sum()


class LSELoss(ViolationLoss):

    """
    Fairness loss that penalizes the log-sum-exp of the violation vector. The log-sum-exp is a smooth approximation of
    the maximum function, hence it approximates the maximum violation (or its :math:`\\Vert \\cdot \\Vert_\\infty` norm)
    """

    def penalize_violation(self, violation: torch.Tensor) -> torch.Tensor:
        loss = torch.logsumexp(violation, dim=-1) - torch.log(torch.tensor(violation.shape[-1]))
        return loss.sum()
