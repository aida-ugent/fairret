import abc
from typing import Any
import torch


class FairnessLoss(abc.ABC, torch.nn.Module):
    """
    Abstract base class for fairness losses, also referred to as fairrets.
    """

    @abc.abstractmethod
    def forward(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, pred_as_logit=True, **stat_kwargs: Any
                ) -> torch.Tensor:
        """
        Abstract method that should be implemented by subclasses to calculate the loss.
        
        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: All arguments used by the statistic that this loss minimizes.
            pred_as_logit (bool): Whether the `pred` tensor should be interpreted as logits. Though most losses are
                will simply take the sigmoid of `pred` if `pred_as_logit` is `True`, some losses benefit from improved
                numerical stability if they handle the conversion themselves.
            **stat_kwargs: All keyword arguments used by the statistic that this loss computes.

        Returns:
            torch.Tensor: The calculated loss as a scalar tensor.
        """
        raise NotImplementedError
