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
            pred (torch.Tensor): Predictions of shape :math:`(N, C)` with `C` the number of classes. For binary
                classification or regression, it can be :math:`C = 1`.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: Any further arguments used to compute the statistic.
            pred_as_logit (bool): Whether the `pred` tensor should be interpreted as logits. Though most losses are
                will simply take the sigmoid of `pred` if `pred_as_logit` is `True`, some losses benefit from improved
                numerical stability if they handle the conversion themselves.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: The calculated loss as a scalar tensor.
        """
        raise NotImplementedError
