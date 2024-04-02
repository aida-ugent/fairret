import abc
from typing import Any
import torch


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
            *stat_args: Any further arguments used to compute the statistic.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: Shape :math:`(S)`.
        """
        raise NotImplementedError
