import abc
from typing import Any
import torch


class Statistic(abc.ABC, torch.nn.Module):
    """
    Abstract base class for a statistic.

    As a subclass of torch.nn.Module, it should implement the :func:`~base.Statistic.forward` method with the
    :func:`~base.Statistic.forward(self, pred, sens, \*stat_args, \*\*stat_kwargs)` signature.
    """

    @abc.abstractmethod
    def forward(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any) -> torch.Tensor:
        """
        Compute the statistic for a batch of `N` samples for each sensitive feature.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: Any further arguments used to compute the statistic.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: Shape :math:`(S)`.
        """
        raise NotImplementedError

    def __call__(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args: Any, pred_as_logit=True, **stat_kwargs: Any
                ) -> torch.Tensor:
        """
        Override __call__ method from torch.nn.Module. If self.torch we use the torch.nn.Module.__call__() otherwise
        we only compute the forward pass (tensorflow case).

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 1)`, as we assume to be performing binary
                classification or regression.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: Any further arguments used to compute the statistic.
            (bool): Whether the `pred` tensor should be interpreted as logits. Though most losses are
                will simply take the sigmoid of `pred` if `pred_as_logit` is `True`, some losses benefit from improved
                numerical stability if they handle the conversion themselves.
            **stat_kwargs: Any keyword arguments used to compute the statistic.

        Returns:
            torch.Tensor: Shape :math:`(S)`.
        """
        if self.torch:
            return super().__call__(pred, sens, stat_args, *pred_as_logit, **stat_kwargs)
        return self.forward(pred, sens, *stat_args, **stat_kwargs)
