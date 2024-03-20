from typing import Any
import torch


class FairnessLoss(torch.nn.Module):
    def forward(self, logit: torch.Tensor, sens: torch.Tensor, *stat_args: Any, **stat_kwargs: Any):
        raise NotImplementedError
