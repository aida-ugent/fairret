import torch


class FairnessLoss(torch.nn.Module):
    def forward(self, logit, feat, sens, label, **kwargs):
        raise NotImplementedError
