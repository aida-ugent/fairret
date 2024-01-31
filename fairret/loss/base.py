import torch


LOSS_BY_NAME = {}


class FairnessLoss(torch.nn.Module):
    def __init_subclass__(cls, name=None, **kwargs):
        # Register the subclasses such that we can find them by name.
        super().__init_subclass__(**kwargs)
        cls.name = name

        if name is None:
            # Registering the class is optional.
            return

        LOSS_BY_NAME[name] = cls

    def forward(self, logit, feat, sens, label, **kwargs):
        raise NotImplementedError
