import torch

from .utils import safe_div


class Statistic(torch.nn.Module):
    def forward(self, feat, sens, label, pred):
        raise NotImplementedError


class LinearFractionalStatistic(Statistic):
    def nom_intercept(self, feat, label):
        raise NotImplementedError

    def nom_slope(self, feat, label):
        raise NotImplementedError

    def denom_intercept(self, feat, label):
        raise NotImplementedError

    def denom_slope(self, feat, label):
        raise NotImplementedError

    def nom(self, pred, feat, sens, label):
        nom = sens * (self.nom_intercept(feat, label) + self.nom_slope(feat, label) * pred)
        return nom.sum(dim=0)

    def denom(self, pred, feat, sens, label):
        denom = sens * (self.denom_intercept(feat, label) + self.denom_slope(feat, label) * pred)
        return denom.sum(dim=0)

    def forward(self, pred, feat, sens, label):
        nom = self.nom(pred, feat, sens, label)
        denom = self.denom(pred, feat, sens, label)
        return safe_div(nom, denom)

    def overall_statistic(self, pred, feat, label):
        return self.forward(pred, feat, 1., label)

    def linearize(self, feat, sens, label, c):
        intercept = self.nom_intercept(feat, label) - self.denom_intercept(feat, label) * c
        intercept = sens * intercept
        slope = self.nom_slope(feat, label) - self.denom_slope(feat, label) * c
        slope = sens * slope
        return intercept, slope


# class StackLinearFractionalStatistic(LinearFractionalStatistic):
#     def __init__(self, stats):
#         super().__init__()
#
#         if not isinstance(stats, (list, tuple)) or len(stats) == 0 or \
#                 not all(isinstance(stat, LinearFractionalStatistic) for stat in stats):
#             raise ValueError(f"Expected a non-empty list of statistics, got {stats}.")
#         self.stats = stats
#
#     def nom_intercept(self, feat, label):
#         return torch.stack([stat.nom_intercept(feat, label) for stat in self.stats], dim=-1)
#
#     def nom_slope(self, feat, label):
#         return torch.stack([stat.nom_slope(feat, label) for stat in self.stats], dim=-1)
#
#     def denom_intercept(self, feat, label):
#         return torch.stack([stat.denom_intercept(feat, label) for stat in self.stats], dim=-1)
#
#     def denom_slope(self, feat, label):
#         return torch.stack([stat.denom_slope(feat, label) for stat in self.stats], dim=-1)
#
#     def nom(self, pred, feat, sens, label):
#         return super().nom(pred.unsqueeze(1), feat, sens.unsqueeze(1), label)
#
#     def denom(self, pred, feat, sens, label):
#         return super().denom(pred.unsqueeze(1), feat, sens.unsqueeze(1), label)
#
#     def linearize(self, feat, sens, label, c):
#         intercept, slope = super().linearize(feat, sens.unsqueeze(1), label, c.squeeze())
#         return intercept.flatten(start_dim=1), slope.flatten(start_dim=1)


class PositiveRate(LinearFractionalStatistic):
    def nom_intercept(self, feat, label):
        return torch.zeros(label.shape)

    def nom_slope(self, feat, label):
        return torch.ones(label.shape)

    def denom_intercept(self, feat, label):
        return torch.ones(label.shape)

    def denom_slope(self, feat, label):
        return torch.zeros(label.shape)


class TruePositiveRate(LinearFractionalStatistic):
    def nom_intercept(self, feat, label):
        return torch.zeros(label.shape)

    def nom_slope(self, feat, label):
        return label

    def denom_intercept(self, feat, label):
        return label

    def denom_slope(self, feat, label):
        return torch.zeros(label.shape)


class FalsePositiveRate(LinearFractionalStatistic):
    def nom_intercept(self, feat, label):
        return torch.zeros(label.shape)

    def nom_slope(self, feat, label):
        return 1 - label

    def denom_intercept(self, feat, label):
        return 1 - label

    def denom_slope(self, feat, label):
        return torch.zeros(label.shape)


class PositivePredictiveValue(LinearFractionalStatistic):
    def nom_intercept(self, feat, label):
        return torch.zeros(label.shape)

    def nom_slope(self, feat, label):
        return label

    def denom_intercept(self, feat, label):
        return torch.zeros(label.shape)

    def denom_slope(self, feat, label):
        return torch.ones(label.shape)


class FalseOmissionRate(LinearFractionalStatistic):
    def nom_intercept(self, feat, label):
        return label

    def nom_slope(self, feat, label):
        return -label

    def denom_intercept(self, feat, label):
        return torch.ones(label.shape)

    def denom_slope(self, feat, label):
        return -torch.ones(label.shape)


class Accuracy(LinearFractionalStatistic):
    def nom_intercept(self, feat, label):
        return 1 - label

    def nom_slope(self, feat, label):
        return 2 * label - 1

    def denom_intercept(self, feat, label):
        return torch.ones(label.shape)

    def denom_slope(self, feat, label):
        return torch.zeros(label.shape)


class FalseNegativeFalsePositiveFraction(LinearFractionalStatistic):
    def nom_intercept(self, feat, label):
        return label

    def nom_slope(self, feat, label):
        return -label

    def denom_intercept(self, feat, label):
        # Small value to avoid numerical issues in initial experiments. Now handled more gracefully.
        # return torch.ones(feat.shape[0]) / feat.shape[0]
        return torch.zeros(label.shape)

    def denom_slope(self, feat, label):
        return 1 - label
