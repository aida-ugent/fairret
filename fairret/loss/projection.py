import torch
import cvxpy as cp

from .base import FairnessLoss
from ..statistic import LinearFractionalStatistic


class ProjectionLoss(FairnessLoss):
    def __init__(self, stat: LinearFractionalStatistic,
                 solver_cfg=None,
                 ensure_nonneg=True):
        super().__init__()
        self.stat = stat
        self.solver_cfg = {
            'solver': 'SCS',
            'warm_start': True,
            'max_iters': 10,
            'ignore_dpp': True
        }
        if solver_cfg is not None:
            if not isinstance(solver_cfg, dict):
                raise ValueError("solver_cfg must be a dict.")
            self.solver_cfg |= solver_cfg
        self.ensure_nonneg = ensure_nonneg

        self._f = None
        self._h = None
        self._intercept = None
        self._slope = None
        self._problem = None

    @staticmethod
    def cvxpy_objective(f, h):
        raise NotImplementedError

    def forward(self, pred, feat, sens, label, **kwargs):
        raise NotImplementedError

    def _init_cvxpy(self, batch_size, constraint_dim):
        f = cp.Variable((batch_size, 2), nonneg=True)
        h = cp.Parameter((batch_size, 2), nonneg=True)
        intercept = cp.Parameter(constraint_dim)
        slope = cp.Parameter((batch_size, constraint_dim))

        objective = cp.Minimize(self.cvxpy_objective(f, h))
        constraints = [
            intercept + f[:, 1] @ slope == 0.,
            cp.sum(f, axis=1) == 1.
        ]
        problem = cp.Problem(objective, constraints)

        # if not problem.is_dcp(dpp=True):
        #     raise ValueError("The problem is not DPP.")
        return (f, h, intercept, slope), problem

    @property
    def _batch_size(self):
        return self._h.shape[0]  # h parameter in cvxpy Problem definition

    def _fit_cvxpy(self, pred, feat, sens, label):
        pred = pred.detach()

        # Constrain the fair set to those where the statistics match c.
        # Here, we choose c as the overall statistic of the predictions.
        c = self.stat.overall_statistic(pred, feat, label)

        # Based on c and the specific statistic, we precompute the intercept and slope of the linear fairness constraint
        intercept, slope = self.stat.linearize(feat, sens, label, c)
        intercept = intercept.sum(dim=0)

        # If this is the first time the loss is called, initialize the convex optimization problem
        if self._problem is None:
            params, problem = self._init_cvxpy(*slope.shape)
            self._f, self._h, self._intercept, self._slope = params
            self._problem = problem

        # If the received batch is smaller than we expected (e.g. because it is the last batch in an epoch), then pad
        # the batch with zeros to match the batch size of the first batch
        batch_size = pred.shape[0]
        if batch_size < self._batch_size:
            gap = self._batch_size - batch_size
            pred = torch.cat([pred, torch.zeros(gap)])
            slope = torch.cat([slope, torch.zeros((gap, slope.shape[1]))])
        elif batch_size > self._batch_size:
            raise ValueError(f"For {self.__class__.__name__} with reuse_definition=True, the batch size must never "
                             f"exceed the batch size of the first batch. Got {batch_size}, yet we initialized with "
                             f"{self._batch_size}!")

        # Expand the prediction to its Bernoulli distribution and solve the convex optimization problem
        pred = torch.cat([1 - pred, pred], dim=-1)

        self._h.value = pred.detach().numpy()
        self._intercept.value = intercept.detach().numpy()
        self._slope.value = slope.detach().numpy()

        self._problem.solve(**self.solver_cfg)

        projection = self._f.value
        if projection is None:
            neg_intercepts = intercept < 0
            if neg_intercepts.any() and (slope[:, neg_intercepts] <= 0).all():
                raise RuntimeError("The slope and intercept of a linear constraint are both negative. This means a "
                                   "valid score function (with only nonnegative scores) does not exist!")
            # intercept, slope = self.stat.linearize(feat, sens, label, c)
            # raise RuntimeError(f"The convex optimization problem did not converge. "
            #                    f"It has status {self._problem.solution.status}")
            projection = pred.detach().numpy()
            print("The convex optimization problem did not converge! Continuing...")

        projection = torch.from_numpy(projection)
        if batch_size < self._batch_size:
            projection = projection[:batch_size]

        if self.ensure_nonneg:
            # The solution should be a Bernoulli distribution, but the convex optimization might not yield one due to
            # numerical issues. We thus ensure that the solution sums to one (at the cost of accuracy).
            projection[:, 0] = 1 - projection[:, 1]
        return projection


class KLProjectionLoss(ProjectionLoss):
    def __init__(self,
                 stat,
                 # The KLD is not defined for distributions with probability exactly 0 or 1.
                 # We thus clamp the solution to be in [eps, 1 - eps].
                 eps=1e-8,
                 **kwargs):
        super().__init__(stat, **kwargs)
        self.eps = eps

    @staticmethod
    def cvxpy_objective(f, h):
        return cp.sum(cp.kl_div(f, h)) / f.shape[0]

    def forward(self, logit, feat, sens, label, as_logit=True, **kwargs):
        if as_logit:
            pred = torch.sigmoid(logit)
        else:
            pred = logit
            logit = torch.logit(pred)

        solution = self._fit_cvxpy(pred, feat, sens, label)
        solution = solution.clamp(min=self.eps, max=1-self.eps)

        # Use log-sigmoid operation for numerical stability
        log_sigmoid_fn = torch.nn.LogSigmoid()
        # Log-probabilities of the Bernoulli distribution. We use that 1 - sigmoid(x) = sigmoid(-x)
        log_pred = torch.cat([log_sigmoid_fn(-logit), log_sigmoid_fn(logit)], dim=-1)

        # Compute the KL divergence between the prediction distribution and the closest fair distribution
        # Note that PyTorch reverses the typical KL divergence ordering!
        dist = torch.nn.functional.kl_div(log_pred, solution, reduction='batchmean')
        return dist


class JensenShannonProjectionLoss(ProjectionLoss):
    def __init__(self,
                 stat,
                 # The KLD is not defined for distributions with probability exactly 0 or 1.
                 # We thus clamp the solution to be in [eps, 1 - eps].
                 eps=1e-8,
                 **kwargs):
        super().__init__(stat, **kwargs)
        self.eps = eps

    @staticmethod
    def cvxpy_objective(f, h):
        avg = (f + h) / 2
        return cp.sum(cp.kl_div(f, avg)) + cp.sum(cp.kl_div(h, avg)) / (2 * f.shape[0])

    def forward(self, logit, feat, sens, label, as_logit=True, **kwargs):
        if as_logit:
            pred = torch.sigmoid(logit)
        else:
            pred = logit

        solution = self._fit_cvxpy(pred, feat, sens, label)
        solution = solution.clamp(min=self.eps, max=1-self.eps)

        pred = torch.cat([1 - pred, pred], dim=-1)
        avg = (pred + solution) / 2
        log_avg = torch.log(avg)

        dist = (torch.nn.functional.kl_div(log_avg, pred, reduction='batchmean') +
                torch.nn.functional.kl_div(log_avg, solution, reduction='batchmean')) / 2
        return dist


class TotalVariationProjectionLoss(ProjectionLoss):
    @staticmethod
    def cvxpy_objective(f, h):
        return 1 / 2 * cp.sum(cp.abs(f - h)) / f.shape[0]

    def forward(self, logit, feat, sens, label, as_logit=True, **kwargs):
        if as_logit:
            pred = torch.sigmoid(logit)
        else:
            pred = logit

        solution = self._fit_cvxpy(pred, feat, sens, label)
        pred = torch.cat([1 - pred, pred], dim=-1)

        dist = torch.sum(torch.abs(pred - solution), dim=-1).mean()
        return dist


class ChiSquaredProjectionLoss(ProjectionLoss):
    @staticmethod
    def cvxpy_objective(f, h):
        return cp.sum(cp.power(f, 2) / h - 1) / f.shape[0]

    def forward(self, logit, feat, sens, label, as_logit=True, **kwargs):
        if as_logit:
            pred = torch.sigmoid(logit)
        else:
            pred = logit

        solution = self._fit_cvxpy(pred, feat, sens, label)
        pred = torch.cat([1 - pred, pred], dim=-1)

        dist = torch.sum(solution ** 2 / pred, dim=-1).mean()
        return dist


class SquaredEuclideanProjectionLoss(ProjectionLoss):
    @staticmethod
    def cvxpy_objective(f, h):
        return cp.sum((f - h) ** 2) / f.shape[0]

    def forward(self, logit, feat, sens, label, as_logit=True, **kwargs):
        if as_logit:
            pred = torch.sigmoid(logit)
        else:
            pred = logit

        solution = self._fit_cvxpy(pred, feat, sens, label)
        solution = solution[:, 1]

        dist = ((pred - solution) ** 2).mean()
        return dist
