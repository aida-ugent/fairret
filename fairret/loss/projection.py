# The code block below uses PEP 563 to make cvxpy optional to load.
# The __future__ import will be deprecated, but the fix will only come with PEP 649 in the future.
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # The dependency is not optional when type checking.
    import cvxpy as cp
else:
    try:
        import cvxpy as cp
    except ImportError:
        cp = None

from typing import Any, Tuple
import abc
import torch

from .base import FairnessLoss
from ..statistic import LinearFractionalStatistic


class ProjectionLoss(FairnessLoss):
    """
    Abstract base class for fairness losses that penalize the statistical distance between a set of predictions and
    the fair projection of those predictions. The fair projection satisfies the linear fairness constraint
    corresponding to a LinearFractionalStatistic that is fixed to a target value (such as the overall statistic).

    The projections are computed using cvxpy. Hence, any subclass is expected to implement the statistical distance
    between distributions in both cvxpy and PyTorch by implementing the
    :py:meth:`~fairret.loss.projection.ProjectionLoss.cvxpy_distance` method and the
    :py:meth:`~fairret.loss.projection.ProjectionLoss.torch_distance` method respectively.

    Optionally, the :py:meth:`~fairret.loss.projection.ProjectionLoss.torch_distance_with_logits` method can be
    overwritten to provide a more numerically stable handling of predictions that are provided as logits. If left
    unimplemented, :py:meth:`~fairret.loss.projection.ProjectionLoss.torch_distance` will be called instead,
    after applying the sigmoid function to the predictions.

    Note:
        We use 'statistical distance' in a broad sense here, and do not require that the distance is a metric. See
        https://en.wikipedia.org/wiki/Statistical_distance for more information.
    """

    def __init__(self, statistic: LinearFractionalStatistic, force_proj_normalized=True, proj_eps=0.,
                 **solver_kwargs: Any):
        """
        Args:
            statistic (LinearFractionalStatistic): The LinearFractionalStatistic that defines the fairness constraint.
                The projection is computed through convex optimization, so the constraint should be linear. This is
                achieved by fixing equality in the LinearFractionalStatistic values to the overall statistic.
            force_proj_normalized (bool): Whether to force the projected distribution to be normalized. This might not
                be the case if the optimization does not converge to a solution that satisfies the normalization
                constraint. Hence, setting this to True will renormalize the projected distribution to sum to 1.
            proj_eps (float): Every probability value in the projected distribution is clamped to the interval
                [proj_eps, 1 - proj_eps]. Default is 0.
            The minimum value of every probability value in the projected distribution. Due to
                normalization, the maximum value in binary classification is then also 1 - proj_eps. Setting this to
                a small, non-negative value helps prevent numerical instability if the optimization is not done to
                convergence.
            solver_kwargs: Any keyword arguments to be passed to the cvxpy solver. The default configuration is::

                {
                    'solver': 'SCS',
                    'warm_start': True,
                    'max_iters': 10,
                    'ignore_dpp': True
                }
                
        """
        if cp is None:
            raise ImportError("The cvxpy package is required for ProjectionLosses. Please install it using "
                              "`pip install cvxpy`.")

        super().__init__()
        self.statistic = statistic
        self.force_proj_normalized = force_proj_normalized
        self.proj_eps = proj_eps
        self.solver_kwargs = {
            'solver': 'SCS',
            'warm_start': True,
            'max_iters': 10,
            'ignore_dpp': True
        }
        if solver_kwargs is not None:
            if not isinstance(solver_kwargs, dict):
                raise ValueError("solver_cfg must be a dict.")
            self.solver_kwargs |= solver_kwargs

        self._proj_cvxpy = None
        self._pred_cvxpy = None
        self._intercept = None
        self._slope = None
        self._problem = None

    @abc.abstractmethod
    def cvxpy_distance(self, pred: cp.Parameter, proj: cp.Variable) -> cp.Expression:
        """
        Compute the statistical distance between `pred` and `proj` in cvxpy. Used for the convex optimization problem.

        Args:
            pred (cp.Parameter): The predicted distribution in shape (N,2). As we assume binary classification, the
                first column is the probability of the negative class and the second column is the probability of the
                positive class.
            proj (cp.Variable): The projected distribution in shape (N,2). As we assume binary classification, the
                first column is the probability of the negative class and the second column is the probability of the
                positive class.

        Returns:
            cp.Expression: The statistical distance in shape (1,).
        """

        raise NotImplementedError

    @abc.abstractmethod
    def torch_distance(self, pred: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute the statistical distance between `pred` and `proj` in PyTorch. Used for computing the gradient of the
        distance between the predictions and the projection (with respect to the predictions).

        Args:
            pred (torch.Tensor): The predicted distribution in shape (N,1). As we assume binary classification, this is
                the probability of the positive class.
            proj (torch.Tensor): The projected distribution in shape (N,2). As we assume binary classification, the
                first column is the probability of the negative class and the second column is the probability of the
                positive class.

        Returns:
            torch.Tensor: The statistical distance as a scalar tensor.
        """

        raise NotImplementedError

    def torch_distance_with_logits(self, pred, proj):
        """
        A more numerically stable alternative method to
        :py:meth:`~fairret.loss.projection.ProjectionLoss.torch_distance`, where `pred` is assumed to be logits.

        Args:
            pred (torch.Tensor): The predicted distribution as logits, in shape (N,1). As we assume binary
                classification, this is the logit of the probability of the positive class.
            proj (torch.Tensor): The projected distribution in shape (N,2) as probabilities. As we assume binary
                classification, the first column is the probability of the negative class and the second column is the
                probability of the positive class.

        Returns:
            torch.Tensor: The statistical distance as a scalar tensor.
        """

        return self.torch_distance(torch.sigmoid(pred), proj)

    def forward(self, pred: torch.Tensor, sens: torch.Tensor, *stat_args, pred_as_logit=True, **stat_kwargs: Any
                ) -> torch.Tensor:
        """
        Calculate the fairness loss by projecting the predictions onto the fair set and computing the statistical
        distance between the predictions and the projection.

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

        if pred_as_logit:
            prob_pred = torch.sigmoid(pred)
        else:
            prob_pred = pred

        proj = self._fit_cvxpy(prob_pred, sens, *stat_args, **stat_kwargs)
        if self.force_proj_normalized:
            proj /= proj.mean(dim=-1, keepdim=True)
        proj = proj.clamp(min=self.proj_eps, max=1 - self.proj_eps)

        if pred_as_logit:
            try:
                return self.torch_distance_with_logits(pred, proj)
            except NotImplementedError:
                pred = torch.sigmoid(pred)
        return self.torch_distance(pred, proj)

    def _init_cvxpy(self, batch_size: int, constraint_dim: int
                    ) -> Tuple[Tuple[cp.Parameter, cp.Variable, cp.Parameter, cp.Parameter], cp.Problem]:
        """
        Initialize the cvxpy problem for the convex optimization.

        Args:
            batch_size (int): The batch size of the predictions.
            constraint_dim (int): The dimension of the linear fairness constraint, typically the number of sensitive
                features.

        Returns:
            Tuple[cp.Parameter, cp.Variable, cp.Parameter, cp.Parameter]: A tuple of the parameters and variables for
                the cvxpy problem, in the order (pred, proj, intercept, slope).
            cp.Problem: The cvxpy problem to be solved.
        """

        pred = cp.Parameter((batch_size, 2), nonneg=True)
        proj = cp.Variable((batch_size, 2), nonneg=True)
        intercept = cp.Parameter(constraint_dim)
        slope = cp.Parameter((batch_size, constraint_dim))

        objective = cp.Minimize(self.cvxpy_distance(pred, proj))
        constraints = [
            intercept + proj[:, 1] @ slope == 0.,
            cp.sum(proj, axis=1) == 1.
        ]
        problem = cp.Problem(objective, constraints)

        # if not problem.is_dcp(dpp=True):
        #     raise ValueError("The problem is not DPP.")
        return (pred, proj, intercept, slope), problem

    @property
    def _batch_size(self) -> int:
        """
        Returns:
            int: The batch size of the first batch that was used to initialize the cvxpy problem.
        """

        return self._pred_cvxpy.shape[0]

    def _fit_cvxpy(self, pred, sens, *stat_args, **stat_kwargs) -> torch.Tensor:
        """
        Iterate over the cvxpy problem to find the projection of the predictions onto the fair set.

        Args:
            pred (torch.Tensor): Predictions of shape :math:`(N, 2)`. The predicted distribution in shape (N,2). As we
                assume binary classification, the first column is the probability of the negative class and the second
                column is the probability of the positive class.
            sens (torch.Tensor): Sensitive features of shape :math:`(N, S)` with `S` the number of sensitive features.
            *stat_args: All arguments used by the statistic that this loss minimizes.
            **stat_kwargs: All keyword arguments used by the statistic that this loss computes.

        Returns:
            torch.Tensor: The projection of the predictions onto the fair set in the shape (N, 2).
        """

        pred = pred.detach()

        # Constrain the fair set to those where the statistics match the target_statistic.
        # Here, we choose target_statistic as the overall statistic of the predictions.
        target_statistic = self.statistic.overall_statistic(pred, *stat_args, **stat_kwargs)

        # Based on the target_statistic and the feature-specific statistic, we precompute the intercept and slope of the
        # linear fairness constraint.
        intercept, slope = self.statistic.fixed_constraint(target_statistic, sens, *stat_args, **stat_kwargs)
        intercept = intercept.sum(dim=0)

        # If this is the first time the loss is called, initialize the convex optimization problem.
        if self._problem is None:
            params, problem = self._init_cvxpy(*slope.shape)
            self._pred_cvxpy, self._proj_cvxpy, self._intercept, self._slope = params
            self._problem = problem

        # If the received batch is smaller than we expected (e.g. because it is the last batch in an epoch), then pad
        # the batch with zeros to match the batch size of the first batch
        batch_size = pred.shape[0]
        if batch_size < self._batch_size:
            gap = self._batch_size - batch_size
            pred = torch.cat([pred, torch.zeros(gap, 1)])
            slope = torch.cat([slope, torch.zeros((gap, slope.shape[1]))])
        elif batch_size > self._batch_size:
            raise ValueError(f"For {self.__class__.__name__} with reuse_definition=True, the batch size must never "
                             f"exceed the batch size of the first batch. Got {batch_size}, yet we initialized with "
                             f"{self._batch_size}!")

        # Expand the prediction to its Bernoulli distribution and fill in the parameters of the cvxpy problem.
        pred = torch.cat([1 - pred, pred], dim=-1)
        self._pred_cvxpy.value = pred.numpy()
        self._intercept.value = intercept.numpy()
        self._slope.value = slope.numpy()

        # Solve the convex optimization problem.
        self._problem.solve(**self.solver_kwargs)

        # Extract the solution from the cvxpy problem and do some checks.
        proj = self._proj_cvxpy.value
        if proj is None:
            neg_intercepts = intercept < 0
            if neg_intercepts.any() and (slope[:, neg_intercepts] <= 0).all():
                raise RuntimeError("The slope and intercept of a linear constraint are both negative. This means a "
                                   "valid score function (with only nonnegative scores) does not exist!")
            # intercept, slope = self.stat.linearize(feat, sens, label, c)
            # raise RuntimeError(f"The convex optimization problem did not converge. "
            #                    f"It has status {self._problem.solution.status}")
            proj = pred.detach().numpy()
            print("The convex optimization problem did not converge! Continuing...")

        proj = torch.from_numpy(proj)
        if batch_size < self._batch_size:
            proj = proj[:batch_size]
        return proj


class KLProjectionLoss(ProjectionLoss):
    """
    Fairness loss that penalizes the Kullback-Leibler divergence between the predicted distribution and the fair
    projection of the predictions.
    """

    def cvxpy_distance(self, pred, proj):
        return cp.sum(cp.kl_div(proj, pred)) / proj.shape[0]

    def torch_distance(self, pred, proj):
        pred = torch.cat([1 - pred, pred], dim=-1)
        log_pred = torch.log(pred)
        dist = torch.nn.functional.kl_div(log_pred, proj, reduction='batchmean')
        return dist

    def torch_distance_with_logits(self, pred, proj):
        # Use log-sigmoid operation for numerical stability
        log_sigmoid_fn = torch.nn.LogSigmoid()
        # Log-probabilities of the Bernoulli distribution. We use that 1 - sigmoid(x) = sigmoid(-x)
        log_pred = torch.cat([log_sigmoid_fn(-pred), log_sigmoid_fn(pred)], dim=-1)

        # Compute the KL divergence between the prediction distribution and the closest fair distribution
        # Note that PyTorch reverses the typical KL divergence ordering!
        dist = torch.nn.functional.kl_div(log_pred, proj, reduction='batchmean')
        return dist


class JensenShannonProjectionLoss(ProjectionLoss):
    """
    Fairness loss that penalizes the Jensen-Shannon divergence between the predicted distribution and the fair
    projection of the predictions.
    """

    def cvxpy_distance(self, pred, proj):
        avg = (proj + pred) / 2
        return cp.sum(cp.kl_div(proj, avg)) + cp.sum(cp.kl_div(pred, avg)) / (2 * proj.shape[0])

    def torch_distance(self, pred, proj):
        pred = torch.cat([1 - pred, pred], dim=-1)
        avg = (pred + proj) / 2
        log_avg = torch.log(avg)

        dist = (torch.nn.functional.kl_div(log_avg, pred, reduction='batchmean') +
                torch.nn.functional.kl_div(log_avg, proj, reduction='batchmean')) / 2
        return dist


class TotalVariationProjectionLoss(ProjectionLoss):
    """
    Fairness loss that penalizes the Total Variation Distance between the predicted distribution and the fair
    projection of the predictions.
    """

    def cvxpy_distance(self, pred, proj):
        return 1 / 2 * cp.sum(cp.abs(proj - pred)) / proj.shape[0]

    def torch_distance(self, pred, proj):
        pred = torch.cat([1 - pred, pred], dim=-1)
        dist = torch.sum(torch.abs(pred - proj), dim=-1).mean()
        return dist


class ChiSquaredProjectionLoss(ProjectionLoss):
    """
    Fairness loss that penalizes the Chi-Squared Distance between the predicted distribution and the fair projection of
    the predictions.
    """

    def cvxpy_distance(self, pred, proj):
        return cp.sum(cp.power(proj, 2) / pred - 1) / proj.shape[0]

    def torch_distance(self, pred, proj):
        pred = torch.cat([1 - pred, pred], dim=-1)
        dist = torch.sum(proj ** 2 / pred, dim=-1).mean()
        return dist


class SquaredEuclideanProjectionLoss(ProjectionLoss):
    """
    Fairness loss that penalizes the Squared Euclidean Distance between the predicted distribution and the fair
    projection of the predictions.
    """

    def cvxpy_distance(self, pred, proj):
        return cp.sum((proj - pred) ** 2) / proj.shape[0]

    def torch_distance(self, pred, proj):
        proj = proj[:, 1]
        dist = ((pred - proj) ** 2).mean()
        return dist
