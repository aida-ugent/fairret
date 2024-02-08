from .base import FairnessLoss
from .violation import (
    ViolationLoss,
    NormLoss,
    LSELoss
)
from .projection import (
    ProjectionLoss,
    KLProjectionLoss,
    JensenShannonProjectionLoss,
    TotalVariationProjectionLoss,
    SquaredEuclideanProjectionLoss
)
