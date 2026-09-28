"""Experimental Torch-native sparse GP building blocks.

Importing this optional backend does not load FLARE's native extension. The
stateful GP model and ASE calculator are available from ``flare.tensor.otf``.
"""

from .b2 import B2
from .kernels import normalized_dot_product
from .lambda_cache import (
    GroupedLambdaCache,
    assemble_cached_grouped_lambda_observation_covariance,
    build_grouped_lambda_cache,
    cached_grouped_lambda_observation_covariance,
)
from .linalg import SparsePosterior, fit_sparse_gp
from .observations import (
    CachedQ,
    ObservationLayout,
    ase_stress_to_native,
    assemble_cached_q_observation_covariance,
    assemble_observation_covariance,
    build_q_cache,
    cached_q_observation_covariance,
    inducing_observation_covariance,
    native_stress_to_ase,
)
from .prediction import MeanPrediction, predict_mean_efs
from .structures import StructureBatch
from .uncertainty import predict_variance_efs, prior_observation_variance

__all__ = [
    "B2", "StructureBatch", "ObservationLayout", "CachedQ", "GroupedLambdaCache",
    "SparsePosterior",
    "MeanPrediction", "normalized_dot_product", "inducing_observation_covariance",
    "assemble_observation_covariance", "build_q_cache",
    "cached_q_observation_covariance", "assemble_cached_q_observation_covariance",
    "build_grouped_lambda_cache", "cached_grouped_lambda_observation_covariance",
    "assemble_cached_grouped_lambda_observation_covariance",
    "fit_sparse_gp", "predict_mean_efs", "predict_variance_efs",
    "prior_observation_variance",
    "native_stress_to_ase", "ase_stress_to_native",
]
