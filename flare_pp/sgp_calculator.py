import numpy as np
from _C_flare import SparseGP_DTC
from scipy.optimize import minimize


def compute_negative_likelihood(hyperparameters, sparse_gp):
    """Compute the negative log likelihood and gradient with respect to the
    hyperparameters."""

    assert(len(hyperparameters) == len(sparse_gp.hyperparameters))

    negative_likelihood = \
        -sparse_gp.compute_likelihood_gradient(hyperparameters)
    negative_likelihood_gradient = -sparse_gp.likelihood_gradient

    return negative_likelihood, negative_likelihood_gradient


def optimize_hyperparameters(sparse_gp, display_results=True,
                             gradient_tolerance=1e-4, max_iterations=10):
    """Optimize the hyperparameters of a sparse GP model."""

    # Optimize the hyperparameters with BFGS.
    initial_guess = sparse_gp.hyperparameters
    arguments = (sparse_gp)

    optimization_result = \
        minimize(compute_negative_likelihood, initial_guess, arguments,
                 method='BFGS', jac=True,
                 options={'disp': display_results,
                          'gtol': gradient_tolerance,
                          'maxiter': max_iterations})

    # Reset the hyperparameters.
    sparse_gp.set_hyperparameters(optimization_result.x)
    sparse_gp.log_marginal_likelihood = -optimization_result.fun
    sparse_gp.likelihood_gradient = -optimization_result.jac

    return optimization_result


if __name__ == '__main__':
    pass
