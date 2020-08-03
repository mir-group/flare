import numpy as np
from flare_pp.sparse_gp import SparseGP
from build._C_flare import DotProductKernel, B2_Calculator, SparseGP_DTC
from flare import struc


# Make random structure.
n_atoms = 4
cell = np.eye(3)
positions = np.random.rand(n_atoms, 3)
species = [0, 1, 0, 1]
test_structure = struc.Structure(cell, species, positions)

# Test update db
custom_range = [1, 3]
energy = np.random.rand()
forces = np.random.rand(n_atoms, 3)
stress = np.random.rand(6)

# Create sparse GP model.
sigma = 1.0
power = 2
kernel = DotProductKernel(1.0, power, 0)
cutoff_function = "quadratic"
cutoff = 1.0
many_body_cutoffs = [cutoff]
radial_basis = "chebyshev"
radial_hyps = [0., cutoff]
cutoff_hyps = []
settings = [2, 4, 3]
calc = B2_Calculator(radial_basis, cutoff_function, radial_hyps, cutoff_hyps,
                     settings, 0)
sigma_e = 1.0
sigma_f = 1.0
sigma_s = 1.0

sgp_cpp = SparseGP_DTC([kernel], sigma_e, sigma_f, sigma_s)
sgp_py = SparseGP([kernel], [calc], cutoff, many_body_cutoffs, sigma_e,
                  sigma_f, sigma_s)


def test_update_db():
    """Check that the covariance matrices have the correct size after the
    sparse GP is updated."""

    sgp_py.update_db(test_structure, forces, custom_range, energy, stress)

    assert(sgp_py.sparse_gp.Kuu.shape[0] == len(custom_range))
    assert(sgp_py.sparse_gp.Kuf_struc.shape[1] == 1 + n_atoms * 3 + 6)


def test_train():
    """Check that the hyperparameters and likelihood are updated when the
    train method is called."""

    hyps_init = tuple(sgp_py.hyps)
    sgp_py.train(max_iterations=20)
    hyps_post = tuple(sgp_py.hyps)

    assert(hyps_init != hyps_post)
    assert(sgp_py.likelihood != 0.)
