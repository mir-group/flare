"""Tests of many body basis functions."""
import numpy as np
from flare import basis, struc, env
from copy import deepcopy


def test_behler_radial():
    """Test Behler radial derivativer."""

    n_total = np.random.randint(2, 20)
    n = np.random.randint(0, high=n_total)
    sigma = np.random.uniform(0.1, 10)
    r_cut = 5
    r_ij = np.random.uniform(0, r_cut)
    delt = 1e-8

    basis_val, basis_derv = basis.behler_radial(r_ij, n, n_total, sigma, r_cut)
    basis_delt, _ = basis.behler_radial(r_ij + delt, n, n_total, sigma, r_cut)
    derv_delt = (basis_delt - basis_val) / delt

    assert(np.isclose(basis_derv, derv_delt))


def test_legendre():
    """Draw a random integer and do a finite difference test of the \
corresponding Legendre polynomial derivative."""

    n = np.random.randint(0, high=11)
    x = 2 * np.random.uniform() - 1
    delta = 1e-8

    val, derv = basis.legendre(n, x)
    val_delt, _ = basis.legendre(n, x + delta)
    derv_delt = (val_delt - val) / delta

    assert(np.isclose(derv, derv_delt))


def test_cos_grad():
    """Test that gradient of cos(theta_ijk) is correctly computed."""

    cell = np.eye(3) * 100
    species = np.array([1, 1, 1])
    positions = np.random.rand(3, 3)
    delt = 1e-8
    cutoffs = np.array([10, 10])
    structure = struc.Structure(cell, species, positions)
    test_env = env.AtomicEnvironment(structure, 0, cutoffs,
                                     compute_angles=True)

    # perturb central atom
    pos_delt_1 = deepcopy(positions)
    coord_1 = np.random.randint(0, 3)
    pos_delt_1[0, coord_1] += delt
    structure_1 = struc.Structure(cell, species, pos_delt_1)
    test_env_1 = env.AtomicEnvironment(structure_1, 0, cutoffs,
                                       compute_angles=True)

    cos_theta = test_env.cos_thetas[0, 1]
    cos_theta_1 = test_env_1.cos_thetas[0, 1]
    bond_vec_j = test_env.bond_array_2[0]
    bond_vec_k = test_env.bond_array_2[1]

    cos_delt_1 = (cos_theta_1 - cos_theta) / delt
    cos_grad_val = basis.cos_grad(cos_theta, bond_vec_j, bond_vec_k)
    assert(np.isclose(cos_delt_1, cos_grad_val[0, coord_1]))

    # perturb environment atom
    pos_delt_2 = deepcopy(positions)
    pert_atom = np.random.randint(1, 3)
    coord_2 = np.random.randint(0, 3)
    pos_delt_2[pert_atom, coord_2] += delt
    structure_2 = struc.Structure(cell, species, pos_delt_2)
    test_env_2 = env.AtomicEnvironment(structure_2, 0, cutoffs,
                                       compute_angles=True)

    cos_theta_2 = test_env_2.cos_thetas[0, 1]
    cos_delt_2 = (cos_theta_2 - cos_theta) / delt

    assert(np.isclose(cos_delt_2, cos_grad_val[pert_atom, coord_2]))
