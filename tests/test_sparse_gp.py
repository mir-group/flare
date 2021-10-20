import pytest
import numpy as np
import os
from copy import deepcopy
import json

from flare_pp._C_flare import SparseGP, NormalizedDotProduct, B2, Structure
from flare_pp.sparse_gp import SGP_Wrapper
from flare_pp.sparse_gp_calculator import SGP_Calculator

from flare.ase.calculator import FLARE_Calculator
from flare.ase.atoms import FLARE_Atoms

from ase.calculators.lj import LennardJones
from ase.build import bulk


def get_random_structure():
    cell = np.eye(3)


@pytest.fixture(scope="module")
def dot_product_kernel():
    sigma = 2.0
    power = 2

    return NormalizedDotProduct(sigma, power)


@pytest.fixture(scope="module")
def cutoff():
    return 5.0


@pytest.fixture(scope="module")
def lj_calc():
    return LennardJones()


@pytest.fixture(scope="module")
def training_structure():
    # Returns a randomly jittered diamond structure.
    a = 3.52678
    supercell = bulk("C", "diamond", a=a, cubic=True)
    supercell.positions += (2 * np.random.rand(8, 3) - 1) * 0.1
    flare_atoms = FLARE_Atoms.from_ase_atoms(supercell)

    return flare_atoms


@pytest.fixture(scope="module")
def b2_calculator(cutoff):
    cutoff_function = "quadratic"
    radial_basis = "chebyshev"
    radial_hyps = [0.0, cutoff]
    cutoff_hyps = []
    descriptor_settings = [1, 8, 3]
    b2_calc = B2(radial_basis, cutoff_function, radial_hyps, cutoff_hyps,
                 descriptor_settings)

    return b2_calc


@pytest.fixture(scope="module")
def bare_sgp_wrapper(dot_product_kernel, b2_calculator, cutoff, lj_calc):
    sigma_e = 0.001
    sigma_f = 0.05
    sigma_s = 0.006
    species_map = {6: 0}
    single_atom_energies = {0: -5}
    variance_type = "local"
    max_iterations = 20
    opt_method = "L-BFGS-B"
    bounds = [(None, None), (sigma_e, None), (None, None), (None, None)]

    sgp_model = SGP_Wrapper(
        [dot_product_kernel],
        [b2_calculator],
        cutoff,
        sigma_e,
        sigma_f,
        sigma_s,
        species_map,
        single_atom_energies=single_atom_energies,
        variance_type=variance_type,
        opt_method=opt_method,
        bounds=bounds,
        max_iterations=max_iterations
    )

    return sgp_model


@pytest.fixture(scope="module")
def sgp_wrapper(bare_sgp_wrapper, training_structure, lj_calc):
    training_structure.calc = lj_calc
    forces = training_structure.get_forces()
    energy = training_structure.get_potential_energy()
    stress = training_structure.get_stress()
    bare_sgp_wrapper.update_db(training_structure, forces,
                               custom_range=(1, 2, 3, 4, 5),
                               energy=energy,
                               stress=stress,
                               mode="specific")

    return bare_sgp_wrapper


@pytest.fixture(scope="module")
def sgp_calc(sgp_wrapper):
    sgp_calc = SGP_Calculator(sgp_wrapper)

    return sgp_calc


def test_update_db(bare_sgp_wrapper, training_structure, lj_calc):
    """Check that the covariance matrices have the correct size after the
    sparse GP is updated."""

    custom_range = [1, 2, 3]
    training_structure.calc = lj_calc
    forces = training_structure.get_forces()
    energy = training_structure.get_potential_energy()
    stress = training_structure.get_stress()
    bare_sgp_wrapper.update_db(
        training_structure, forces, custom_range, energy, stress,
        mode="specific"
    )

    n_envs = len(custom_range)
    n_atoms = len(training_structure)
    assert bare_sgp_wrapper.sparse_gp.Kuu.shape[0] == n_envs
    assert bare_sgp_wrapper.sparse_gp.Kuf.shape[1] == 1 + n_atoms * 3 + 6


def test_train(sgp_wrapper):
    """Check that the hyperparameters and likelihood are updated when the
    train method is called."""

    hyps_init = tuple(sgp_wrapper.hyps)
    sgp_wrapper.train()
    hyps_post = tuple(sgp_wrapper.hyps)

    assert hyps_init != hyps_post
    assert sgp_wrapper.likelihood != 0.0


def test_dict(sgp_wrapper):
    """
    Check the method from_dict and as_dict
    """

    out_dict = sgp_wrapper.as_dict()
    assert len(sgp_wrapper) == len(out_dict["training_structures"])
    new_sgp, _ = SGP_Wrapper.from_dict(out_dict)
    assert len(sgp_wrapper) == len(new_sgp)
    assert len(sgp_wrapper.sparse_gp.kernels) == len(new_sgp.sparse_gp.kernels)
    assert np.allclose(sgp_wrapper.hyps, new_sgp.hyps)


def test_dump(sgp_wrapper):
    """
    Check the method from_file and write_model of SGP_Wrapper
    """

    sgp_wrapper.write_model("sgp.json")
    new_sgp, _ = SGP_Wrapper.from_file("sgp.json")
    assert len(sgp_wrapper) == len(new_sgp)
    assert len(sgp_wrapper.sparse_gp.kernels) == len(new_sgp.sparse_gp.kernels)
    assert np.allclose(sgp_wrapper.hyps, new_sgp.hyps)
    os.remove("sgp.json")


def test_calc(sgp_wrapper):
    """
    Check the method from_file and write_model of SGP_Calculator
    """

    calc = SGP_Calculator(sgp_wrapper)
    calc.write_model("sgp_calc.json")
    new_calc = SGP_Calculator.from_file("sgp_calc.json")
    assert len(calc.gp_model) == len(new_calc.gp_model)
    os.remove("sgp_calc.json")


def test_write_model(sgp_calc, training_structure):
    """Test that a reconstructed SGP calculator predicts the same forces
    as the original."""

    # Predict on training structure.
    training_structure.calc = sgp_calc
    forces = training_structure.get_forces()
    print(forces)

    # Write the SGP to JSON.
    sgp_name = "sgp_calc.json"
    sgp_calc.write_model(sgp_name)

    # Odd Pybind-related bug here that seems to be caused by kernel pointers.
    # Possibly related to: https://stackoverflow.com/questions/49633990/polymorphism-and-pybind11

    # Load the SGP.
    with open(sgp_name, "r") as f:
        gp_dict = json.loads(f.readline())
    sgp, _ = SGP_Wrapper.from_dict(gp_dict["gp_model"])
    sgp_calc_2 = SGP_Calculator(sgp)

    # sgp_calc_2 = SGP_Calculator.from_file(sgp_name)

    os.remove(sgp_name)

    # Compute forces with reconstructed SGP.
    training_structure.calc = sgp_calc_2
    forces_2 = training_structure.get_forces()

    # Check that they're the same.
    max_abs_diff = np.max(np.abs(forces - forces_2))
    assert max_abs_diff < 1e-8
