import pytest
import numpy as np
import os
from copy import deepcopy
import json

from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.calculator import SGP_Calculator

from flare.bffs.gp.calculator import FLARE_Calculator
from flare.atoms import FLARE_Atoms

from ase.calculators.lj import LennardJones
from ase.build import bulk

from .get_sgp import get_random_atoms, get_empty_sgp, get_updated_sgp, get_sgp_calc


multiple_cutoff = [False, True]


@pytest.mark.parametrize("multicut", multiple_cutoff)
def test_update_db(multicut):
    """Check that the covariance matrices have the correct size after the
    sparse GP is updated."""

    # Create a labelled structure.
    custom_range = [1, 2, 3]
    training_structure = get_random_atoms()
    training_structure.calc = LennardJones()
    forces = training_structure.get_forces()
    energy = training_structure.get_potential_energy()
    stress = training_structure.get_stress()

    # Update the SGP.
    sgp = get_empty_sgp(multiple_cutoff=multicut)
    sgp.update_db(
        training_structure, forces, custom_range, energy, stress, mode="specific"
    )

    n_envs = len(custom_range)
    n_atoms = len(training_structure)
    assert sgp.sparse_gp.Kuu.shape[0] == n_envs
    assert sgp.sparse_gp.Kuf.shape[1] == 1 + n_atoms * 3 + 6


@pytest.mark.parametrize("multicut", multiple_cutoff)
def test_train(multicut):
    """Check that the hyperparameters and likelihood are updated when the
    train method is called."""

    sgp = get_updated_sgp(multiple_cutoff=multicut)
    hyps_init = tuple(sgp.hyps)
    sgp.train()
    hyps_post = tuple(sgp.hyps)

    assert hyps_init != hyps_post
    assert sgp.likelihood != 0.0


@pytest.mark.parametrize("multicut", multiple_cutoff)
def test_dict(multicut):
    """
    Check the method from_dict and as_dict
    """

    sgp_wrapper = get_updated_sgp(multiple_cutoff=multicut)
    out_dict = sgp_wrapper.as_dict()
    assert len(sgp_wrapper) == len(out_dict["training_structures"])
    new_sgp, _ = SGP_Wrapper.from_dict(out_dict)
    assert len(sgp_wrapper) == len(new_sgp)
    assert len(sgp_wrapper.sparse_gp.kernels) == len(new_sgp.sparse_gp.kernels)
    assert np.allclose(sgp_wrapper.hyps, new_sgp.hyps)


@pytest.mark.parametrize("multicut", multiple_cutoff)
def test_dump(multicut):
    """
    Check the method from_file and write_model of SGP_Wrapper
    """

    sgp_wrapper = get_updated_sgp(multiple_cutoff=multicut)
    sgp_wrapper.write_model(f"sgp_{multicut}.json")
    new_sgp, _ = SGP_Wrapper.from_file(f"sgp_{multicut}.json")
    os.remove(f"sgp_{multicut}.json")
    assert len(sgp_wrapper) == len(new_sgp)
    assert len(sgp_wrapper.sparse_gp.kernels) == len(new_sgp.sparse_gp.kernels)
    assert np.allclose(sgp_wrapper.hyps, new_sgp.hyps)


@pytest.mark.parametrize("multicut", multiple_cutoff)
def test_calc(multicut):
    """
    Check the method from_file and write_model of SGP_Calculator
    """

    sgp_wrapper = get_updated_sgp(multiple_cutoff=multicut)
    calc = SGP_Calculator(sgp_wrapper)
    calc.write_model(f"sgp_calc_{multicut}.json")
    new_calc, _ = SGP_Calculator.from_file(f"sgp_calc_{multicut}.json")
    os.remove(f"sgp_calc_{multicut}.json")
    assert len(calc.gp_model) == len(new_calc.gp_model)


@pytest.mark.parametrize("multicut", multiple_cutoff)
def test_write_model(multicut):
    """Test that a reconstructed SGP calculator predicts the same forces
    as the original."""

    training_structure = get_random_atoms()
    sgp_calc = get_sgp_calc(multiple_cutoff=multicut)

    # Predict on training structure.
    training_structure.calc = sgp_calc
    forces = training_structure.get_forces()

    # Write the SGP to JSON.
    sgp_name = f"sgp_calc_{multicut}.json"
    sgp_calc.write_model(sgp_name)

    # Odd Pybind-related issue here that seems to be related to polymorphic
    # kernel pointers. Need to return the kernel list for SGP prediction to
    # work. Possibly related to:
    # https://stackoverflow.com/questions/49633990/polymorphism-and-pybind11
    sgp_calc_2, _ = SGP_Calculator.from_file(sgp_name)

    os.remove(sgp_name)

    # Compute forces with reconstructed SGP.
    training_structure.calc = sgp_calc_2
    forces_2 = training_structure.get_forces()

    # Check that they're the same.
    max_abs_diff = np.max(np.abs(forces - forces_2))
    assert max_abs_diff < 1e-8
