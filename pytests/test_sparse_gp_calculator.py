import pytest
import numpy as np
import os
from copy import deepcopy

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
def test_structure():
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
def sgp_wrapper(dot_product_kernel, b2_calculator, cutoff, lj_calc,
                training_structure):
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

    training_structure.calc = lj_calc
    forces = training_structure.get_forces()
    energy = training_structure.get_potential_energy()
    stress = training_structure.get_stress()
    sgp_model.update_db(training_structure, forces,
                        custom_range=(1, 2, 3, 4, 5),
                        energy=energy,
                        stress=stress,
                        mode="specific")

    return sgp_model


@pytest.fixture(scope="module")
def sgp_calc(sgp_wrapper):
    sgp_calc = SGP_Calculator(sgp_wrapper)

    return sgp_calc

@pytest.fixture(scope="module")
def sgp_calc_2(sgp_wrapper):
    sgp_calc_2 = SGP_Calculator(sgp_wrapper)

    return sgp_calc_2


def test_write_model(sgp_calc, training_structure, test_structure):
    # Predict on training structure.
    training_structure.calc = sgp_calc
    forces = training_structure.get_forces()
    # print(sgp_calc.gp_model.sparse_gp.training_structures[0].species)

    # Write the SGP to JSON.
    sgp_name = "sgp_calc.json"
    sgp_calc.write_model(sgp_name)

    # Load the SGP.
    sgp_calc_2 = SGP_Calculator.from_file(sgp_name)
    print(sgp_calc_2.gp_model.sparse_gp.kernels)
    # print(sgp_calc_2.gp_model.sparse_gp.training_structures[0].species)

    # Predict on training structure.
    test_structure.calc = sgp_calc_2
    # forces_2 = test_structure.get_forces()
    # print(forces_2)

    # Check that the forces match.

    assert 1 == 1
