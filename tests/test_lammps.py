import pytest
import numpy as np
from flare.struc import Structure
from ase.calculators.lammpsrun import LAMMPS
from flare.utils.element_coder import _Z_to_mass, _Z_to_element, _element_to_Z
import os

from .get_sgp import get_sgp_calc, get_random_atoms


@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found in environment: Please install LAMMPS and set the "
        "$lmp environment variable to point to the executatble."
    ),
)
def test_write_potential():
    """Test the flare_pp pair style."""

    # Write potential file.
    sgp_model = get_sgp_calc()
    potential_name = "LJ.txt"
    contributor = "Jon"
    kernel_index = 0
    sgp_model.gp_model.sparse_gp.write_mapping_coefficients(
        potential_name, contributor, kernel_index)

    # Predict with SGP.
    test_structure = get_random_atoms()
    test_structure.calc = sgp_model
    energy = test_structure.get_potential_energy()
    forces = test_structure.get_forces()
    stress = test_structure.get_stress()

    # Set up LAMMPS calculator.
    species = ["C", "O"]
    parameters = {
        "command": os.environ.get("lmp"),  # set up executable for ASE
        "newton": "on",
        "pair_style": "flare",
        "pair_coeff": ["* * LJ.txt"],
    }

    lmp_calc = LAMMPS(tmp_dir="./tmp/", parameters=parameters,
                      files=[potential_name], specorder=species)

    # Predict with LAMMPS.
    test_structure.calc = lmp_calc
    energy_lmp = test_structure.get_potential_energy()
    forces_lmp = test_structure.get_forces()
    stress_lmp = test_structure.get_stress()

    thresh = 1e-6
    assert(np.abs(energy - energy_lmp) < thresh)
    assert(np.max(np.abs(forces - forces_lmp)) < thresh)
    assert(np.max(np.abs(stress - stress_lmp)) < thresh)

    # Remove files.
    os.remove(potential_name)
    os.system("rm -r tmp")
