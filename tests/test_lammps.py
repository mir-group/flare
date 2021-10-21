import pytest
import numpy as np
from flare.lammps import lammps_calculator
from flare.struc import Structure
import os

from .get_sgp import get_updated_sgp, get_random_atoms


def test_write_potential():
    """Test the flare_pp pair style."""

    # Write potential file.
    sgp_model = get_updated_sgp()
    potential_name = "LJ.txt"
    contributor = "Jon"
    kernel_index = 0
    sgp_model.sparse_gp.write_mapping_coefficients(
        potential_name, contributor, kernel_index)
    os.remove(potential_name)


# TODO: Test that SGP and Lammps forces match.
# @pytest.mark.skipif(
#     not os.environ.get("lmp", False),
#     reason=(
#         "lmp not found in environment: Please install LAMMPS and set the "
#         "$lmp environment variable to point to the executatble."
#     ),
# )
