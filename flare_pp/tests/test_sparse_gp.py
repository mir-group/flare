import os
import numpy as np
import pytest
import sys

sys.path.append("../..")
sys.path.append("../../build")

from ase.io import read
from flare import struc
from flare.lammps import lammps_calculator

from flare_pp.sparse_gp import SGP_Wrapper
from _C_flare import NormalizedDotProduct, B2, SparseGP, Structure


# Make random structure.
n_atoms = 4
cell = np.eye(3)
train_positions = np.random.rand(n_atoms, 3)
test_positions = np.random.rand(n_atoms, 3)
atom_types = [1, 2]
atom_masses = [2, 4]
species = [1, 2, 1, 2]
train_structure = struc.Structure(cell, species, train_positions)
test_structure = struc.Structure(cell, species, test_positions)

# Test update db
custom_range = [1, 3]
energy = np.random.rand()
forces = np.random.rand(n_atoms, 3)
stress = np.random.rand(6)
np.savez(
    "random_data",
    train_pos=train_positions,
    test_pos=test_positions,
    energy=energy,
    forces=forces,
    stress=stress,
)

# Create sparse GP model.
sigma = 1.0
power = 1 
kernel = NormalizedDotProduct(1.0, power)
cutoff_function = "quadratic"
cutoff = 1.0
many_body_cutoffs = [cutoff]
radial_basis = "chebyshev"
radial_hyps = [0.0, cutoff]
cutoff_hyps = []
settings = [2, 4, 3]
calc = B2(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)
sigma_e = 1.0
sigma_f = 1.0
sigma_s = 1.0
species_map = {1: 0, 2: 1}
max_iterations = 20

sgp_cpp = SparseGP([kernel], sigma_e, sigma_f, sigma_s)
sgp_py = SGP_Wrapper(
    [kernel],
    [calc],
    cutoff,
    sigma_e,
    sigma_f,
    sigma_s,
    species_map,
    max_iterations=max_iterations,
)


def test_update_db():
    """Check that the covariance matrices have the correct size after the
    sparse GP is updated."""

    sgp_py.update_db(
        train_structure, forces, custom_range, energy, stress, mode="specific"
    )

    n_envs = len(custom_range)
    assert sgp_py.sparse_gp.Kuu.shape[0] == n_envs
    assert sgp_py.sparse_gp.Kuf.shape[1] == 1 + n_atoms * 3 + 6


def test_train():
    """Check that the hyperparameters and likelihood are updated when the
    train method is called."""

    hyps_init = tuple(sgp_py.hyps)
    sgp_py.train()
    hyps_post = tuple(sgp_py.hyps)

    assert hyps_init != hyps_post
    assert sgp_py.likelihood != 0.0



@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found "
        "in environment: Please install LAMMPS "
        "and set the $lmp env. "
        "variable to point to the executatble."
    ),
)
def test_lammps():
    sgp_py.write_mapping_coefficients("beta.txt", "A", 0)
    pow1_sgp = sgp_py.write_varmap_coefficients("beta_var.txt", "B", 0)

    # set up input and data files
    data_file_name = "tmp.data"
    lammps_location = "beta_2.txt"
    style_string = "flare"
    coeff_string = "* * {}".format(lammps_location)
    lammps_executable = os.environ.get("lmp")
    dump_file_name = "tmp.dump"
    input_file_name = "tmp.in"
    output_file_name = "tmp.out"
    newton = True

    # write data file
    data_text = lammps_calculator.lammps_dat(
        test_structure, atom_types, atom_masses, species
    )
    lammps_calculator.write_text(data_file_name, data_text)

    # write input file
    input_text = lammps_calculator.generic_lammps_input(
        data_file_name, style_string, coeff_string, dump_file_name, newton=newton,
        std_string="beta_var.txt", std_style="flare_pp",
    )
    lammps_calculator.write_text(input_file_name, input_text)

    # run lammps
    lammps_calculator.run_lammps(lammps_executable, input_file_name, output_file_name)

    # read output
    lmp_dump = read(dump_file_name, format="lammps-dump-text")
    lmp_forces = lmp_dump.get_forces()
    lmp_var_1 = lmp_dump.get_array("c_std[1]")
    lmp_var_2 = lmp_dump.get_array("c_std[2]")
    lmp_var_3 = lmp_dump.get_array("c_std[3]")
    lmp_var = np.hstack([lmp_var_1, lmp_var_2, lmp_var_3])
    print(lmp_var)

    # compare with sgp_py prediction
    assert len(sgp_py.training_data) > 0

    # Convert coded species to 0, 1, 2, etc.
    coded_species = []
    for spec in test_structure.coded_species:
        coded_species.append(species_map[spec])

    test_cpp_struc = Structure(
        test_structure.cell,
        coded_species,
        test_structure.positions,
        sgp_py.cutoff,
        sgp_py.descriptor_calculators,
    )

    sgp_py.sparse_gp.predict_DTC(test_cpp_struc)
    sgp_efs = test_cpp_struc.mean_efs
    sgp_forces = np.reshape(sgp_efs[1:len(sgp_efs)-6], (test_structure.nat, 3))
    assert np.allclose(lmp_forces, sgp_forces)

    test_cpp_struc_pow1 = Structure(
        test_structure.cell,
        coded_species,
        test_structure.positions,
        pow1_sgp.cutoff,
        pow1_sgp.descriptor_calculators,
    )
    pow1_sgp.sparse_gp.predict_DTC(test_cpp_struc_pow1)
    sgp_var = np.reshape(test_cpp_struc_pow1.variance_efs[1:len(sgp_efs)-6], (test_structure.nat, 3))
    print(sgp_var)
