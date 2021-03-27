import os
import numpy as np
import pytest
import sys

from ase.io import read
from flare import struc
from flare.lammps import lammps_calculator

from flare_pp.sparse_gp import SGP_Wrapper
from flare_pp._C_flare import NormalizedDotProduct, B2, SparseGP, Structure


np.random.seed(10)

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
custom_range = [1, 2, 3]
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
power = 2
kernel = NormalizedDotProduct(sigma, power)
cutoff_function = "quadratic"
cutoff = 1.5
many_body_cutoffs = [cutoff]
radial_basis = "chebyshev"
radial_hyps = [0.0, cutoff]
cutoff_hyps = []
settings = [len(atom_types), 4, 3] 
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

#    sgp_py.update_db(
#        train_structure, forces, custom_range, energy, stress, mode="specific"
#    )

    sgp_py.update_db(
        train_structure, forces, [3], energy, stress, mode="uncertain"
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
    sgp_py.write_mapping_coefficients("lmp.flare", "A", 0)
    from fln.utils import get_ase_lmp_calc
    lmp_calc = get_ase_lmp_calc(
        ff_preset="flare_pp", 
        specorder=["H", "He"], 
        coeff_dir="./", 
        lmp_command=os.environ.get("lmp"),
    ) 
    test_atoms = test_structure.to_ase_atoms()
    test_atoms.calc = lmp_calc
    lmp_f = test_atoms.get_forces()
    lmp_e = test_atoms.get_potential_energy()
    lmp_s = test_atoms.get_stress()


    new_kern = sgp_py.write_varmap_coefficients("beta_var.txt", "B", 0) # here the new kernel needs to be returned, otherwise the kernel won't be found in the current module

    assert sgp_py.sparse_gp.sparse_indices[0] == sgp_py.sgp_var.sparse_indices[0], \
            "the sparse_gp and sgp_var don't have the same training data"

    for s in range(len(atom_types)):
        org_desc = sgp_py.sparse_gp.sparse_descriptors[0].descriptors[s]
        new_desc = sgp_py.sgp_var.sparse_descriptors[0].descriptors[s]
        if not np.allclose(org_desc, new_desc): # the atomic order might change
            assert np.allclose(org_desc.shape, new_desc.shape)
            for i in range(org_desc.shape[0]):
                flag = False
                for j in range(new_desc.shape[0]): # seek in new_desc for matching of org_desc
                    if np.allclose(org_desc[i], new_desc[j]):
                        flag = True
                        break
                assert flag, "the sparse_gp and sgp_var don't have the same descriptors"

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

    print("GP predicting")
    sgp_py.sparse_gp.predict_DTC(test_cpp_struc)
    sgp_efs = test_cpp_struc.mean_efs

    print("Forces")
    sgp_forces = np.reshape(sgp_efs[1 : len(sgp_efs) - 6], (test_structure.nat, 3))
    print(np.concatenate([lmp_f, sgp_forces], axis=1))
    assert np.allclose(lmp_f, sgp_forces)
    print("GP forces match LMP forces")

    print("Stress")
    lmp_s_ordered = - lmp_s[[0, 5, 4, 1, 3, 2]]
    print(lmp_s[[0, 5, 4, 1, 3, 2]])
    print(sgp_efs[-6:])
    assert np.allclose(lmp_s_ordered, sgp_efs[-6:])

    print("Energy")
    print(lmp_e, sgp_efs[0])
    assert np.allclose(lmp_e, sgp_efs[0])

    # set up input and data files
    data_file_name = "tmp.data"
    lammps_location = "beta.txt"
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
#    lmp_std = lmp_dump.get_array("c_std")
#    lmp_var = lmp_std ** 2
#    print(lmp_var)

    sgp_py.sgp_var.predict_DTC(test_cpp_struc)
    #sgp_var = np.reshape(test_cpp_struc_pow1.variance_efs[1:len(sgp_efs)-6], (test_structure.nat, 3))
    sgp_var = sgp_py.sgp_var.compute_cluster_uncertainties(test_cpp_struc)
#    sgp_var = test_cpp_struc.variance_efs[0]
    print(sgp_var)

    n_descriptors = np.shape(test_cpp_struc.descriptors[0].descriptors[0])[1]
    n_species = len(atom_types)
    desc_en = np.zeros((n_species, n_descriptors))
    py_var = 0.0
#    for s in range(n_species):
#        n_struc_s = test_cpp_struc_pow1.descriptors[0].n_clusters_by_type[s];
#        assert np.shape(test_cpp_struc_pow1.descriptors[0].descriptors[s])[0] == n_struc_s
#        for i in range(n_descriptors):
#            desc_en[s, i] = np.sum(test_cpp_struc_pow1.descriptors[0].descriptors[s][:, i] / test_cpp_struc_pow1.descriptors[0].descriptor_norms[s], axis=0)
#
    print("len(varmap_coeffs)", len(sgp_py.sgp_var.varmap_coeffs))
    beta_matrices = []
    for s1 in range(n_species):
    #    for s2 in range(n_species):
        beta_matr = np.reshape(sgp_py.sgp_var.varmap_coeffs[s1, :], (n_descriptors, n_descriptors))
        print(s1, s1, "max", np.max(np.abs(beta_matr)), beta_matr.shape)
        beta_matrices.append(beta_matr)
#        py_var += desc_en[s1, :].dot(desc_en[s1, :].dot(beta_matr))
#    print(py_var)

    struc_desc = test_cpp_struc.descriptors[0]
    n_descriptors = np.shape(struc_desc.descriptors[0])[1]
    n_species = len(atom_types)
#    desc = np.zeros((n_atoms, 3, n_species, n_descriptors))

    print("len(varmap_coeffs)", len(sgp_py.sgp_var.varmap_coeffs))
    py_var = np.zeros(n_atoms)
    for s in range(n_species):
        n_struc = struc_desc.n_clusters_by_type[s];
        beta_matr = beta_matrices[s]
        for j in range(n_struc):
            norm = struc_desc.descriptor_norms[s][j]
            n_neigh = struc_desc.neighbor_counts[s][j]
            c_neigh = struc_desc.cumulative_neighbor_counts[s][j]
            atom_index = struc_desc.atom_indices[s][j]
            print("atom", atom_index, "n_neigh", n_neigh)
            desc = struc_desc.descriptors[s][j]
            py_var[atom_index] = desc.dot(desc.dot(beta_matr)) / norm ** 2
            #py_var[atom_index] = desc.dot(desc) / norm ** 2 * sgp_py.sgp_var.kernels[0].sigma ** 2 
#            for k in range(n_neigh):
#                ind = c_neigh + k
#                neighbor_index = struc_desc.neighbor_indices[s][ind];
#
#                neighbor_coord = struc_desc.neighbor_coordinates[s][ind];
#                neigh_dist = np.sum((neighbor_coord - test_positions[atom_index]) ** 2) 
#                #print("neighbor", neighbor_index, "dist", neigh_dist)
#                for comp in range(3):
#                    f1 = struc_desc.descriptor_force_dervs[s][3 * ind + comp] / norm
#                    f2 = struc_desc.descriptors[s][j] * struc_desc.descriptor_force_dots[s][3 * ind + comp] / norm ** 3
#                    f3 = f1 - f2
#                    desc[atom_index, comp, s, :] += norm #f3
#                    desc[neighbor_index, comp, s, :] -= norm #f3
#
#
#    for i in range(n_atoms):
#        for comp in range(3):
#            for s1 in range(n_species):
#                for s2 in range(n_species):
#                    beta_matr = np.reshape(sgp_py.sgp_var.varmap_coeffs[s1 * n_species + s2, :], (n_descriptors, n_descriptors))
##                    print(s1, s2, "max", np.max(np.abs(beta_matr)))
#                    py_var[i, comp] += desc[i, comp, s1].dot(desc[i, comp, s2].dot(beta_matr))
##                    py_var[i, comp] += np.sum(desc[i, comp, s1]) #np.sum(beta_matr[:, comp]) #np.sum(desc[i, comp, s1])
    print(py_var)

    #for i in range(4):
    #    print([sgp_py.sparse_gp.varmap_coeffs[0, n_descriptors * i + j] for j in range(3)])
    #for i in range(3):
    #    print(np.sum(test_cpp_struc_pow1.descriptors[0].descriptors[0][:, i] / test_cpp_struc_pow1.descriptors[0].descriptor_norms[0], axis=0))
