import numpy as np
from flare import struc, energy_gp_algebra, mc_simple, energy_gp
from flare.energy_gp_algebra import get_ky_block, get_ky_mat

cell = np.eye(3) * 10
pos1 = np.array([[0, 0, 0], [0.1, 0.2, 0.3], [-0.1, -0.14, 0.5]])
pos2 = np.array([[0, 0, 0], [0.25, 0.5, 0.1]])
species1 = [1, 2, 1]
species2 = [1, 2]

kernel = mc_simple.two_plus_three_body_mc
energy_kernel = mc_simple.two_plus_three_mc_en
force_energy_kernel = mc_simple.two_plus_three_mc_force_en
kernel_grad = None
cutoffs = np.array([4., 3.])
hyps = np.array([0.1, 1, 0.01, 1, 0.01, 0.001])

energy1 = 5
energy2 = 2
forces1 = np.array([[-1, -2, -3], [2, 5, 3], [0, 1, 2]])
forces2 = np.array([[3, 1, 4], [5, 2, 6]])

struc1 = struc.Structure(cell, species1, pos1, forces=forces1)
struc2 = struc.Structure(cell, species2, pos2, energy=energy2)


# test gp construction
en_gp = energy_gp.EnergyGP(kernel, force_energy_kernel, energy_kernel,
                           kernel_grad, hyps, cutoffs)


# test database update
en_gp.update_db(struc1)
en_gp.update_db(struc2)

# test structure comparison
block = get_ky_block(hyps, en_gp.training_strucs[0], en_gp.training_envs[0],
                     en_gp.training_atoms[0], en_gp.training_strucs[1],
                     en_gp.training_envs[1], en_gp.training_atoms[1],
                     kernel, force_energy_kernel, energy_kernel, cutoffs)

# test covariance matrix
k_test = get_ky_mat(hyps, en_gp.training_strucs, en_gp.training_envs,
                    en_gp.training_atoms, en_gp.training_labels_np,
                    kernel, force_energy_kernel, energy_kernel, cutoffs)

assert(np.isclose(k_test, k_test.transpose()).all())
