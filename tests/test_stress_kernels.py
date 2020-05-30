import pytest
from flare.kernels.stress_kernels import two_body_energy
from flare.kernels.two_body_mc_simple import TwoBodyKernel
from flare.kernels.three_body_mc_simple import ThreeBodyKernel
from flare.cutoffs import quadratic_cutoff
from flare import struc, env
import numpy as np


@pytest.fixture(scope='module')
def perturbed_envs():
    # Create two random structures.
    n_atoms = 4
    cell = np.eye(3) * 10
    np.random.seed(0)
    positions_1 = np.random.rand(n_atoms, 3)
    positions_2 = np.random.rand(n_atoms, 3)
    species = [0, 1, 1, 0]
    structure_1 = struc.Structure(cell, species, positions_1)
    structure_2 = struc.Structure(cell, species, positions_2)
    strucs = [structure_1, structure_2]

    # Record the environments of the structures.
    cutoffs = np.array([1.5, 1.5])
    struc_envs = []
    for structure in strucs:
        envs_curr = []
        for n in range(n_atoms):
            env_curr = env.AtomicEnvironment(structure, n, cutoffs)
            envs_curr.append(env_curr)
        struc_envs.append(envs_curr)

    # Perturb atom 0 in both structures up and down and in all directions.
    delta = 1e-4
    signs = [1, -1]
    dims = [0, 1, 2]
    force_envs = []
    for structure in strucs:
        sign_envs_curr = []
        for sign in signs:
            dim_envs_curr = []
            for dim in dims:
                positions_pert = np.copy(structure.positions)
                positions_pert[0, dim] += delta * sign
                struc_pert = struc.Structure(cell, species, positions_pert)
                atom_envs = []
                for n in range(n_atoms):
                    env_curr = env.AtomicEnvironment(struc_pert, n, cutoffs)
                    atom_envs.append(env_curr)
                dim_envs_curr.append(atom_envs)
            sign_envs_curr.append(dim_envs_curr)
        force_envs.append(sign_envs_curr)

    # Strain both structures up and down and in all directions.
    stress_envs = []

    for structure in strucs:
        sign_envs_curr = []
        for sign in signs:
            strain_envs_curr = []
            for m in range(3):
                for n in range(m, 3):
                    cell_pert = np.copy(structure.cell)
                    positions_pert = np.copy(structure.positions)

                    # Strain the cell.
                    for p in range(3):
                        cell_pert[p, m] += structure.cell[p, n] * delta * sign

                    # Strain the positions.
                    for k in range(n_atoms):
                        positions_pert[k, m] += \
                            structure.positions[k, n] * delta * sign

                    struc_pert = \
                        struc.Structure(cell_pert, species, positions_pert)

                    atom_envs = []
                    for q in range(n_atoms):
                        env_curr = \
                            env.AtomicEnvironment(struc_pert, q, cutoffs)
                        atom_envs.append(env_curr)
                    strain_envs_curr.append(atom_envs)
            sign_envs_curr.append(strain_envs_curr)
        stress_envs.append(sign_envs_curr)

    yield struc_envs, force_envs, stress_envs, delta


def test_kernel(perturbed_envs):
    """Check that kernel derivatives are implemented correctly."""

    # Retrieve perturbed environments.
    struc_envs = perturbed_envs[0]
    force_envs = perturbed_envs[1]
    stress_envs = perturbed_envs[2]
    delta = perturbed_envs[3]

    # Define kernel. (Generalize this later.)
    signal_variance = 1.
    length_scale = 1.
    hyperparameters = np.array([signal_variance, length_scale])
    cutoff = 1.5

    # kernel = TwoBodyKernel(hyperparameters, cutoff)
    kernel = ThreeBodyKernel(hyperparameters, cutoff)

    # Set the test threshold.
    threshold = 1e-4

    # Check force/energy kernel.
    force_en_exact = np.zeros(3)
    force_en_finite = np.zeros(3)
    perturbed_env = struc_envs[0][0]
    for m in range(len(struc_envs[1])):
        force_en_exact += kernel.force_energy(perturbed_env, struc_envs[1][m])

        for n in range(3):
            for p in range(len(struc_envs[0])):
                env_pert_up = force_envs[0][0][n][p]
                env_pert_down = force_envs[0][1][n][p]

                kern_pert_up = \
                    kernel.energy_energy(env_pert_up, struc_envs[1][m])
                kern_pert_down = \
                    kernel.energy_energy(env_pert_down, struc_envs[1][m])
                force_en_finite[n] +=\
                    -(kern_pert_up - kern_pert_down) / (2 * delta)

    assert (np.abs(force_en_exact - force_en_finite) < threshold).all(), \
        'Your force/energy kernel is wrong.'

    # Check stress/energy kernel.
    stress_en_exact = np.zeros(6)
    stress_en_finite = np.zeros(6)
    perturbed_env = struc_envs[0][0]
    for m in range(len(struc_envs[0])):
        for n in range(len(struc_envs[1])):
            stress_en_exact += \
                kernel.stress_energy(struc_envs[0][m], struc_envs[1][n])

            for p in range(6):
                env_pert_up = stress_envs[0][0][p][m]
                env_pert_down = stress_envs[0][1][p][m]

                kern_pert_up = \
                    kernel.energy_energy(env_pert_up, struc_envs[1][n])
                kern_pert_down = \
                    kernel.energy_energy(env_pert_down, struc_envs[1][n])
                stress_en_finite[p] +=\
                    -(kern_pert_up - kern_pert_down) / (2 * delta)

    assert (np.abs(stress_en_exact - stress_en_finite) < threshold).all(), \
        'Your stress/energy kernel is wrong.'

    # assert np.abs(force_en_exact - force_en_finite).all() < threshold, \
    #     'Your force/energy kernel is wrong.'

    # # Check stress/energy kernel by finite difference.
    # for n in range(6):
    #     env_pert_up = stress_environments[1][n]
    #     env_pert_down = stress_environments[3][n]
    #     kern_pert_up = \
    #         kernel.energy_energy(test_env_1, env_pert_up)
    #     kern_pert_down = \
    #         kernel.energy_energy(test_env_1, env_pert_down)
    #     finite_diff_val = -(kern_pert_up - kern_pert_down) / (2 * delta)

    #     assert np.abs(finite_diff_val - stress_energy_kernel[n]) < \
    #         threshold, 'The stress/energy kernel is wrong.'

    # # Check force/force kernel by finite difference.
    # for m in range(3):
    #     pert1_up = force_environments[0][m]
    #     pert1_down = force_environments[2][m]
    #     for n in range(3):
    #         pert2_up = force_environments[1][n]
    #         pert2_down = force_environments[3][n]
    #         kern1 = kernel.energy_energy(pert1_up, pert2_up)
    #         kern2 = kernel.energy_energy(pert1_up, pert2_down)
    #         kern3 = kernel.energy_energy(pert1_down, pert2_up)
    #         kern4 = kernel.energy_energy(pert1_down, pert2_down)

    #         finite_diff_val = \
    #             (kern1 - kern2 - kern3 + kern4) / (4 * delta * delta)

    #         assert np.abs(finite_diff_val * 4 - force_force_kernel[m, n]) < \
    #             threshold, 'The force/force kernel is wrong.'

    # # Check stress/force kernel by finite difference.
    # for m in range(6):
    #     pert1_up = stress_environments[0][m]
    #     pert1_down = stress_environments[2][m]
    #     for n in range(3):
    #         pert2_up = force_environments[1][n]
    #         pert2_down = force_environments[3][n]
    #         kern1 = kernel.energy_energy(pert1_up, pert2_up)
    #         kern2 = kernel.energy_energy(pert1_up, pert2_down)
    #         kern3 = kernel.energy_energy(pert1_down, pert2_up)
    #         kern4 = kernel.energy_energy(pert1_down, pert2_down)

    #         finite_diff_val = \
    #             (kern1 - kern2 - kern3 + kern4) / (4 * delta * delta)

    #         assert np.abs(finite_diff_val * 2 - stress_force_kernel[m, n]) < \
    #             threshold, 'The stress/force kernel is wrong.'

    # # Check stress/stress kernel by finite difference.
    # for m in range(6):
    #     pert1_up = stress_environments[0][m]
    #     pert1_down = stress_environments[2][m]
    #     for n in range(6):
    #         pert2_up = stress_environments[1][n]
    #         pert2_down = stress_environments[3][n]
    #         kern1 = kernel.energy_energy(pert1_up, pert2_up)
    #         kern2 = kernel.energy_energy(pert1_up, pert2_down)
    #         kern3 = kernel.energy_energy(pert1_down, pert2_up)
    #         kern4 = kernel.energy_energy(pert1_down, pert2_down)

    #         finite_diff_val = \
    #             (kern1 - kern2 - kern3 + kern4) / (4 * delta * delta)

    #         assert np.abs(finite_diff_val - stress_stress_kernel[m, n]) < \
    #             threshold, 'The stress/stress kernel is wrong.'

    # # Check force/force gradient.
    # print(force_force_gradient[1])
    # kernel.signal_variance = signal_variance + delta
    # force_sig_up = kernel.force_force(test_env_1, test_env_2)

    # kernel.signal_variance = signal_variance - delta
    # force_sig_down = kernel.force_force(test_env_1, test_env_2)

    # kernel.signal_variance = signal_variance
    # kernel.length_scale = length_scale + delta
    # force_ls_up = kernel.force_force(test_env_1, test_env_2)

    # kernel.length_scale = length_scale - delta
    # force_ls_down = kernel.force_force(test_env_1, test_env_2)

    # for m in range(3):
    #     for n in range(3):
    #         sig_val = \
    #             (force_sig_up[m, n] - force_sig_down[m, n]) / (2 * delta)

    #         ls_val = \
    #             (force_ls_up[m, n] - force_ls_down[m, n]) / (2 * delta)

    #         assert np.abs(sig_val -
    #                       force_force_gradient[1][0, m, n]) < \
    #             threshold, 'The force/force gradient is wrong.'

    #         assert np.abs(ls_val -
    #                       force_force_gradient[1][1, m, n]) < \
    #             threshold, 'The force/force gradient is wrong.'


# def test_stress_energy(perturbed_envs):
#     # Retrieve perturbed environments.
#     environments = perturbed_envs[0]
#     force_environments = perturbed_envs[1]
#     stress_environments = perturbed_envs[2]

#     test_env_1 = environments[0]
#     test_env_2 = environments[1]

#     # Compute energy, force, and stress kernels.
#     sig = 1.
#     ls = 1.
#     r_cut = 5.
#     cutoff_func = quadratic_cutoff

#     energy_kernel, force_kernels, stress_kernels = \
#         two_body_energy(test_env_1.bond_array_2, test_env_1.ctype,
#                         test_env_1.etypes, test_env_2.bond_array_2,
#                         test_env_2.ctype, test_env_2.etypes,
#                         sig, ls, r_cut, cutoff_func)

#     # Check force kernels by finite difference.
#     delta = 1e-4
#     stress_count = 0
#     for n in range(3):
#         env_pert = force_environments[1][n]

#         kern_pert, _, _ = \
#             two_body_energy(test_env_1.bond_array_2, test_env_1.ctype,
#                             test_env_1.etypes, env_pert.bond_array_2,
#                             env_pert.ctype, env_pert.etypes,
#                             sig, ls, r_cut, cutoff_func)
#         finite_diff_val = -(kern_pert - energy_kernel) / delta

#         assert(kern_pert != 0)
#         assert(np.abs(finite_diff_val - force_kernels[n] / 2) < 1e-3)

#     # Check stress kernels by finite difference.
#     delta = 1e-4
#     stress_count = 0
#     for m in range(3):
#         for n in range(m, 3):
#             env_pert = stress_environments[1][stress_count]

#             kern_pert, _, _ = \
#                 two_body_energy(test_env_1.bond_array_2, test_env_1.ctype,
#                                 test_env_1.etypes, env_pert.bond_array_2,
#                                 env_pert.ctype, env_pert.etypes,
#                                 sig, ls, r_cut, cutoff_func)
#             finite_diff_val = -(kern_pert - energy_kernel) / delta

#             assert(kern_pert != 0)
#             assert(np.abs(finite_diff_val - stress_kernels[stress_count])
#                    < 1e-3)
#             stress_count += 1
