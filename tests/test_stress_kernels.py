import pytest
from flare.kernels.stress_kernels import two_body_energy
from flare.kernels.two_body_mc_simple import TwoBodyKernel
from flare.kernels.three_body_mc_simple import ThreeBodyKernel
from flare.cutoffs import quadratic_cutoff
from flare import struc, env
import numpy as np


@pytest.fixture(scope='module')
def perturbed_envs():
    # Create an arbitrary structure.
    n_atoms = 4
    cell = np.eye(3) * 10
    np.random.seed(0)
    positions = np.random.rand(n_atoms, 3)
    species = [0, 1, 1, 0]
    structure = struc.Structure(cell, species, positions)

    # Take two environments from the structure.
    cutoffs = np.array([1.5, 1.5])
    test_env_1 = env.AtomicEnvironment(structure, 0, cutoffs)
    test_env_2 = env.AtomicEnvironment(structure, 3, cutoffs)
    environments = [test_env_1, test_env_2]

    # Perturb the central atoms.
    force_environments = [[], [], [], []]
    delta = 1e-4
    for n in range(3):
        positions_pert0 = np.copy(positions)
        positions_pert0_down = np.copy(positions)
        positions_pert3 = np.copy(positions)
        positions_pert3_down = np.copy(positions)

        positions_pert0[0, n] += delta
        positions_pert0_down[0, n] -= delta
        positions_pert3[3, n] += delta
        positions_pert3_down[3, n] -= delta

        test_struc0_pert = struc.Structure(cell, species, positions_pert0)
        test_struc0_down = struc.Structure(cell, species,
                                           positions_pert0_down)
        test_struc3_pert = struc.Structure(cell, species, positions_pert3)
        test_struc3_down = struc.Structure(cell, species,
                                           positions_pert3_down)

        env_pert_1 = env.AtomicEnvironment(test_struc0_pert, 0, cutoffs)
        env_pert_2 = env.AtomicEnvironment(test_struc3_pert, 3, cutoffs)
        env_pert_3 = env.AtomicEnvironment(test_struc0_down, 0, cutoffs)
        env_pert_4 = env.AtomicEnvironment(test_struc3_down, 3, cutoffs)

        force_environments[0].append(env_pert_1)
        force_environments[1].append(env_pert_2)
        force_environments[2].append(env_pert_3)
        force_environments[3].append(env_pert_4)

    # Strain the environments.
    stress_environments = [[], [], [], []]
    for m in range(3):
        for n in range(m, 3):
            cell_pert = np.copy(cell)
            cell_pert_down = np.copy(cell)
            positions_pert = np.copy(positions)
            positions_pert_down = np.copy(positions)

            # Strain the cell.
            for p in range(3):
                cell_pert[p, m] += cell[p, n] * delta
                cell_pert_down[p, m] -= cell[p, n] * delta

            # Strain the positions.
            for k in range(n_atoms):
                positions_pert[k, m] += positions[k, n] * delta
                positions_pert_down[k, m] -= positions[k, n] * delta

            test_struc_pert = \
                struc.Structure(cell_pert, species, positions_pert)
            test_struc_down = \
                struc.Structure(cell_pert_down, species, positions_pert_down)
            env_pert_1 = env.AtomicEnvironment(test_struc_pert, 0, cutoffs)
            env_pert_2 = env.AtomicEnvironment(test_struc_pert, 3, cutoffs)
            env_pert_3 = env.AtomicEnvironment(test_struc_down, 0, cutoffs)
            env_pert_4 = env.AtomicEnvironment(test_struc_down, 3, cutoffs)

            stress_environments[0].append(env_pert_1)
            stress_environments[1].append(env_pert_2)
            stress_environments[2].append(env_pert_3)
            stress_environments[3].append(env_pert_4)

    yield environments, force_environments, stress_environments, delta


def test_kernel(perturbed_envs):
    # Retrieve perturbed environments.
    environments = perturbed_envs[0]
    force_environments = perturbed_envs[1]
    stress_environments = perturbed_envs[2]
    delta = perturbed_envs[3]

    test_env_1 = environments[0]
    test_env_2 = environments[1]

    # Define kernel. (Generalize this later.)
    signal_variance = 1.
    length_scale = 1.
    hyperparameters = np.array([signal_variance, length_scale])
    cutoff = 1.5

    kernel = TwoBodyKernel(hyperparameters, cutoff)
    mult_fac = 2
    # kernel = ThreeBodyKernel(hyperparameters, cutoff)
    # mult_fac = 3

    # Set the test threshold.
    threshold = 1e-4

    # Compute kernels.
    energy_energy_kernel = kernel.energy_energy(test_env_1, test_env_2)
    force_energy_kernel = kernel.force_energy(test_env_2, test_env_1)
    stress_energy_kernel = kernel.stress_energy(test_env_2, test_env_1)
    # force_force_kernel = kernel.force_force(test_env_1, test_env_2)
    # stress_force_kernel = kernel.stress_force(test_env_1, test_env_2)
    # stress_stress_kernel = kernel.stress_stress(test_env_1, test_env_2)
    # force_force_gradient = kernel.force_force_gradient(test_env_1, test_env_2)

    # Check that the unit test isn't trivial.
    passive_aggressive_string = 'This unit test is trivial.'
    assert energy_energy_kernel != 1, passive_aggressive_string
    assert (force_energy_kernel != 0).all(), passive_aggressive_string

    # Check force/energy kernel by finite difference.
    for n in range(3):
        env_pert_up = force_environments[1][n]
        env_pert_down = force_environments[3][n]

        kern_pert_up = \
            kernel.energy_energy(test_env_1, env_pert_up)
        kern_pert_down = \
            kernel.energy_energy(test_env_1, env_pert_down)
        finite_diff_val = -(kern_pert_up - kern_pert_down) / (2 * delta)

        assert np.abs(finite_diff_val - force_energy_kernel[n] / mult_fac) < \
            threshold, 'Your force/energy kernel is wrong.'

    # Check stress/energy kernel by finite difference.
    for n in range(6):
        env_pert_up = stress_environments[1][n]
        env_pert_down = stress_environments[3][n]
        kern_pert_up = \
            kernel.energy_energy(test_env_1, env_pert_up)
        kern_pert_down = \
            kernel.energy_energy(test_env_1, env_pert_down)
        finite_diff_val = -(kern_pert_up - kern_pert_down) / (2 * delta)

        assert np.abs(finite_diff_val - stress_energy_kernel[n]) < \
            threshold, 'The stress/energy kernel is wrong.'

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
