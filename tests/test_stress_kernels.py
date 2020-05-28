import pytest
from flare.kernels.stress_kernels import two_body_energy
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
    cutoffs = np.array([5., 5.])
    test_env_1 = env.AtomicEnvironment(structure, 0, cutoffs)
    test_env_2 = env.AtomicEnvironment(structure, 3, cutoffs)
    environments = [test_env_1, test_env_2]

    # Perturb the central atoms.
    force_environments = [[], []]
    delta = 1e-8
    for n in range(3):
        positions_pert = np.copy(positions)
        positions_pert[3, n] += delta

        test_struc_pert = struc.Structure(cell, species, positions_pert)
        env_pert_1 = env.AtomicEnvironment(test_struc_pert, 0, cutoffs)
        env_pert_2 = env.AtomicEnvironment(test_struc_pert, 3, cutoffs)

        force_environments[0].append(env_pert_1)
        force_environments[1].append(env_pert_2)

    # Strain the environments.
    stress_environments = [[], []]
    for m in range(3):
        for n in range(m, 3):
            cell_pert = np.copy(cell)
            positions_pert = np.copy(positions)

            # Strain the cell.
            cell_pert[0, m] += cell[0, n] * delta
            cell_pert[1, m] += cell[1, n] * delta
            cell_pert[2, m] += cell[2, n] * delta

            # Strain the positions.
            for k in range(n_atoms):
                positions_pert[k, m] += positions[k, n] * delta

            test_struc_pert = \
                struc.Structure(cell_pert, species, positions_pert)
            env_pert_1 = env.AtomicEnvironment(test_struc_pert, 0, cutoffs)
            env_pert_2 = env.AtomicEnvironment(test_struc_pert, 3, cutoffs)

            stress_environments[0].append(env_pert_1)
            stress_environments[1].append(env_pert_2)

    yield environments, force_environments, stress_environments


def test_stress_energy(perturbed_envs):
    # Retrieve perturbed environments.
    environments = perturbed_envs[0]
    force_environments = perturbed_envs[1]
    stress_environments = perturbed_envs[2]

    test_env_1 = environments[0]
    test_env_2 = environments[1]

    # Compute energy, force, and stress kernels.
    sig = 1.
    ls = 1.
    r_cut = 5.
    cutoff_func = quadratic_cutoff

    energy_kernel, force_kernels, stress_kernels = \
        two_body_energy(test_env_1.bond_array_2, test_env_1.ctype,
                        test_env_1.etypes, test_env_2.bond_array_2,
                        test_env_2.ctype, test_env_2.etypes,
                        sig, ls, r_cut, cutoff_func)

    # Check force kernels by finite difference.
    delta = 1e-8
    stress_count = 0
    for n in range(3):
        env_pert = force_environments[1][n]

        kern_pert, _, _ = \
            two_body_energy(test_env_1.bond_array_2, test_env_1.ctype,
                            test_env_1.etypes, env_pert.bond_array_2,
                            env_pert.ctype, env_pert.etypes,
                            sig, ls, r_cut, cutoff_func)
        finite_diff_val = -(kern_pert - energy_kernel) / delta

        assert(kern_pert != 0)
        assert(np.abs(finite_diff_val - force_kernels[n] / 2) < 1e-3)

    # Check stress kernels by finite difference.
    delta = 1e-8
    stress_count = 0
    for m in range(3):
        for n in range(m, 3):
            env_pert = stress_environments[1][stress_count]

            kern_pert, _, _ = \
                two_body_energy(test_env_1.bond_array_2, test_env_1.ctype,
                                test_env_1.etypes, env_pert.bond_array_2,
                                env_pert.ctype, env_pert.etypes,
                                sig, ls, r_cut, cutoff_func)
            finite_diff_val = -(kern_pert - energy_kernel) / delta

            assert(kern_pert != 0)
            assert(np.abs(finite_diff_val - stress_kernels[stress_count])
                   < 1e-3)
            stress_count += 1
