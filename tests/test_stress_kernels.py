import pytest
from flare.kernels.two_body_mc_simple import TwoBodyKernel
from flare.kernels.three_body_mc_simple import ThreeBodyKernel
from flare.kernels.cutoffs import quadratic_cutoff
from flare.atoms import FLARE_Atoms
from flare.descriptors import env
import numpy as np


# Choose parameters for the random structures.
n_atoms = 5
cell = np.eye(3) * 10
species = [1, 3, 3, 1, 7]

# Define kernels to test.
signal_variance = 1.0
length_scale = 1.0
hyperparameters = np.array([signal_variance, length_scale])
cutoff = 1.5
cutoffs = np.array([cutoff, cutoff])

kernel_2b = TwoBodyKernel(hyperparameters, cutoff)
kernel_3b = ThreeBodyKernel(hyperparameters, cutoff)
kernel_list = [kernel_2b, kernel_3b]

# Set the perturbation size and test threshold.
delta = 1e-4
threshold = 1e-4


@pytest.fixture(scope="module")
def strucs():
    """Create two random structures."""

    np.random.seed(0)
    positions_1 = np.random.rand(n_atoms, 3)
    positions_2 = np.random.rand(n_atoms, 3)
    structure_1 = FLARE_Atoms(cell=cell, symbols=species, positions=positions_1)
    structure_2 = FLARE_Atoms(cell=cell, symbols=species, positions=positions_2)
    strucs = [structure_1, structure_2]

    yield strucs


@pytest.fixture(scope="module")
def struc_envs(strucs):
    """Store the environments of the random structures."""

    struc_envs = []
    for structure in strucs:
        envs_curr = []
        for n in range(structure.nat):
            env_curr = env.AtomicEnvironment(structure, n, cutoffs)
            envs_curr.append(env_curr)
        struc_envs.append(envs_curr)

    yield struc_envs


@pytest.fixture(scope="module")
def force_envs(strucs):
    """Perturb atom 0 in both structures up and down and in all directions."""

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
                struc_pert = FLARE_Atoms(
                    cell=structure.cell,
                    symbols=structure.numbers,
                    positions=positions_pert,
                )
                atom_envs = []
                for n in range(structure.nat):
                    env_curr = env.AtomicEnvironment(struc_pert, n, cutoffs)
                    atom_envs.append(env_curr)
                dim_envs_curr.append(atom_envs)
            sign_envs_curr.append(dim_envs_curr)
        force_envs.append(sign_envs_curr)

    yield force_envs


@pytest.fixture(scope="module")
def stress_envs(strucs):
    """Strain both structures up and down and in all directions."""

    stress_envs = []
    signs = [1, -1]
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
                    for k in range(structure.nat):
                        positions_pert[k, m] += structure.positions[k, n] * delta * sign

                    struc_pert = FLARE_Atoms(
                        cell=cell_pert,
                        symbols=structure.numbers,
                        positions=positions_pert,
                    )

                    atom_envs = []
                    for q in range(structure.nat):
                        env_curr = env.AtomicEnvironment(struc_pert, q, cutoffs)
                        atom_envs.append(env_curr)
                    strain_envs_curr.append(atom_envs)
            sign_envs_curr.append(strain_envs_curr)
        stress_envs.append(sign_envs_curr)

    yield stress_envs


@pytest.mark.parametrize("kernel", kernel_list)
def test_force_energy(struc_envs, force_envs, kernel):
    """Check that the force/energy kernel is implemented correctly."""

    # Check force/energy kernel.
    force_en_exact = np.zeros(3)
    force_en_finite = np.zeros(3)
    perturbed_env = struc_envs[0][0]
    for m in range(len(struc_envs[1])):
        force_en_exact += kernel.force_energy(perturbed_env, struc_envs[1][m])

        # Compute kernel by finite difference.
        for n in range(3):
            for p in range(len(struc_envs[0])):
                env_pert_up = force_envs[0][0][n][p]
                env_pert_down = force_envs[0][1][n][p]

                kern_pert_up = kernel.energy_energy(env_pert_up, struc_envs[1][m])
                kern_pert_down = kernel.energy_energy(env_pert_down, struc_envs[1][m])
                force_en_finite[n] += -(kern_pert_up - kern_pert_down) / (2 * delta)

    assert (
        np.abs(force_en_exact - force_en_finite) < threshold
    ).all(), "Your force/energy kernel is wrong."


@pytest.mark.parametrize("kernel", kernel_list)
def test_stress_energy(struc_envs, stress_envs, kernel):
    """Check that the stress/energy kernel is implemented correctly."""

    # Check stress/energy kernel.
    stress_en_exact = np.zeros(6)
    stress_en_finite = np.zeros(6)
    for m in range(len(struc_envs[0])):
        for n in range(len(struc_envs[1])):
            stress_en_exact += kernel.stress_energy(struc_envs[0][m], struc_envs[1][n])

            # Compute kernel by finite difference.
            for p in range(6):
                env_pert_up = stress_envs[0][0][p][m]
                env_pert_down = stress_envs[0][1][p][m]

                kern_pert_up = kernel.energy_energy(env_pert_up, struc_envs[1][n])
                kern_pert_down = kernel.energy_energy(env_pert_down, struc_envs[1][n])
                stress_en_finite[p] += -(kern_pert_up - kern_pert_down) / (2 * delta)

    assert (
        np.abs(stress_en_exact - stress_en_finite) < threshold
    ).all(), "Your stress/energy kernel is wrong."


@pytest.mark.parametrize("kernel", kernel_list)
def test_force_force(struc_envs, force_envs, kernel):
    """Check that the force/force kernel is implemented correctly."""

    force_force_exact = kernel.force_force(struc_envs[0][0], struc_envs[1][0])

    # Compute force/force kernels by finite difference.
    force_force_finite = np.zeros((3, 3))
    for m in range(len(struc_envs[0])):
        for n in range(len(struc_envs[1])):
            for p in range(3):
                pert1_up = force_envs[0][0][p][m]
                pert1_down = force_envs[0][1][p][m]

                for q in range(3):
                    pert2_up = force_envs[1][0][q][n]
                    pert2_down = force_envs[1][1][q][n]

                    kern1 = kernel.energy_energy(pert1_up, pert2_up)
                    kern2 = kernel.energy_energy(pert1_up, pert2_down)
                    kern3 = kernel.energy_energy(pert1_down, pert2_up)
                    kern4 = kernel.energy_energy(pert1_down, pert2_down)

                    force_force_finite[p, q] += (kern1 - kern2 - kern3 + kern4) / (
                        4 * delta * delta
                    )

    assert (
        np.abs(force_force_exact - force_force_finite) < threshold
    ).all(), "Your force/force kernel is wrong."


@pytest.mark.parametrize("kernel", kernel_list)
def test_stress_force(struc_envs, stress_envs, force_envs, kernel):
    """Check that the stress/force kernel is implemented correctly."""

    stress_force_exact = np.zeros((6, 3))
    stress_force_finite = np.zeros((6, 3))

    for m in range(len(struc_envs[0])):
        stress_force_exact += kernel.stress_force(struc_envs[0][m], struc_envs[1][0])
        for n in range(len(struc_envs[1])):
            for p in range(6):
                pert1_up = stress_envs[0][0][p][m]
                pert1_down = stress_envs[0][1][p][m]

                for q in range(3):
                    pert2_up = force_envs[1][0][q][n]
                    pert2_down = force_envs[1][1][q][n]

                    kern1 = kernel.energy_energy(pert1_up, pert2_up)
                    kern2 = kernel.energy_energy(pert1_up, pert2_down)
                    kern3 = kernel.energy_energy(pert1_down, pert2_up)
                    kern4 = kernel.energy_energy(pert1_down, pert2_down)

                    stress_force_finite[p, q] += (kern1 - kern2 - kern3 + kern4) / (
                        4 * delta * delta
                    )

    assert (
        np.abs(stress_force_exact - stress_force_finite) < threshold
    ).all(), "Your stress/force kernel is wrong."


@pytest.mark.parametrize("kernel", kernel_list)
def test_stress_stress(struc_envs, stress_envs, force_envs, kernel):

    stress_stress_exact = np.zeros((6, 6))
    stress_stress_finite = np.zeros((6, 6))

    for m in range(len(struc_envs[0])):
        for n in range(len(struc_envs[1])):
            stress_stress_exact += kernel.stress_stress(
                struc_envs[0][m], struc_envs[1][n]
            )
            for p in range(6):
                pert1_up = stress_envs[0][0][p][m]
                pert1_down = stress_envs[0][1][p][m]

                for q in range(6):
                    pert2_up = stress_envs[1][0][q][n]
                    pert2_down = stress_envs[1][1][q][n]

                    kern1 = kernel.energy_energy(pert1_up, pert2_up)
                    kern2 = kernel.energy_energy(pert1_up, pert2_down)
                    kern3 = kernel.energy_energy(pert1_down, pert2_up)
                    kern4 = kernel.energy_energy(pert1_down, pert2_down)

                    stress_stress_finite[p, q] += (kern1 - kern2 - kern3 + kern4) / (
                        4 * delta * delta
                    )

    assert (
        np.abs(stress_stress_exact - stress_stress_finite) < threshold
    ).all(), "Your stress/stress kernel is wrong."


@pytest.mark.parametrize("kernel", kernel_list)
def test_force_grad(struc_envs, stress_envs, force_envs, kernel):

    test_env_1 = struc_envs[0][0]
    test_env_2 = struc_envs[1][0]

    # Check force/force gradient.
    force_force_gradient = kernel.force_force_gradient(test_env_1, test_env_2)
    # print(force_force_gradient)

    # Perturb signal variance.
    kernel.signal_variance = signal_variance + delta
    force_sig_up = kernel.force_force(test_env_1, test_env_2)

    kernel.signal_variance = signal_variance - delta
    force_sig_down = kernel.force_force(test_env_1, test_env_2)

    # Perturb length scale.
    kernel.signal_variance = signal_variance
    kernel.length_scale = length_scale + delta
    force_ls_up = kernel.force_force(test_env_1, test_env_2)

    kernel.length_scale = length_scale - delta
    force_ls_down = kernel.force_force(test_env_1, test_env_2)

    for m in range(3):
        for n in range(3):
            sig_val = (force_sig_up[m, n] - force_sig_down[m, n]) / (2 * delta)

            ls_val = (force_ls_up[m, n] - force_ls_down[m, n]) / (2 * delta)

            assert (
                np.abs(sig_val - force_force_gradient[1][0, m, n]) < threshold
            ), "The force/force gradient is wrong."

            assert (
                np.abs(ls_val - force_force_gradient[1][1, m, n]) < threshold
            ), "The force/force gradient is wrong."


@pytest.mark.parametrize("kernel", kernel_list)
def test_efs(struc_envs, kernel):
    test_env_1 = struc_envs[0][0]
    test_env_2 = struc_envs[1][0]

    e1, f1, s1 = kernel.efs_energy(test_env_2, test_env_1)

    e2 = kernel.energy_energy(test_env_2, test_env_1)
    f2 = kernel.force_energy(test_env_2, test_env_1)
    s2 = kernel.stress_energy(test_env_2, test_env_1)

    assert np.abs(e1 - e2) < threshold
    assert (np.abs(f1 - f2) < threshold).all()
    assert (np.abs(s1 - s2) < threshold).all()

    e3, f3, s3 = kernel.efs_force(test_env_1, test_env_2)

    e4 = kernel.force_energy(test_env_2, test_env_1)
    f4 = kernel.force_force(test_env_1, test_env_2)
    s4 = kernel.stress_force(test_env_1, test_env_2)

    assert (np.abs(e3 - e4) < threshold).all()
    assert (np.abs(f3 - f4) < threshold).all()
    assert (np.abs(s3 - s4) < threshold).all()


@pytest.mark.parametrize("kernel", kernel_list)
def test_efs_self(struc_envs, kernel):
    test_env_1 = struc_envs[0][0]

    e1, f1, s1 = kernel.efs_self(test_env_1)

    e2 = kernel.energy_energy(test_env_1, test_env_1)
    f2 = kernel.force_force(test_env_1, test_env_1)
    s2 = kernel.stress_stress(test_env_1, test_env_1)

    print(e1)
    print(e2)

    print(f1)
    print(f2)

    print(s1)
    print(s2)

    assert np.abs(e1 - e2) < threshold
    assert (np.abs(f1 - np.diag(f2)) < threshold).all()
    assert (np.abs(s1 - np.diag(s2)) < threshold).all()
