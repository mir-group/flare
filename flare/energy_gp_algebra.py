"""Matrix algebra for GP models trained on energy and force data."""
import numpy as np


def kernel_ee(envs1, envs2, energy_kernel, hyps, cutoffs):
    """Compute energy/energy kernel between two structures."""

    kern = 0
    for env1 in envs1:
        for env2 in envs2:
            kern += \
                energy_kernel(env1, env2, hyps, cutoffs)

    return kern


def kernel_fe(env1, envs2, component, force_energy_kernel, hyps, cutoffs):
    """Compute force/energy kernel between a structure and an environment."""

    kern = 0
    for env2 in envs2:
        kern += \
            force_energy_kernel(env1, env2, component, hyps, cutoffs)

    return kern


def get_ky_block(hyps, struc1, envs1, atoms1, struc2, envs2, atoms2,
                 kernel, force_energy_kernel, energy_kernel, cutoffs):
    """Get the covariance matrix between two structures."""

    block = np.zeros((len(struc1.labels), len(struc2.labels)))
    index1 = 0
    index2 = 0

    if struc1.energy is not None:
        # energy/energy
        if struc2.energy is not None:
            block[index1, index2] = \
                kernel_ee(envs1, envs2, energy_kernel, hyps,
                          cutoffs)
            index2 += 1

        # energy/force
        if struc2.forces is not None:
            for atom in atoms2:
                force_env = envs2[atom]
                for d in range(3):
                    block[index1, index2] = \
                        kernel_fe(force_env, envs1, d+1, force_energy_kernel,
                                  hyps, cutoffs)
                    index2 += 1
        index1 += 1

    if struc1.forces is not None:
        for atom in atoms1:
            env1 = envs1[atom]
            for d_1 in range(3):
                index2 = 0

                # force/energy
                if struc2.energy is not None:
                    block[index1, index2] = \
                        kernel_fe(env1, envs2, d_1+1, force_energy_kernel,
                                  hyps, cutoffs)
                    index2 += 1

                # force/force
                if struc2.forces is not None:
                    for env2 in envs2:
                        for d_2 in range(3):
                            block[index1, index2] = \
                                kernel(env1, env2, d_1+1, d_2+1, hyps, cutoffs)
                            index2 += 1

                index1 += 1

    return block


def noise_matrix(hyps, strucs, atoms, k_size):
    """Compute diagonal noise matrix."""

    noise_vec = np.zeros(k_size)
    index = 0

    # assume energy noise is final hyperparameter
    for struc, atom_list in zip(strucs, atoms):
        if struc.energy is not None:
            noise_vec[index] = hyps[-1]**2
            index += 1
        if struc.forces is not None:
            no_comps = 3 * len(atom_list)
            noise_vec[index:index+no_comps] = hyps[-2]**2
            index += no_comps

    return np.diag(noise_vec)


def get_ky_mat(hyps: np.ndarray, training_strucs, training_envs,
               training_atoms, training_labels_np: np.ndarray,
               kernel, force_energy_kernel, energy_kernel, cutoffs=None):

    # initialize matrices
    size = len(training_labels_np)
    k_mat = np.zeros((size, size))

    # covariance matrix has a block structure, where each comparison of
    # two structures gets its own block
    index1 = 0
    for m in range(len(training_strucs)):
        struc1 = training_strucs[m]
        size1 = len(struc1.labels)
        envs1 = training_envs[m]
        atoms1 = training_atoms[m]
        index2 = index1

        for n in range(m, len(training_strucs)):
            struc2 = training_strucs[n]
            size2 = len(struc2.labels)
            envs2 = training_envs[n]
            atoms2 = training_atoms[n]

            ky_block = \
                get_ky_block(hyps, struc1, envs1, atoms1, struc2, envs2,
                             atoms2, kernel, force_energy_kernel,
                             energy_kernel, cutoffs)

            k_mat[index1:index1 + size1, index2:index2 + size2] = ky_block
            k_mat[index2:index2 + size2, index1:index1 + size1] = \
                ky_block.transpose()

            index2 += size2
        index1 += size1

    # add noise
    noise_mat = noise_matrix(hyps, training_strucs, training_atoms, size)
    k_mat = k_mat + noise_mat

    return k_mat
