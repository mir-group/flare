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
    """Compute energy/force kernel between a structure and an environment."""

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

    if struc1.energy:
        # energy/energy
        if struc2.energy:
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
                if struc2.energy:
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


def get_ky_mat(hyps: np.ndarray, training_strucs, training_envs,
               training_atoms, training_labels_np: np.ndarray,
               kernel, force_energy_kernel, energy_kernel, cutoffs=None):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_en = hyps[-1]
    sigma_frc = hyps[-2]

    # initialize matrices
    size = len(training_labels_np)
    k_mat = np.zeros([size, size])

    # covariance matrix has a block structure, where each comparison of
    # two structures gets its own block
    index1 = 0
    for m in range(len(training_strucs)):
        struc1 = training_strucs[m]
        size1 = len(struc1.labels)
        envs1 = training_envs[m]
        atoms1 = training_atoms[m]
        index2 = 0

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

    return k_mat
