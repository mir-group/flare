import numpy as np


def kernel_ee(envs1, envs2, energy_kernel, hyps, cutoffs):
    """Compute energy/energy kernel between two structures."""

    kern = 0
    for env1 in envs1:
        for env2 in envs2:
            kern += \
                energy_kernel(env1, env2, hyps, cutoffs)

    return kern


def kernel_ef(envs1, env2, component, energy_force_kernel, hyps,
              cutoffs):
    """Compute energy/force kernel between a structure and an environment."""

    kern = 0
    for env1 in envs1:
        kern += \
            energy_force_kernel(env2, env1, component, hyps, cutoffs)

    return kern


def get_ky_block(hyps, struc1, envs1, atoms1, struc2, envs2, atoms2,
                 kernel, energy_force_kernel, energy_kernel, cutoffs):
    """Get the covariance block of a structure comparison."""

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
                        kernel_ef(envs1, force_env, d+1, energy_force_kernel,
                                  hyps, cutoffs)
                    index2 += 1

    index1 += 1
    index2 = 0
    if struc1.forces is not None:
        for atom in atoms1:
            env1 = envs1[atom]
            for d_1 in range(3):
                # force/energy
                if struc2.energy:
                    block[index1, index2] = \
                        kernel_ef(envs2, env1, d_1+1, energy_force_kernel,
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
               kernel, energy_force_kernel, energy_kernel, cutoffs=None):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_en = hyps[-1]
    sigma_frc = hyps[-2]

    # initialize matrices
    size = len(training_labels_np)
    k_mat = np.zeros([size, size])
    no_strucs = len(training_strucs)

    # covariance matrix has a block structure, where each comparison of
    # two structures gets its own block

    pass
