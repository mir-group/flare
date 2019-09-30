import numpy as np


def get_ky_block(hyps, struc1, envs1, atoms1, struc2, envs2, atoms2,
                 kernel, energy_force_kernel, energy_kernel, cutoffs):
    pass


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

    index1 = 0
    for struc1_no in range(no_strucs):
        struc1 = training_strucs[struc1_no]
        index2 = 0
        for struc2_no in range(struc1_no, no_strucs):
            struc2 = training_strucs[struc2_no]

            # energy/energy kernel
            if struc1.energy and struc2.energy:
                en_kern = 0
                for env1 in training_envs[struc1_no]:
                    for env2 in training_envs[struc2_no]:
                        en_kern += \
                            energy_kernel(env1, env2, hyps, cutoffs)
                k_mat[index1, index2] = en_kern
                index2 += 1
            



    # if structure.energy is not None:
    #     en_kern = 0
    #     for train_env in training_envs[struc_no]:
    #         en_kern += \
    #             self.energy_force_kernel(test_env, train_env, d_1,
    #                                         self.hyps, self.cutoffs)
    #     kernel_vector[index] = en_kern
    #     index += 1

    # if structure.forces is not None:
    #     for atom in self.training_atoms[struc_no]:
    #         train_env = self.training_envs[struc_no][atom]
    #         for d_2 in range(3):
    #             kernel_vector[index] = \
    #                 self.kernel(test_env, train_env, d_1, d_2 + 1,
    #                             self.hyps, self.cutoffs)
    #             index += 1

    # # calculate elements
    # for m_index in range(size):
    #     x_1 = training_data[int(math.floor(m_index / 3))]
    #     d_1 = ds[m_index % 3]

    #     for n_index in range(m_index, size):
    #         x_2 = training_data[int(math.floor(n_index / 3))]
    #         d_2 = ds[n_index % 3]

    #         # calculate kernel and gradient
    #         kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
    #                            cutoffs)

    #         # store kernel value
    #         k_mat[m_index, n_index] = kern_curr
    #         k_mat[n_index, m_index] = kern_curr

    # # matrix manipulation
    # ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    # return ky_mat