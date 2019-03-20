import numpy as np
import math


def get_ky_mat(hyps: np.ndarray, training_data: list,
               training_labels_np: np.ndarray,
               kernel, cutoffs=None):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]
    kern_hyps = hyps[0:(number_of_hyps - 1)]

    # initialize matrices
    size = len(training_data) * 3
    k_mat = np.zeros([size, size])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        for n_index in range(m_index, size):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            kern_curr = kernel(x_1, x_2, d_1, d_2, kern_hyps,
                               cutoffs)

            # store kernel value
            k_mat[m_index, n_index] = kern_curr
            k_mat[n_index, m_index] = kern_curr

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    return ky_mat


def get_ky_and_hyp(hyps: np.ndarray, training_data: list,
                   training_labels_np: np.ndarray,
                   kernel_grad, cutoffs=None):
    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]
    kern_hyps = hyps[0:(number_of_hyps - 1)]

    # initialize matrices
    size = len(training_data) * 3
    k_mat = np.zeros([size, size])
    hyp_mat = np.zeros([size, size, number_of_hyps])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        for n_index in range(m_index, size):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            cov = kernel_grad(x_1, x_2, d_1, d_2, kern_hyps, cutoffs)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            k_mat[n_index, m_index] = cov[0]

            # store gradients (excluding noise variance)
            for p_index in range(number_of_hyps - 1):
                hyp_mat[m_index, n_index, p_index] = cov[1][p_index]
                hyp_mat[n_index, m_index, p_index] = cov[1][p_index]

    # add gradient of noise variance
    hyp_mat[:, :, number_of_hyps - 1] = np.eye(size) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    return hyp_mat, ky_mat


def get_like_from_ky_mat(ky_mat, training_labels_np):
        # catch linear algebra errors
        try:
            ky_mat_inv = np.linalg.inv(ky_mat)
            l_mat = np.linalg.cholesky(ky_mat)
        except:
            return -1e8

        alpha = np.matmul(ky_mat_inv, training_labels_np)

        # calculate likelihood
        like = (-0.5 * np.matmul(training_labels_np, alpha) -
                np.sum(np.log(np.diagonal(l_mat))) -
                math.log(2 * np.pi) * ky_mat.shape[1] / 2)

        return like


def get_like_grad_from_mats(ky_mat, hyp_mat, training_labels_np):

        number_of_hyps = hyp_mat.shape[2]

        # catch linear algebra errors
        try:
            ky_mat_inv = np.linalg.inv(ky_mat)
            l_mat = np.linalg.cholesky(ky_mat)
        except:
            return 1e8, np.zeros(number_of_hyps)

        alpha = np.matmul(ky_mat_inv, training_labels_np)
        alpha_mat = np.matmul(alpha.reshape(alpha.shape[0], 1),
                              alpha.reshape(1, alpha.shape[0]))
        like_mat = alpha_mat - ky_mat_inv

        # calculate likelihood
        like = (-0.5 * np.matmul(training_labels_np, alpha) -
                np.sum(np.log(np.diagonal(l_mat))) -
                math.log(2 * np.pi) * ky_mat.shape[1] / 2)

        # calculate likelihood gradient
        like_grad = np.zeros(number_of_hyps)
        for n in range(number_of_hyps):
            like_grad[n] = 0.5 * \
                           np.trace(np.matmul(like_mat, hyp_mat[:, :, n]))

        return like, like_grad


def get_neg_likelihood(hyps: np.ndarray, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel, cutoffs=None, monitor: bool = False):

    if monitor:
        print('hyps: ' + str(hyps))

    ky_mat = get_ky_mat(hyps, training_data, training_labels_np,
                        kernel, cutoffs)

    like = get_like_from_ky_mat(ky_mat, training_labels_np)

    if monitor:
        print('like: ' + str(like))
        print('\n')

    return -like


def get_neg_like_grad(hyps: np.ndarray, training_data: list,
                      training_labels_np: np.ndarray,
                      kernel_grad, cutoffs=None,
                      monitor: bool = False):

    if monitor:
        print('hyps: ' + str(hyps))

    hyp_mat, ky_mat = \
        get_ky_and_hyp(hyps, training_data, training_labels_np,
                       kernel_grad, cutoffs)

    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, training_labels_np)

    if monitor:
        print('like grad: ' + str(like_grad))
        print('like: ' + str(like))
        print('\n')

    return -like, -like_grad
