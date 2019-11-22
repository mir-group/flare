import numpy as np
import math
import multiprocessing as mp
import time

from typing import List

#######################################
##### COVARIANCE MATRIX FUNCTIONS
#######################################

def get_cov_row(x_1, d_1, m_index: int, size: int, training_data: list,
                kernel: callable, kern_hyps: np.array,
                cutoffs: np.array)-> list:
    """ Compute an individual row of a covariance matrix.

    :param x_1: Atomic environment to compare to At. Envs. in training data
    :param d_1: Index of force component (x,y,z), indexed as (1,2,3)
    :param m_index:
    :param size:
    :param training_data: Set of atomic environments to compare against
    :param kernel: Kernel function to compare x_1 against training data with
    :param kern_hyps: Hyperparameters which parameterize kernel function
    :param cutoffs: The cutoff values used for the atomic environments
    :return: covs, list of covariance matrix row elements
    """

    covs = []
    ds = [1, 2, 3]
    for n_index in range(m_index, size):
        x_2 = training_data[int(math.floor(n_index / 3))]
        d_2 = ds[n_index % 3]

        # calculate kernel and gradient
        kern_curr = kernel(x_1, x_2, d_1, d_2, kern_hyps,
                           cutoffs)
        covs.append(kern_curr)

    return covs


def get_cov_row_derv(x_1, d_1: int, m_index: int, size: int,
                     training_data: list,
                     kernel_grad: callable,
                     kern_hyps: np.array, cutoffs: np.array)-> (list, list):
    """ Compute a single row of covariance matrix and hyperparameter gradient matrix

    :param x_1: Atomic Environment to compare to At. Envs. in training data
    :param d_1: Index of force
    :param m_index:
    :param size:
    :param training_data: Set of atomic environments to compare against
    :param kernel: Kernel callable to compare x_1 against training data with
    :param kernel_grad: Callable for gradient of kernel
    :param kern_hyps: Hyperparameters which parameterize kernel function
    :param cutoffs: The cutoff values used for the atomic environments
    :return: covs, hyps
    """
    covs = []
    hyps = []
    ds = [1, 2, 3]
    for n_index in range(m_index, size):
        x_2 = training_data[int(math.floor(n_index / 3))]
        d_2 = ds[n_index % 3]

        # calculate kernel and gradient
        kern_curr = kernel_grad(x_1, x_2, d_1, d_2, kern_hyps,
                                cutoffs)
        covs.append(kern_curr[0])
        hyps.append(kern_curr[1])

    return covs, hyps

#######################################
##### KY MATRIX FUNCTIONS
#######################################


def get_ky_mat(hyps: np.array, training_data: list,
               kernel: callable, cutoffs=None):
    """ Compute covariance matrix K by comparing training data with itself

    :param hyps:
    :param training_data:
    :param training_labels_np:
    :param kernel:
    :param cutoffs:
    :return:
    """

    # assume sigma_n is the final hyperparameter
    sigma_n = hyps[-1]

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
            kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
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
    """ Compute covariance matrix K and matrix to estimate likelihood gradient

    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param training_labels_np: numpy array of forces
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: list of 2 float numbers for 2b and 3b cutoff
    :return: hyp_mat, ky_mat
    """
    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]

    # initialize matrices
    size = len(training_data) * 3
    k_mat = np.zeros([size, size])
    hyp_mat = np.zeros([number_of_hyps, size, size])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        for n_index in range(m_index, size):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            cov = kernel_grad(x_1, x_2, d_1, d_2, hyps, cutoffs)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            k_mat[n_index, m_index] = cov[0]

            # store gradients (excluding noise variance)
            hyp_mat[:-1, m_index, n_index] = cov[1]
            hyp_mat[:-1, n_index, m_index] = cov[1]

    # add gradient of noise variance
    hyp_mat[-1, :, :] = np.eye(size) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    return hyp_mat, ky_mat

def get_ky_mat_par(hyps: np.ndarray, training_data: list,
                   kernel, cutoffs=None, no_cpus=None):
    """
    Parallelized version of function which computes ky matrix
    If the cpu set up is None, it uses as much as posible cpus

    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param training_labels_np: numpy array of forces
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: list of 2 float numbers for 2b and 3b cutoff
    :param no_cpus: number of cpus to use.
    :return: hyp_mat, ky_mat
    """

    if (no_cpus is None):
        ncpus = mp.cpu_count()
    else:
        ncpus = no_cpus

    if (ncpus == 1):
        return get_ky_mat(hyps, training_data,
                          kernel, cutoffs)

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]

    # initialize matrices
    size = len(training_data) * 3
    k_mat = np.zeros([size, size])

    ds = [1, 2, 3]

    # calculate elements
    results = []
    with mp.Pool(processes=ncpus) as pool:
        for m_index in range(size):
            x_1 = training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            results.append(pool.apply_async(get_cov_row,
                                            args=(x_1, d_1, m_index, size,
                                                  training_data, kernel,
                                                  hyps, cutoffs)))
        pool.close()
        pool.join()

    # construct covariance matrix
    for m in range(size):
        res_cur = results[m].get()
        for n in range(m, size):
            k_mat[m, n] = res_cur[n-m]
            k_mat[n, m] = res_cur[n-m]

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    return ky_mat


def get_ky_and_hyp_par(hyps: np.ndarray, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel_grad, cutoffs=None, no_cpus=None):
    """
    Parallelized version of function which computes ky matrix
    and its derivative to hyper-parameter
    If the cpu set up is None, it uses as much as posible cpus

    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param training_labels_np: numpy array of forces
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: list of 2 float numbers for 2b and 3b cutoff
    :param no_cpus: number of cpus to use.
    :return: hyp_mat, ky_mat
    """

    if (no_cpus is None):
        ncpus = mp.cpu_count()
    else:
        ncpus = no_cpus

    if (ncpus == 1):
        return get_ky_and_hyp(hyps, training_data,
                              training_labels_np,
                              kernel_grad, cutoffs)

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]

    # initialize matrices
    size = len(training_data) * 3
    k_mat = np.zeros([size, size])
    hyp_mat = np.zeros([number_of_hyps, size, size])

    ds = [1, 2, 3]

    # calculate elements
    results = []
    with mp.Pool(processes=ncpus) as pool:
        for m_index in range(size):
            x_1 = training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            results.append(pool.apply_async(get_cov_row_derv,
                                            args=(x_1, d_1, m_index, size,
                                                  training_data, kernel_grad,
                                                  hyps, cutoffs)))
        pool.close()
        pool.join()

    # construct covariance matrix
    for m in range(size):
        res_cur = results[m].get()
        for n in range(m, size):
            k_mat[m, n] = res_cur[0][n-m]
            k_mat[n, m] = res_cur[0][n-m]
            hyp_mat[:-1, m, n] = res_cur[1][n-m]
            hyp_mat[:-1, n, m] = res_cur[1][n-m]

    # add gradient of noise variance
    hyp_mat[-1, :, :] = np.eye(size) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    return hyp_mat, ky_mat



def get_ky_mat_update_row(params):
    '''
    used for update_L_alpha, especially for parallelization

    :param: tuple of index, atomic environment,
            funciton object of get_kernel_vector
    '''
    ind, x_t, get_kernel_vector = params
    k_vi = np.array([get_kernel_vector(x_t, d + 1)
                     for d in range(3)]).T  # (n+3m) x 3
    return k_vi

def get_ky_mat_update(ky_mat_old, training_data, get_kernel_vector, hyps, no_cpus=None):
    '''
    used for update_L_alpha, especially for parallelization
    parallelized for added atoms, for example, if add 10 atoms to the training
    set, the K matrix will add 10x3 columns and 10x3 rows, and the task will
    be distributed to 30 processors

    :param ky_mat_old:
    :param training_data:
    :param get_kernel_vector:
    :param hyps:
    :param par:
    '''

    if (no_cpus is None):
        ncpus = mp.cpu_count()
    else:
        ncpus = no_cpus

    n = ky_mat_old.shape[0]
    N = len(training_data)
    m = N - n // 3  # number of new data added
    ky_mat = np.zeros((3 * N, 3 * N))
    ky_mat[:n, :n] = ky_mat_old

    # calculate kernels for all added data
    params_list = [(n//3+i, training_data[n//3+i], get_kernel_vector)\
                   for i in range(m)]
    if ncpus>1:
        pool = mp.Pool(processes=ncpus)
        k_vi_list = pool.map(get_ky_mat_update_row, params_list)
        pool.close()
        pool.join()

    for i in range(m):
        params = params_list[i]
        ind = params[0]
        if par:
            k_vi = k_vi_list[i]
        else:
            k_vi = get_ky_mat_update_row(params)
        ky_mat[:, 3 * ind:3 * ind + 3] = k_vi
        ky_mat[3 * ind:3 * ind + 3, :n] = k_vi[:n, :].T

    sigma_n = hyps[-1]
    ky_mat[n:, n:] += sigma_n ** 2 * np.eye(3 * m)
    return ky_mat

#######################################
##### LIKELIHOOD + LIKELIHOOD GRADIENT
#######################################

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

        number_of_hyps = hyp_mat.shape[0]

        # catch linear algebra errors
        try:
            ky_mat_inv = np.linalg.inv(ky_mat)
            l_mat = np.linalg.cholesky(ky_mat)
        except:
            return -1e8, np.zeros(number_of_hyps)

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
                           np.trace(np.matmul(like_mat, hyp_mat[n, :, :]))

        return like, like_grad


def get_neg_likelihood(hyps: np.ndarray, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel, cutoffs=None, output = None,
                       par=False, no_cpus=None):

    if par:
        ky_mat = \
            get_ky_mat_par(hyps, training_data, training_labels_np,
                           kernel, cutoffs, no_cpus)
    else:
        ky_mat = \
            get_ky_mat(hyps, training_data, training_labels_np,
                       kernel, cutoffs)

    like = get_like_from_ky_mat(ky_mat, training_labels_np)

    if output is not None:
        output.write_hyps(None, hyps, None, like, "NA", name="hyps")

    return -like


def get_neg_like_grad(hyps: np.ndarray, training_data: list,
                      training_labels_np: np.ndarray,
                      kernel_grad, cutoffs=None,
                      output = None, par=False, no_cpus=None):

    if par:
        hyp_mat, ky_mat = \
            get_ky_and_hyp_par(hyps, training_data, training_labels_np,
                               kernel_grad, cutoffs, no_cpus)
    else:
        hyp_mat, ky_mat = \
            get_ky_and_hyp(hyps, training_data, training_labels_np,
                           kernel_grad, cutoffs)

    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, training_labels_np)

    if output is not None:
        output.write_hyps(None, hyps, None, like, like_grad, name="hyps")

    return -like, -like_grad
