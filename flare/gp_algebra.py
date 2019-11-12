import numpy as np
import math
import multiprocessing as mp
import time


def get_ky_mat_par(hyps: np.ndarray, training_data: list,
                   training_labels_np: np.ndarray,
                   kernel, cutoffs=None, no_cpus=None, nsample=100):

    if (no_cpus is None):
        ncpus = mp.cpu_count()
    else:
        ncpus = no_cpus


    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]

    # initialize matrices
    size = len(training_data)
    size3 = 3*len(training_data)
    ns = int(math.ceil(size/nsample))
    k_mat_slice = []
    k_mat = np.zeros([size3, size3])
    with mp.Pool(processes=ncpus) as pool:
        for ibatch1 in range(ns):
            s1 = nsample*ibatch1
            e1 = np.min([s1 + nsample, size])
            t1 = training_data[s1:e1]
            for ibatch2 in range(ibatch1, ns):
                s2 = nsample*ibatch2
                e2 = np.min([s2 + nsample, size])
                t2 = training_data[s2:e2]
                k_mat_slice += [pool.apply_async(
                                          get_ky_mat_pack,
                                          args=(hyps,
                                            t1, t2,
                                            bool(ibatch1==ibatch2),
                                            kernel, cutoffs))]
        slice_count=0
        for ibatch1 in range(ns):
            s1 = nsample*ibatch1
            e1 = np.min([s1 + nsample, size])
            for ibatch2 in range(ibatch1, ns):
                s2 = nsample*ibatch2
                e2 = np.min([s2 + nsample, size])
                k_mat_block = k_mat_slice[slice_count].get()
                slice_count += 1
                k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                if (ibatch1 != ibatch2):
                    k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
        pool.close()
        pool.join()


    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return ky_mat

def get_ky_mat_pack(hyps: np.ndarray, training_data1: list,
               training_data2:list, same: bool,
               kernel, cutoffs):

    # initialize matrices
    size1 = len(training_data1) * 3
    size2 = len(training_data2) * 3
    k_mat = np.zeros([size1, size2])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size1):
        x_1 = training_data1[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]
        if (same):
            lowbound = m_index
        else:
            lowbound = 0
        for n_index in range(lowbound, size2):
            x_2 = training_data2[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            kern_curr = kernel(x_1, x_2, d_1, d_2, hyps, cutoffs)

            # store kernel value
            k_mat[m_index, n_index] = kern_curr
            if (same):
                k_mat[n_index, m_index] = kern_curr

    return k_mat


def get_ky_and_hyp_par(hyps: np.ndarray, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel_grad, cutoffs=None, no_cpus=None, nsample=100):

    if (no_cpus is None):
        ncpus = mp.cpu_count()
    else:
        ncpus = no_cpus

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[-1]

    size = len(training_data)
    size3 = size*3

    # initialize matrices
    k_mat = np.zeros([size3, size3])
    hyp_mat = np.zeros([number_of_hyps, size3, size3])

    with mp.Pool(processes=ncpus) as pool:
        ns = int(math.ceil(size/nsample))
        mat_slice = []
        for ibatch1 in range(ns):
            s1 = nsample*ibatch1
            e1 = np.min([s1 + nsample, size])
            t1 = training_data[s1:e1]
            for ibatch2 in range(ibatch1, ns):
                s2 = nsample*ibatch2
                e2 = np.min([s2 + nsample, size])
                t2 = training_data[s2:e2]
                if (ibatch1 == ibatch2):
                    same=True
                else:
                    same = False
                mat_slice += [pool.apply_async(
                                        get_ky_and_hyp_pack,
                                        args=(
                                            hyps, t1, t2,
                                            same, kernel_grad, cutoffs))]
        slice_count=0
        for ibatch1 in range(ns):
            s1 = nsample*ibatch1
            e1 = np.min([s1 + nsample, size])
            for ibatch2 in range(ibatch1, ns):
                s2 = nsample*ibatch2
                e2 = np.min([s2 + nsample, size])
                h_mat_block, k_mat_block = mat_slice[slice_count].get()
                slice_count += 1
                k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                hyp_mat[:-1, s1*3:e1*3, s2*3:e2*3] = h_mat_block
                if (ibatch1 != ibatch2):
                    k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
                    hyp_mat[:-1, s2*3:e2*3, s1*3:e1*3] = h_mat_block.T
        pool.close()


    # add gradient of noise variance
    hyp_mat[-1, :, :] = np.eye(size3) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return hyp_mat, ky_mat

def get_ky_and_hyp_pack(hyps: np.ndarray, training_data1: list,
                   training_data2: list, same: bool,
                   kernel_grad, cutoffs=None):

    # assume sigma_n is the final hyperparameter
    non_noise_hyps = len(hyps)-1

    # initialize matrices
    size1 = len(training_data1) * 3
    size2 = len(training_data2) * 3
    k_mat = np.zeros([size1, size2])
    hyp_mat = np.zeros([non_noise_hyps, size1, size2])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size1):
        x_1 = training_data1[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        if (same):
            lowbound = m_index
        else:
            lowbound = 0
        for n_index in range(lowbound, size2):
            x_2 = training_data2[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            cov = kernel_grad(x_1, x_2, d_1, d_2, hyps, cutoffs)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            hyp_mat[:, m_index, n_index] = cov[1]

            if (same):
                k_mat[n_index, m_index] = cov[0]
                hyp_mat[:, n_index, m_index] = cov[1]

    return hyp_mat, k_mat


def get_ky_mat(hyps: np.ndarray, training_data: list,
               training_labels_np: np.ndarray,
               kernel, cutoffs=None):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]

    # initialize matrices
    size3 = len(training_data) * 3
    k_mat = np.zeros([size3, size3])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size3):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        for n_index in range(m_index, size3):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
                               cutoffs)

            # store kernel value
            k_mat[m_index, n_index] = kern_curr
            k_mat[n_index, m_index] = kern_curr

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return ky_mat


def get_ky_and_hyp(hyps: np.ndarray, training_data: list,
                   training_labels_np: np.ndarray,
                   kernel_grad, cutoffs=None):
    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]

    # initialize matrices
    size3 = len(training_data) * 3
    k_mat = np.zeros([size3, size3])
    hyp_mat = np.zeros([number_of_hyps, size3, size3])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size3):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        for n_index in range(m_index, size3):
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
    hyp_mat[-1, :, :] = np.eye(size3) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return hyp_mat, ky_mat

def get_ky_mat_update_row(params):
    '''
    used for update_L_alpha, especially for parallelization
    '''
    ind, x_t, get_kernel_vector = params
    k_vi = np.array([get_kernel_vector(x_t, d + 1)
                     for d in range(3)]).T  # (n+3m) x 3
    return k_vi

def get_ky_mat_update(ky_mat_old, training_data, get_kernel_vector, hyps, par, no_cpus=None):
    '''
    used for update_L_alpha, especially for parallelization
    parallelized for added atoms, for example, if add 10 atoms to the training
    set, the K matrix will add 10x3 columns and 10x3 rows, and the task will
    be distributed to 30 processors
    '''
    n = ky_mat_old.shape[0]
    N = len(training_data)
    m = N - n // 3  # number of new data added
    ky_mat = np.zeros((3 * N, 3 * N))
    ky_mat[:n, :n] = ky_mat_old

    # calculate kernels for all added data
    params_list = [(n//3+i, training_data[n//3+i], get_kernel_vector)\
                   for i in range(m)]
    if par:
        if (no_cpus is None):
            pool = mp.Pool(processes=mp.cpu_count())
        else:
            pool = mp.Pool(processes=no_cpus)
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

    print("like", like, like_grad)
    print("hyps", hyps)

    if output is not None:
        output.write_hyps(None, hyps, None, like, like_grad, name="hyps")

    return -like, -like_grad

def get_kernel_vector_unit(training_data, x,
                      d_1, kernel, hyps, cutoffs,
                      hyps_mask):

    ds = [1, 2, 3]
    size = len(training_data) * 3
    k_v = np.zeros(size, )

    if (hyps_mask is not None):
        for m_index in range(size):
            x_2 = training_data[int(math.floor(m_index / 3))]
            d_2 = ds[m_index % 3]
            k_v[m_index] = kernel(x, x_2, d_1, d_2,
                                  hyps, cutoffs,
                                  hyps_mask=hyps_mask)
    else:
        for m_index in range(size):
            x_2 = training_data[int(math.floor(m_index / 3))]
            d_2 = ds[m_index % 3]
            k_v[m_index] = kernel(x, x_2, d_1, d_2,
                                  hyps, cutoffs)
    return k_v

def get_kernel_vector_par(training_data, x,
                          d_1, hyps, cutoffs,
                          hyps_mask, nsample=100,
                          no_cpus=None):
    """
    Compute kernel vector, comparing input environment to all             environments
    in the GP's training set.

    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param d_1: Cartesian component of force vector to get (1=x,2=y,3=z)
    :type d_1: int
    :return: kernel vector
    :rtype: np.ndarray
    """

    if (no_cpus is None):
        ncpus = mp.cpu_count()
    else:
        ncpus = no_cpus

    with mp.Pool(processes=processes) as pool:
        size = len(training_data)
        ns = int(math.ceil(size/nsample))
        k12_slice = []
        for ibatch in range(ns):

            s = nsample*ibatch
            e = np.min([s + nsample, size])
            k12_slice.append(pool.apply_async(get_kernel_vector_unit,
                                              args=(training_data[s: e],
                                                    x, d_1, hyps,
                                                    cutoffs,
                                                    hyps_mask)))
        size3 = size*3
        nsample3 = nsample*3
        k12_v = np.zeros(size3)
        for ibatch in range(ns):
            s = nsample3*ibatch
            e = np.min([s + nsample3, size3])
            k12_v[s:e] = k12_slice[ibatch].get()
        pool.close()
        pool.join()

    return k_v
