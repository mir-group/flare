import numpy as np
import math
import multiprocessing as mp
import time
from flare.gp_algebra import get_like_from_ky_mat, get_like_grad_from_mats


def get_cov_row(x_1, d_1, m_index, size, training_data, kernel,
                kern_hyps, cutoffs, hyps_mask):
    covs = []
    ds = [1, 2, 3]
    for n_index in range(m_index, size):
        x_2 = training_data[int(math.floor(n_index / 3))]
        d_2 = ds[n_index % 3]

        # calculate kernel and gradient
        kern_curr = kernel(x_1, x_2, d_1, d_2, kern_hyps,
                           cutoffs, hyps_mask=hyps_mask)
        covs.append(kern_curr)

    return covs


def get_cov_row_derv(x_1, d_1, m_index, size, training_data, kernel_grad,
                     kern_hyps, cutoffs, hyps_mask):
    covs = []
    hyps = []
    ds = [1, 2, 3]
    for n_index in range(m_index, size):
        x_2 = training_data[int(math.floor(n_index / 3))]
        d_2 = ds[n_index % 3]

        # calculate kernel and gradient
        kern_curr = kernel_grad(x_1, x_2, d_1, d_2, kern_hyps,
                                cutoffs, hyps_mask=hyps_mask)
        covs.append(kern_curr[0])

        hyps.append(kern_curr[1])

    return covs, hyps


def get_ky_mat_par(hyps: np.ndarray, training_data: list,
                   training_labels_np: np.ndarray,
                   kernel, cutoffs=None, hyps_mask=None, no_cpus=None):

    if (no_cpus is None):
        ncpus =mp.cpu_count()
    else:
        ncpus =no_cpus

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[-1]
    if ('train_noise' in hyps_mask.keys()):
        if (hyps_mask['train_noise'] is False):
            sigma_n = hyps_mask['original'][-1]

    # initialize matrices
    size = len(training_data) * 3
    k_mat = np.zeros([size, size])

    ds = [1, 2, 3]

    # calculate elements
    results = []
    with mp.Pool(processes=cpu) as pool:
        for m_index in range(size):
            x_1 = training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            results.append(pool.apply_async(get_cov_row,
                                            args=(x_1, d_1, m_index, size,
                                                  training_data, kernel,
                                                  hyps, cutoffs, hyps_mask)))
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


def get_ky_and_hyp_par(hyps: np.ndarray, hyps_mask, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel_grad, cutoffs=None, no_cpus=None):


    if (no_cpus is None):
        cpu = mp.cpu_count()
    else:
        cpu = no_cpus

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[-1]
    train_noise = True
    if ('train_noise' in hyps_mask.keys()):
        if (hyps_mask['train_noise'] is False):
            sigma_n = hyps_mask['original'][-1]
            train_noise = False

    # initialize matrices
    size = len(training_data) * 3
    k_mat = np.zeros([size, size])
    hyp_mat = np.zeros([number_of_hyps, size, size])

    ds = [1, 2, 3]

    # calculate elements
    results = []
    with mp.Pool(processes=cpu) as pool:
        for m_index in range(size):
            x_1 = training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            results.append(pool.apply_async(get_cov_row_derv,
                                            args=(x_1, d_1, m_index, size,
                                                  training_data, kernel_grad,
                                                  hyps, cutoffs,
                                                  hyps_mask)))
        pool.close()
        pool.join()

    # construct covariance matrix
    if (train_noise):
        # add gradient of noise variance
        hyp_mat[-1, :, :] = np.eye(size) * 2 * sigma_n
        for m in range(size):
            res_cur = results[m].get()
            for n in range(m, size):
                k_mat[m, n] = res_cur[0][n-m]
                k_mat[n, m] = res_cur[0][n-m]
                hyp_mat[:-1, m, n] = res_cur[1][n-m]
                hyp_mat[:-1, n, m] = res_cur[1][n-m]
    else:
        for m in range(size):
            res_cur = results[m].get()
            for n in range(m, size):
                k_mat[m, n] = res_cur[0][n-m]
                k_mat[n, m] = res_cur[0][n-m]
                hyp_mat[:, m, n] = res_cur[1][n-m]
                hyp_mat[:, n, m] = res_cur[1][n-m]

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    return hyp_mat, ky_mat


def get_ky_mat(hyps: np.ndarray, training_data: list,
               training_labels_np: np.ndarray,
               kernel, cutoffs=None, hyps_mask=None):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]
    if ('train_noise' in hyps_mask.keys()):
        if (hyps_mask['train_noise'] is False):
            sigma_n = hyps_mask['original'][-1]

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
                               cutoffs, hyps_mask=hyps_mask)

            # store kernel value
            k_mat[m_index, n_index] = kern_curr
            k_mat[n_index, m_index] = kern_curr

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    return ky_mat


def get_ky_and_hyp(hyps: np.ndarray, hyps_mask, training_data: list,
                   training_labels_np: np.ndarray,
                   kernel_grad, cutoffs=None):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]
    train_noise = True
    if ('train_noise' in hyps_mask.keys()):
        if (hyps_mask['train_noise'] is False):
            sigma_n = hyps_mask['original'][-1]
            number_of_hyps -= 1
            train_noise = False

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
            cov = kernel_grad(x_1, x_2, d_1, d_2, hyps, cutoffs,
                    hyps_mask=hyps_mask)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            k_mat[n_index, m_index] = cov[0]

            if (train_noise):
                # store gradients (excluding noise variance)
                hyp_mat[:-1, m_index, n_index] = cov[1]
                hyp_mat[:-1, n_index, m_index] = cov[1]
            else:
                hyp_mat[:, m_index, n_index] = cov[1]
                hyp_mat[:, n_index, m_index] = cov[1]

    # add gradient of noise variance
    if (train_noise):
        hyp_mat[-1, :, :] = np.eye(size) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

    return hyp_mat, ky_mat


def get_neg_likelihood(hyps: np.ndarray, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel, cutoffs=None, output = None,
                       par=False, hyps_mask=None):

    if output is not None:
        ostring="hyps:"
        for hyp in hyps:
            ostring+=f" {hyp}"
        ostring+="\n"
        output.write_to_log(ostring, name="hyps")

    time0 = time.time()
    if par:
        ky_mat = \
            get_ky_mat_par(hyps, training_data, training_labels_np,
                           kernel, cutoffs, hyps_mask=hyps_mask)
    else:
        ky_mat = \
            get_ky_mat(hyps, training_data, training_labels_np,
                       kernel, cutoffs, hyps_mask=hyps_mask)

    output.write_to_log(f"get_key_mat {time.time()-time0}\n", name="hyps")

    time0 = time.time()

    like = get_like_from_ky_mat(ky_mat, training_labels_np)

    output.write_to_log(f"get_like_from_ky_mat {time.time()-time0}\n", name="hyps")

    if output is not None:
        output.write_to_log('like: ' + str(like)+'\n', name="hyps")

    return -like


def get_neg_like_grad(hyps: np.ndarray, training_data: list,
                      training_labels_np: np.ndarray,
                      kernel_grad, cutoffs=None,
                      output = None,
                      par=False, no_cpus=None, hyps_mask=None):

    time0 = time.time()
    if output is not None:
        ostring="hyps:"
        for hyp in hyps:
            ostring+=f" {hyp}"
        ostring+="\n"
        output.write_to_log(ostring, name="hyps")

    if par:
        hyp_mat, ky_mat = \
            get_ky_and_hyp_par(hyps, hyps_mask,
                               training_data, training_labels_np,
                               kernel_grad, cutoffs, no_cpus)
    else:
        hyp_mat, ky_mat = \
            get_ky_and_hyp(hyps, hyps_mask, training_data, training_labels_np,
                           kernel_grad, cutoffs)

    if output is not None:
        output.write_to_log(f"get_ky_and_hyp {time.time()-time0}\n", name="hyps")

    time0 = time.time()

    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, training_labels_np)

    if output is not None:
        output.write_to_log(f"get_like_grad_from_mats {time.time()-time0}\n", name="hyps")

    if output is not None:
        ostring="like grad:"
        for lg in like_grad:
            ostring+=f" {lg}"
        ostring+="\n"
        output.write_to_log(ostring, name="hyps")
        output.write_to_log('like: ' + str(like)+'\n', name="hyps")

    return -like, -like_grad
