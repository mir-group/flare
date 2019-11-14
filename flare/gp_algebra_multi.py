import numpy as np
import math
import multiprocessing as mp
import time
from flare.gp_algebra import get_like_from_ky_mat, get_like_grad_from_mats

def get_ky_mat(hyps: np.ndarray, training_data: list,
               training_labels_np: np.ndarray,
               kernel, cutoffs=None, hyps_mask=None):

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                sigma_n = hyps_mask['original'][-1]

    # matrix manipulation
    k_mat = get_ky_mat_pack(hyps, training_data,
                            training_data, True, kernel,
                            cutoffs, hyps_mask)
    size3 = len(training_data)*3
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return ky_mat


def get_ky_and_hyp(hyps: np.ndarray, hyps_mask, training_data: list,
                   training_labels_np: np.ndarray,
                   kernel_grad, cutoffs=None):


    hyp_mat0, k_mat = get_ky_and_hyp_pack(hyps, hyps_mask, training_data,
                                          training_data, True,
                                          kernel_grad, cutoffs)

    # obtain noise parameter
    train_noise = True
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                train_noise = False
    # add gradient of noise variance
    size3 = 3*len(training_data)
    if (train_noise):
        sigma_mat = np.eye(size3) * 2 * sigma_n
        hyp_mat = np.zeros([hyp_mat0.shape[0]+1,
                            hyp_mat0.shape[1],
                            hyp_mat0.shape[2]])
        hyp_mat[-1, :, :] = sigma_mat
        hyp_mat[:-1, :, :] = hyp_mat0
    else:
        hyp_mat = hyp_mat0

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

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

    print("like", like, like_grad)
    print("hyps", hyps)

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


def get_ky_mat_pack(hyps: np.ndarray, training_data1: list,
               training_data2:list, same: bool,
               kernel, cutoffs, hyps_mask):

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
            kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
                               cutoffs, hyps_mask=hyps_mask)

            # store kernel value
            k_mat[m_index, n_index] = kern_curr
            if (same):
                k_mat[n_index, m_index] = kern_curr

    return k_mat


def get_ky_and_hyp_pack(hyps: np.ndarray, hyps_mask, training_data1: list,
                   training_data2: list, same: bool,
                   kernel_grad, cutoffs=None):

    # assume sigma_n is the final hyperparameter
    non_noise_hyps = len(hyps)-1
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                non_noise_hyps = len(hyps)

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
            cov = kernel_grad(x_1, x_2, d_1, d_2, hyps, cutoffs,
                    hyps_mask=hyps_mask)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            hyp_mat[:, m_index, n_index] = cov[1]

            if (same):
                k_mat[n_index, m_index] = cov[0]
                hyp_mat[:, n_index, m_index] = cov[1]

    return hyp_mat, k_mat


def get_ky_mat_par(hyps: np.ndarray, training_data: list,
                   training_labels_np: np.ndarray,
                   kernel, cutoffs=None, hyps_mask=None,
                   no_cpus=None, nsample=100):

    if (no_cpus is None):
        ncpus =mp.cpu_count()
    else:
        ncpus =no_cpus

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                sigma_n = hyps_mask['original'][-1]

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
                if (ibatch1 == ibatch2):
                    same = True
                else:
                    same = False
                k_mat_slice.append(pool.apply_async(
                                          get_ky_mat_pack,
                                          args=(hyps,
                                            t1, t2, same, kernel,
                                            cutoffs, hyps_mask)))
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
    ky_mat = k_mat
    del k_mat_block
    ky_mat += sigma_n ** 2 * np.eye(size3)

    return ky_mat

def hello(s1, e1, s2, e2):
    return [s1, e1, s2, e2]


def get_ky_and_hyp_par(hyps: np.ndarray, hyps_mask, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel_grad, cutoffs=None,
                       no_cpus=None, nsample=100):


    if (no_cpus is None):
        cpu = mp.cpu_count()
    else:
        cpu = no_cpus

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    non_noise_hyps = len(hyps)-1
    sigma_n = hyps[-1]
    train_noise = True
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                sigma_n = hyps_mask['original'][-1]
                train_noise = False
                non_noise_hyps = len(hyps)

    # initialize matrices
    size = len(training_data)
    size3 = size*3
    k_mat = np.zeros([size3, size3])
    hyp_mat0 = np.zeros([non_noise_hyps, size3, size3])

    with mp.Pool(processes=cpu) as pool:
        mat_slice = []
        ns = int(math.ceil(size/nsample))
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
                mat_slice.append(pool.apply_async(
                                        get_ky_and_hyp_pack,
                                        args=(
                                            hyps, hyps_mask,
                                            t1, t2, same,
                                            kernel_grad, cutoffs)))
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
                hyp_mat0[:, s1*3:e1*3, s2*3:e2*3] = h_mat_block
                if (ibatch1 != ibatch2):
                    k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
                    for idx in range(hyp_mat0.shape[0]):
                        hyp_mat0[idx, s2*3:e2*3, s1*3:e1*3] = h_mat_block[idx].T
        pool.close()
        pool.join()

    del mat_slice

    # obtain noise parameter
    train_noise = True
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                train_noise = False
    # add gradient of noise variance
    if (train_noise):
        sigma_mat = np.zeros([1, size3, size3])
        sigma_mat[0, :, :] = np.eye(size3) * 2 * sigma_n
        hyp_mat = np.vstack([sigma_mat, hyp_mat0])
    else:
        hyp_mat = hyp_mat0

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return hyp_mat, ky_mat
