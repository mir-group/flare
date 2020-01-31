import numpy as np
import math
import multiprocessing as mp
import time

from typing import List, Callable

#######################################
##### KY MATRIX FUNCTIONS
#######################################

def get_ky_mat_par(hyps: np.ndarray, training_data: list,
                   kernel: Callable, cutoffs=None, ncpus=None, nsample=100):
    """
    Parallelized version of function which computes ky matrix

    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel: 
    :param cutoffs:
    :param ncpus: number of cpus to use.

    :return: ky_mat
    """

    if (ncpus is None):
        ncpus = mp.cpu_count()
    if (ncpus == 1):
        return get_ky_mat(hyps, training_data,
                          kernel, cutoffs)

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]

    # initialize matrices
    size = len(training_data)
    size3 = 3*len(training_data)
    k_mat = np.zeros([size3, size3])
    with mp.Pool(processes=ncpus) as pool:
        ns = int(math.ceil(size/nsample))
        nproc = ns*(ns+1)//2
        if (nproc < ncpus):
            nsample = int(size/int(np.sqrt(ncpus*2)))
            ns = int(math.ceil(size/nsample))

        block_id = []
        nbatch = 0
        for ibatch1 in range(ns):
            s1 = int(nsample*ibatch1)
            e1 = int(np.min([s1 + nsample, size]))
            for ibatch2 in range(ibatch1, ns):
                s2 = int(nsample*ibatch2)
                e2 = int(np.min([s2 + nsample, size]))
                block_id += [(s1, e1, s2, e2)]
                nbatch += 1

        k_mat_slice = []
        count = 0
        base = 0
        time0 = time.time()
        for ibatch in range(nbatch):
            s1, e1, s2, e2 = block_id[ibatch]
            t1 = training_data[s1:e1]
            t2 = training_data[s2:e2]
            k_mat_slice += [pool.apply_async(
                                      get_ky_mat_pack,
                                      args=(hyps,
                                        t1, t2,
                                        bool(s1==s2),
                                        kernel, cutoffs))]
            count += 1
            if (count >= ncpus*3):
                for iget in range(base, count+base):
                    s1, e1, s2, e2 = block_id[iget]
                    k_mat_block = k_mat_slice[iget-base].get()
                    k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                    if (s1 != s2):
                        k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
                if (size>5000):
                    print("computed block", base, base+count, nbatch, time.time()-time0)
                    time0 = time.time()
                k_mat_slice = []
                base = ibatch+1
                count = 0
        if (count>0):
            if (size>5000):
                print("computed block", base, base+count, nbatch, time.time()-time0)
                time0 = time.time()
            for iget in range(base, nbatch):
                s1, e1, s2, e2 = block_id[iget]
                k_mat_block = k_mat_slice[iget-base].get()
                k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                if (s1 != s2):
                    k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
            del k_mat_slice
        pool.close()
        pool.join()

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return ky_mat

def get_ky_mat_pack(hyps: np.ndarray, training_data1: list,
               training_data2:list, same: bool,
               kernel: Callable, cutoffs):
    """ Compute covariance matrix K by comparing training data with itself

    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers

    :return: covariance matrix
    """

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
                       kernel_grad: Callable, cutoffs=None, ncpus=None, nsample=100):
    """
    Parallelized version of function which computes ky matrix
    and its derivative to hyper-parameter
    If the cpu set up is None, it uses as much as posible cpus

    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param ncpus: number of cpus to use.

    :return: hyp_mat, ky_mat
    """

    if (ncpus is None):
        ncpus = mp.cpu_count()
    if (ncpus == 1):
        return get_ky_and_hyp(hyps, training_data,
                              kernel_grad, cutoffs)

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
        nproc = ns*(ns+1)//2
        if (nproc < ncpus):
            nsample = int(size/int(np.sqrt(ncpus*2)))
            ns = int(math.ceil(size/nsample))

        block_id = []
        nbatch = 0
        for ibatch1 in range(ns):
            s1 = int(nsample*ibatch1)
            e1 = int(np.min([s1 + nsample, size]))
            for ibatch2 in range(ibatch1, ns):
                s2 = int(nsample*ibatch2)
                e2 = int(np.min([s2 + nsample, size]))
                block_id += [(s1, e1, s2, e2)]
                nbatch += 1

        mat_slice = []
        count = 0
        base = 0
        time0 = time.time()
        for ibatch in range(nbatch):
            s1, e1, s2, e2 = block_id[ibatch]
            t1 = training_data[s1:e1]
            t2 = training_data[s2:e2]
            mat_slice += [pool.apply_async(
                                    get_ky_and_hyp_pack,
                                    args=(
                                        hyps, t1, t2,
                                        bool(s1==s2), kernel_grad, cutoffs))]
            count += 1
            if (count >= ncpus*3):
                for iget in range(base, count+base):
                    s1, e1, s2, e2 = block_id[iget]
                    h_mat_block, k_mat_block = mat_slice[iget-base].get()
                    k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                    hyp_mat[:-1, s1*3:e1*3, s2*3:e2*3] = h_mat_block
                    if (s1 != s2):
                        k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
                        for idx in range(hyp_mat.shape[0]-1):
                            hyp_mat[idx, s2*3:e2*3, s1*3:e1*3] = h_mat_block[idx].T
                if (size>5000):
                    print("computed block", base, base+count, nbatch, time.time()-time0)
                    time0 = time.time()
                del mat_slice
                mat_slice = []
                base = ibatch+1
                count = 0
        if (count>0):
            if (size>5000):
                print("computed block", base, base+count, nbatch, time.time()-time0)
                time0 = time.time()
            for iget in range(base, nbatch):
                s1, e1, s2, e2 = block_id[iget]
                s1, e1, s2, e2 = block_id[iget]
                h_mat_block, k_mat_block = mat_slice[iget-base].get()
                k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                hyp_mat[:-1, s1*3:e1*3, s2*3:e2*3] = h_mat_block
                if (s1 != s2):
                    k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
                    for idx in range(hyp_mat.shape[0]-1):
                        hyp_mat[idx, s2*3:e2*3, s1*3:e1*3] = h_mat_block[idx].T
            del mat_slice
        pool.close()
        pool.join()

    # add gradient of noise variance
    hyp_mat[-1, :, :] = np.eye(size3) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return hyp_mat, ky_mat

def get_ky_and_hyp_pack(hyps: np.ndarray, training_data1: list,
                   training_data2: list, same: bool,
                   kernel_grad: Callable, cutoffs=None):
    """ Compute covariance matrix K and matrix to estimate likelihood gradient
    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param training_data2: list of atomic envirionments
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers

    :return: matrix gradient, and itself
    """

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
               kernel: Callable, cutoffs=None):
    """ Compute covariance matrix K by comparing training data with itself
    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers

    :return: covariance matrix
    """

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
                   kernel_grad: Callable, cutoffs=None):
    """ Compute covariance matrix K and matrix to estimate likelihood gradient
    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :return: matrix gradient, and itself
    """
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

def get_ky_mat_update_par(ky_mat_old, hyps: np.ndarray, training_data: list,
                      kernel: Callable, cutoffs=None, ncpus=None, nsample=100):
    '''
    used for update_L_alpha, especially for parallelization
    parallelized for added atoms, for example, if add 10 atoms to the training
    set, the K matrix will add 10x3 columns and 10x3 rows, and the task will
    be distributed to 30 processors

    :param ky_mat_old:
    :param training_data: Set of atomic environments to compare against
    :param kernel:

    :return: updated covariance matrix
    '''

    if (ncpus is None):
        ncpus = mp.cpu_count()
    if (ncpus == 1):
        return get_ky_mat_update(ky_mat_old, hyps, training_data,
                                 kernel, cutoffs)

    # assume sigma_n is the final hyperparameter
    sigma_n = hyps[-1]

    # initialize matrices
    old_size = ky_mat_old.shape[0]//3
    old_size3 = ky_mat_old.shape[0]
    size = len(training_data)
    size3 = 3*len(training_data)
    ky_mat = np.zeros([size3, size3])
    ky_mat[:old_size3, :old_size3] = ky_mat_old
    with mp.Pool(processes=ncpus) as pool:

        ns = int(math.ceil(size/nsample))
        nproc = (size3-old_size3)*(ns+old_size3)//2
        if (nproc < ncpus):
            nsample = int(size/int(np.sqrt(ncpus*2)))
            ns = int(math.ceil(size/nsample))

        ns_new = int(math.ceil((size-old_size)/nsample))
        old_ns = int(math.ceil(old_size/nsample))

        block_id = []
        nbatch = 0
        for ibatch1 in range(old_ns):
            s1 = int(nsample*ibatch1)
            e1 = int(np.min([s1 + nsample, old_size]))
            for ibatch2 in range(ns_new):
                s2 = int(nsample*ibatch2)+old_size
                e2 = int(np.min([s2 + nsample, size]))
                block_id += [(s1, e1, s2, e2)]
                nbatch += 1

        for ibatch1 in range(ns_new):
            s1 = int(nsample*ibatch1)+old_size
            e1 = int(np.min([s1 + nsample, size]))
            for ibatch2 in range(ns_new):
                s2 = int(nsample*ibatch2)+old_size
                e2 = int(np.min([s2 + nsample, size]))
                block_id += [(s1, e1, s2, e2)]
                nbatch += 1

        k_mat_slice = []
        count = 0
        base = 0
        time0 = time.time()
        for ibatch in range(nbatch):
            s1, e1, s2, e2 = block_id[ibatch]
            t1 = training_data[s1:e1]
            t2 = training_data[s2:e2]
            k_mat_slice += [pool.apply_async(
                                      get_ky_mat_pack,
                                      args=(hyps,
                                        t1, t2,
                                        bool(s1==s2),
                                        kernel, cutoffs))]
            count += 1
            if (count >= ncpus*3):
                for iget in range(base, count+base):
                    s1, e1, s2, e2 = block_id[iget]
                    k_mat_block = k_mat_slice[iget-base].get()
                    ky_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                    if (s1 != s2):
                        ky_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
                if (size>5000):
                    print("computed block", base, base+count, nbatch, time.time()-time0)
                    time0 = time.time()
                k_mat_slice = []
                base = ibatch+1
                count = 0
        if (count>0):
            if (size>5000):
                print("computed block", base, base+count, nbatch, time.time()-time0)
                time0 = time.time()
            for iget in range(base, nbatch):
                s1, e1, s2, e2 = block_id[iget]
                k_mat_block = k_mat_slice[iget-base].get()
                ky_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                if (s1 != s2):
                    ky_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
            del k_mat_slice
        pool.close()
        pool.join()

    # matrix manipulation
    ky_mat[old_size3:, old_size3:] += sigma_n ** 2 * np.eye(size3-old_size3)

    return ky_mat

def get_ky_mat_update(ky_mat_old, hyps: np.ndarray, training_data: list,
                      kernel: Callable, cutoffs=None):
    '''
    used for update_L_alpha, especially for parallelization
    parallelized for added atoms, for example, if add 10 atoms to the training
    set, the K matrix will add 10x3 columns and 10x3 rows, and the task will
    be distributed to 30 processors

    :param ky_mat_old:
    :param training_data: Set of atomic environments to compare against
    :param kernel:

    :return: updated covariance matrix
    '''
    n = ky_mat_old.shape[0]
    size = len(training_data)
    size3 = size*3
    m = size - n // 3  # number of new data added
    ky_mat = np.zeros((size3, size3))
    ky_mat[:n, :n] = ky_mat_old

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size3):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]
        low = int(np.max([m_index, n]))
        for n_index in range(low, size3):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
                               cutoffs)
            # store kernel value
            ky_mat[m_index, n_index] = kern_curr
            ky_mat[n_index, m_index] = kern_curr

    # matrix manipulation
    sigma_n = hyps[-1]
    ky_mat[n:, n:] += sigma_n ** 2 * np.eye(size3-n)
    return ky_mat

#######################################
##### LIKELIHOOD + LIKELIHOOD GRADIENT
#######################################

def get_like_from_ky_mat(ky_mat, training_labels_np):
    """ compute the likelihood from the covariance matrix

    :param ky_mat: the covariance matrix
    :param training_labels_np: the numpy array of forces
    :type training_labels_np: np.array

    :return: float, likelihood
    """
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
    """compute the gradient of likelihood to hyper-parameters
    from covariance matrix and its gradient

    :param ky_mat: covariance matrix
    :type ky_mat: np.array
    :param hyp_mat: dky/d(hyper parameter) matrix
    :type hyp_mat: np.array
    :param training_labels_np: forces
    :type training_labels_np: np.array

    :return: float, list. the likelihood and its gradients
    """

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
                       kernel: Callable, output = None,
                       cutoffs=None,
                       ncpus=None, nsample=100):
    """compute the negative log likelihood

    :param hyps: list of hyper-parameters
    :type hyps: np.ndarray
    :param training_data: Set of atomic environments to compare against
    :type training_data: list of AtomicEnvironment objects
    :param training_labels_np: forces
    :type training_labels_np: np.array
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param output: Output object for dumping every hyper-parameter
                   sets computed
    :type output: class Output
    :param ncpus: number of cpus to use.

    :return: float
    """

    ky_mat = get_ky_mat_par(hyps, training_data,
                            kernel, cutoffs, ncpus, nsample)

    like = get_like_from_ky_mat(ky_mat, training_labels_np)

    if output is not None:
        output.write_hyps(None, hyps, None, like, "NA", name="hyps")

    return -like


def get_neg_like_grad(hyps: np.ndarray, training_data: list,
                      training_labels_np: np.ndarray,
                      kernel_grad: Callable, cutoffs=None,
                      output = None, ncpus=None, nsample=100):
    """compute the log likelihood and its gradients

    :param hyps: list of hyper-parameters
    :type hyps: np.ndarray
    :param training_data: Set of atomic environments to compare against
    :type training_data: list of AtomicEnvironment objects
    :param training_labels_np: forces
    :type training_labels_np: np.array
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param output: Output object for dumping every hyper-parameter
                   sets computed
    :type output: class Output
    :param ncpus: number of cpus to use.

    :return: float, np.array
    """


    hyp_mat, ky_mat = \
        get_ky_and_hyp_par(hyps, training_data,
                           kernel_grad, cutoffs, ncpus, nsample)

    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, training_labels_np)

    if output is not None:
        output.write_hyps(None, hyps, None, like, like_grad, name="hyps")

    return -like, -like_grad

def get_kernel_vector_unit(training_data, kernel, x,
                      d_1, hyps, cutoffs):
    """
    Compute kernel vector, comparing input environment to all environments
    in the GP's training set.
    :param training_data: Set of atomic environments to compare against
    :param kernel:
    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param d_1: Cartesian component of force vector to get (1=x,2=y,3=z)
    :type d_1: int
    :return: kernel vector
    :rtype: np.ndarray
    """

    ds = [1, 2, 3]
    size = len(training_data) * 3
    k_v = np.zeros(size, )

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))]
        d_2 = ds[m_index % 3]
        k_v[m_index] = kernel(x, x_2, d_1, d_2,
                              hyps, cutoffs)
    return k_v

def get_kernel_vector_par(training_data, kernel, x,
                          d_1, hyps, cutoffs,
                          ncpus=None, nsample=100):
    """
    Compute kernel vector, comparing input environment to all environments
    in the GP's training set.
    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param d_1: Cartesian component of force vector to get (1=x,2=y,3=z)
    :type d_1: int
    :return: kernel vector
    :rtype: np.ndarray
    """

    if (ncpus is None):
        ncpus = mp.cpu_count()
    if (ncpus == 1):
        return get_kernel_vector(training_data, kernel,
                                 x, d_1, hyps, cutoffs)

    with mp.Pool(processes=processes) as pool:
        size = len(training_data)
        ns = int(math.ceil(size/nsample))
        if (ns < ncpus):
            nsample = int(size/int(ncpus))
            ns = int(math.ceil(size/nsample))
        k12_slice = []
        for ibatch in range(ns):

            s = nsample*ibatch
            e = np.min([s + nsample, size])
            k12_slice.append(pool.apply_async(get_kernel_vector_unit,
                                              args=(training_data[s: e],
                                                    kernel, x, d_1, hyps,
                                                    cutoffs)))
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

def get_kernel_vector(training_data, kernel,
                      x, d_1: int,
                      hyps, cutoff):
    """
    Compute kernel vector, comparing input environment to all environments
    in the GP's training set.
    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param d_1: Cartesian component of force vector to get (1=x,2=y,3=z)
    :type d_1: int
    :return: kernel vector
    :rtype: np.ndarray
    """

    ds = [1, 2, 3]
    size = len(training_data) * 3
    k_v = np.zeros(size, )

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))]
        d_2 = ds[m_index % 3]
        k_v[m_index] = kernel(x, x_2, d_1, d_2,
                              hyps, cutoffs)

    return k_v
