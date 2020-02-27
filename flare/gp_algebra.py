import math
import time
import numpy as np
import multiprocessing as mp

from typing import List, Callable

#######################################
##### KY MATRIX FUNCTIONS
#######################################

def get_ky_mat(hyps: np.ndarray, training_data: list,
               kernel, cutoffs=None, hyps_mask=None):

    """ Compute covariance matrix K by comparing training data with itself
    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: covariance matrix
    """

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[number_of_hyps - 1]
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask):
            if (hyps_mask['train_noise'] is False):
                sigma_n = hyps_mask['original'][-1]

    # matrix manipulation
    k_mat = get_ky_mat_pack(hyps, training_data,
                            training_data, True, kernel,
                            cutoffs, hyps_mask)
    size3 = len(training_data)*3
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return ky_mat


def get_ky_mat_pack(hyps: np.ndarray, training_data1: list,
               training_data2:list, same: bool,
               kernel, cutoffs=None, hyps_mask=None):
    """ Compute covariance matrix element between set1 and set2
    :param hyps: list of hyper-parameters
    :param training_data1: row elements
    :param training_data2: column elements
    :param same: whether the row and column are the same
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

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
            if (hyps_mask is not None):
                kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
                                   cutoffs, hyps_mask=hyps_mask)
            else:
                kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
                                   cutoffs)

            # store kernel value
            k_mat[m_index, n_index] = kern_curr
            if (same):
                k_mat[n_index, m_index] = kern_curr

    return k_mat


def get_ky_mat_par(hyps: np.ndarray, training_data: list,
                   kernel, cutoffs=None, hyps_mask=None,
                   n_cpus=None, nsample=100):
    """ parallel version of get_ky_mat
    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: covariance matrix
    """

    if (n_cpus is None):
        n_cpus =mp.cpu_count()
    if (n_cpus == 1):
        return get_ky_mat(hyps, training_data,
                          kernel, cutoffs, hyps_mask)

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
    k_mat_slice = []
    k_mat = np.zeros([size3, size3])
    with mp.Pool(processes=n_cpus) as pool:

        ns = int(math.ceil(size/nsample))
        nproc = ns*(ns+1)//2
        if (nproc < n_cpus):
            nsample = int(math.ceil(size/int(np.sqrt(n_cpus*2))))
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
            k_mat_slice.append(pool.apply_async(
                                      get_ky_mat_pack,
                                      args=(hyps,
                                        t1, t2, bool(s1==s2),
                                        kernel, cutoffs,
                                        hyps_mask)))
            count += 1
            if (count >= n_cpus*3):
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
    ky_mat = k_mat
    del k_mat_block
    ky_mat += sigma_n ** 2 * np.eye(size3)

    return ky_mat


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

#######################################
##### KY MATRIX FUNCTIONS and gradients
#######################################


def get_ky_and_hyp(hyps: np.ndarray, training_data: list,
                   kernel_grad, cutoffs=None, hyps_mask=None):

    """
    computes ky matrix and its derivative to hyper-parameter
    If the cpu set up is None, it uses as much as posible cpus

    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: hyp_mat, ky_mat
    """

    hyp_mat0, k_mat = get_ky_and_hyp_pack(hyps, training_data,
                                          training_data, True,
                                          kernel_grad,
                                          cutoffs=cutoffs,
                                          hyps_mask=hyps_mask)


    # obtain noise parameter
    train_noise = True
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask):
            if (hyps_mask['train_noise'] is False):
                train_noise = False
                sigma_n = hyps_mask['original'][-1]
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


def get_ky_and_hyp_pack(hyps: np.ndarray, training_data1: list,
                   training_data2: list, same: bool,
                   kernel_grad, cutoffs=None, hyps_mask=None):
    """
    computes a block of ky matrix and its derivative to hyper-parameter
    If the cpu set up is None, it uses as much as posible cpus

    :param hyps: list of hyper-parameters
    :param training_data1: row elements
    :param training_data2: column elements
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: hyp_mat, ky_mat
    """

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    non_noise_hyps = len(hyps)-1
    train_noise = True
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                train_noise = False
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
            if (hyps_mask is not None):
                cov = kernel_grad(x_1, x_2, d_1, d_2, hyps,
                                  cutoffs=cutoffs, hyps_mask=hyps_mask)
            else:
                cov = kernel_grad(x_1, x_2, d_1, d_2, hyps, cutoffs)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            hyp_mat[:, m_index, n_index] = cov[1]

            if (same):
                k_mat[n_index, m_index] = cov[0]
                hyp_mat[:, n_index, m_index] = cov[1]

    return hyp_mat, k_mat


def get_ky_and_hyp_par(hyps: np.ndarray, training_data: list,
                       kernel_grad, cutoffs=None,
                       hyps_mask=None,
                       n_cpus=None, nsample=100):
    """
    parallel version of get_ky_and_hyp

    :param hyps: list of hyper-parameters
    :param training_data: list of atomic envirionments
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters
    :param n_cpus: number of cpus to use.
    :param nsample: the size of block for matrix to compute

    :return: hyp_mat, ky_mat
    """


    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return get_ky_and_hyp(hyps, training_data,
                              kernel_grad,
                              cutoffs, hyps_mask)

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

    with mp.Pool(processes=n_cpus) as pool:

        ns = int(math.ceil(size/nsample))
        nproc = ns*(ns+1)//2
        if (nproc < n_cpus):
            nsample = int(math.ceil(size/np.sqrt(n_cpus*2)))
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

        count = 0
        base = 0
        mat_slice = []
        for ibatch in range(nbatch):
            s1, e1, s2, e2 = block_id[ibatch]
            if (size>5000):
                print("sending", s1, e1, s2, e2)
            t1 = training_data[s1:e1]
            t2 = training_data[s2:e2]
            mat_slice.append(pool.apply_async(
                                    get_ky_and_hyp_pack,
                                    args=(
                                        hyps, t1, t2,
                                        bool(s1==s2),
                                        kernel_grad,
                                        cutoffs, hyps_mask)))

            count += 1
            if (count > n_cpus*3):
                for iget in range(base, count+base):
                    s1, e1, s2, e2 = block_id[iget]
                    h_mat_block, k_mat_block = mat_slice[iget-base].get()
                    k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                    hyp_mat0[:, s1*3:e1*3, s2*3:e2*3] = h_mat_block
                    if (s1 != s2):
                        k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
                        for idx in range(hyp_mat0.shape[0]):
                            hyp_mat0[idx, s2*3:e2*3, s1*3:e1*3] = h_mat_block[idx].T
                    if (size>5000):
                        print("computed block", base, base+count)
                mat_slice = []
                base = ibatch+1
                count = 0
        if (count>0):
            for iget in range(base, nbatch):
                s1, e1, s2, e2 = block_id[iget]
                h_mat_block, k_mat_block = mat_slice[iget-base].get()
                k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
                hyp_mat0[:, s1*3:e1*3, s2*3:e2*3] = h_mat_block
                if (s1 != s2):
                    k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
                    for idx in range(hyp_mat0.shape[0]):
                        hyp_mat0[idx, s2*3:e2*3, s1*3:e1*3] = h_mat_block[idx].T
            del mat_slice
        pool.close()
        pool.join()

    # add gradient of noise variance
    if (train_noise):
        hyp_mat = np.zeros([hyp_mat0.shape[0]+1,
                            hyp_mat0.shape[1],
                            hyp_mat0.shape[2]])
        hyp_mat[:-1, :, :] = hyp_mat0
        hyp_mat[-1, :, :] = np.eye(size3) * 2 * sigma_n
    else:
        hyp_mat = hyp_mat0

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return hyp_mat, ky_mat


def get_neg_likelihood(hyps: np.ndarray, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel: Callable, output = None,
                       cutoffs=None, hyps_mask=None,
                       n_cpus=None, nsample=100):
    """compute the negative log likelihood

    :param hyps: list of hyper-parameters
    :type hyps: np.ndarray
    :param training_data: Set of atomic environments to compare against
    :type training_data: list of AtomicEnvironment objects
    :param training_labels_np: forces
    :type training_labels_np: np.array
    :param kernel: function object of the kernel
    :param output: Output object for dumping every hyper-parameter
                   sets computed
    :type output: class Output
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters
    :param n_cpus: number of cpus to use.
    :param nsample: the size of block for matrix to compute

    :return: float
    """


    if output is not None:
        ostring="hyps:"
        for hyp in hyps:
            ostring+=f" {hyp}"
        ostring+="\n"
        output.write_to_log(ostring, name="hyps")

    time0 = time.time()
    ky_mat = \
        get_ky_mat_par(hyps, training_data, kernel,
                       cutoffs=cutoffs, hyps_mask=hyps_mask,
                       n_cpus=n_cpus, nsample=nsample)

    output.write_to_log(f"get_key_mat {time.time()-time0}\n", name="hyps")

    time0 = time.time()

    like = get_like_from_ky_mat(ky_mat, training_labels_np)

    output.write_to_log(f"get_like_from_ky_mat {time.time()-time0}\n",
                        name="hyps")

    if output is not None:
        output.write_to_log('like: ' + str(like)+'\n', name="hyps")

    return -like


def get_neg_like_grad(hyps: np.ndarray, training_data: list,
                      training_labels_np: np.ndarray,
                      kernel_grad, output = None,
                      cutoffs=None, hyps_mask=None,
                      n_cpus=None, nsample=100,
                      print_progress=False):
    """compute the log likelihood and its gradients

    :param hyps: list of hyper-parameters
    :type hyps: np.ndarray
    :param training_data: Set of atomic environments to compare against
    :type training_data: list of AtomicEnvironment objects
    :param training_labels_np: forces
    :type training_labels_np: np.array
    :param kernel_grad: function object of the kernel gradient
    :param output: Output object for dumping every hyper-parameter
                   sets computed
    :type output: class Output
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters
    :param n_cpus: number of cpus to use.
    :param nsample: the size of block for matrix to compute

    :return: float, np.array
    """

    time0 = time.time()
    if output is not None:
        ostring = "hyps:"
        for hyp in hyps:
            ostring +=  f" {hyp}"
        ostring += "\n"
        output.write_to_log(ostring, name="hyps")

    hyp_mat, ky_mat = \
        get_ky_and_hyp_par(hyps,
                           training_data,
                           kernel_grad,
                           cutoffs=cutoffs,
                           hyps_mask=hyps_mask,
                           n_cpus=n_cpus, nsample=nsample)

    if output is not None:
        output.write_to_log(f"get_ky_and_hyp {time.time()-time0}\n",
                            name="hyps")

    time0 = time.time()

    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, training_labels_np)

    if output is not None:
        output.write_to_log(f"get_like_grad_from_mats {time.time()-time0}\n",
                            name="hyps")

    if output is not None:
        ostring = "like grad:"
        for lg in like_grad:
            ostring += f" {lg}"
        ostring += "\n"
        output.write_to_log(ostring, name="hyps")
        output.write_to_log('like: ' + str(like)+'\n', name="hyps")

    if print_progress:
        print('\nhyperparameters: '+str(hyps))
        print('likelihood: ' + str(like))
        print('likelihood gradient: ' + str(like_grad))

    return -like, -like_grad


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


def get_ky_mat_update_par(ky_mat_old, hyps: np.ndarray, training_data: list,
                      kernel, cutoffs=None, hyps_mask=None,
                      n_cpus=None, nsample=100):
    '''
    used for update_L_alpha, especially for parallelization
    parallelized for added atoms, for example, if add 10 atoms to the training
    set, the K matrix will add 10x3 columns and 10x3 rows, and the task will
    be distributed to 30 processors

    :param ky_mat_old: old covariance matrix
    :param hyps: list of hyper-parameters
    :param training_data: Set of atomic environments to compare against
    :param kernel:
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters
    :param n_cpus: number of cpus to use.
    :param nsample: the size of block for matrix to compute

    :return: updated covariance matrix

    '''

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return get_ky_mat_update(ky_mat_old, hyps, training_data,
                                 kernel, cutoffs, hyps_mask)

    # assume sigma_n is the final hyperparameter
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                sigma_n = hyps_mask['original'][-1]

    # initialize matrices
    old_size3 = ky_mat_old.shape[0]
    old_size = old_size3//3
    size = len(training_data)
    size3 = 3*len(training_data)
    ky_mat = np.zeros([size3, size3])
    ky_mat[:old_size3, :old_size3] = ky_mat_old
    ds = [1, 2, 3]
    with mp.Pool(processes=n_cpus) as pool:
        ns = int(math.ceil(size/nsample))
        nproc = (size3-old_size3)*(ns+old_size3)//2
        if (nproc < n_cpus):
            nsample = int(math.ceil(size/np.sqrt(n_cpus*2)))
            ns = int(math.ceil(size/nsample))

        ns_new = int(math.ceil((size-old_size)/nsample))
        old_ns = int(math.ceil(old_size/nsample))

        nbatch = 0
        block_id = []
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
                                        kernel, cutoffs, hyps_mask))]
            count += 1
            if (count >= n_cpus*3):
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
                      kernel, cutoffs=None, hyps_mask=None):
    '''
    used for update_L_alpha. if add 10 atoms to the training
    set, the K matrix will add 10x3 columns and 10x3 rows


    :param ky_mat_old: old covariance matrix
    :param hyps: list of hyper-parameters
    :param training_data: Set of atomic environments to compare against
    :param kernel:
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

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
            if (hyps_mask is not None):
                kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
                                   cutoffs, hyps_mask=hyps_mask)
            else:
                kern_curr = kernel(x_1, x_2, d_1, d_2, hyps,
                                   cutoffs)
            # store kernel value
            ky_mat[m_index, n_index] = kern_curr
            ky_mat[n_index, m_index] = kern_curr

    # matrix manipulation
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        if ('train_noise' in hyps_mask.keys()):
            if (hyps_mask['train_noise'] is False):
                sigma_n = hyps_mask['original'][-1]
    ky_mat[n:, n:] += sigma_n ** 2 * np.eye(size3-n)
    return ky_mat


def get_kernel_vector_unit(training_data, kernel, x,
                      d_1, hyps,
                      cutoffs=None, hyps_mask=None):
    """
    Compute kernel vector, comparing input environment to all environments
    in the GP's training set.
    :param training_data: Set of atomic environments to compare against
    :param kernel:
    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param d_1: Cartesian component of force vector to get (1=x,2=y,3=z)
    :type d_1: int
    :param hyps: list of hyper-parameters
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: kernel vector
    :rtype: np.ndarray
    """


    ds = [1, 2, 3]
    size = len(training_data) * 3
    k_v = np.zeros(size, )

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))]
        d_2 = ds[m_index % 3]
        if (hyps_mask is not None):
            k_v[m_index] = kernel(x, x_2, d_1, d_2,
                                  hyps, cutoffs, hyps_mask=hyps_mask)
        else:
            k_v[m_index] = kernel(x, x_2, d_1, d_2,
                                  hyps, cutoffs)
    return k_v

def get_kernel_vector_par(training_data, kernel,
                          x, d_1, hyps,
                          cutoffs=None, hyps_mask=None,
                          n_cpus=None, nsample=100):
    """
    Compute kernel vector, comparing input environment to all environments
    in the GP's training set.

    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param d_1: Cartesian component of force vector to get (1=x,2=y,3=z)
    :type d_1: int
    :param hyps: list of hyper-parameters
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters
    :param n_cpus: number of cpus to use.
    :param nsample: the size of block for matrix to compute

    :return: kernel vector
    :rtype: np.ndarray
    """

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return get_kernel_vector(training_data, kernel,
                                 x, d_1, hyps,
                                 cutoffs, hyps_mask)

    with mp.Pool(processes=n_cpus) as pool:

        # sort of partition
        size = len(training_data)
        ns = int(math.ceil(size/nsample/n_cpus)*n_cpus)
        nsample = int(math.ceil(size/ns))
        ns = int(math.ceil(size/nsample))

        k12_slice = []
        for ibatch in range(ns):
            s = nsample*ibatch
            e = np.min([s + nsample, size])
            k12_slice.append(pool.apply_async(get_kernel_vector_unit,
                                              args=(training_data[s: e],
                                                    kernel, x, d_1, hyps,
                                                    cutoffs, hyps_mask)))

        size3 = size*3
        nsample3 = nsample*3
        k12_v = np.zeros(size3)
        for ibatch in range(ns):
            s = nsample3*ibatch
            e = np.min([s + nsample3, size3])
            k12_v[s:e] = k12_slice[ibatch].get()
        pool.close()
        pool.join()

    return k12_v

def get_kernel_vector(training_data, kernel,
                      x, d_1: int,
                      hyps, cutoffs=None, hyps_mask=None):
    """
    Compute kernel vector, comparing input environment to all environments
    in the GP's training set.

    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param d_1: Cartesian component of force vector to get (1=x,2=y,3=z)
    :type d_1: int
    :param hyps: list of hyper-parameters
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    """

    ds = [1, 2, 3]
    size = len(training_data) * 3
    k_v = np.zeros(size, )

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))]
        d_2 = ds[m_index % 3]
        if (hyps_mask is not None):
            k_v[m_index] = kernel(x, x_2, d_1, d_2,
                                  hyps, cutoffs,
                                  hyps_mask=hyps_mask)
        else:
            k_v[m_index] = kernel(x, x_2, d_1, d_2,
                                  hyps, cutoffs)

    return k_v
