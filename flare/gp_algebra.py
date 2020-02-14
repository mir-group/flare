import math
import time
import numpy as np
import multiprocessing as mp

from typing import List, Callable

_global_training_data = {}
_global_training_labels = {}

def queue_wrapper(result_queue, wid,
                  func, args):
    result_queue.put((wid, func(*args)))

def partition_cr(nsample, size, n_cpus):
    nsample = int(math.ceil(np.sqrt(size*size/n_cpus/2.)))
    block_id=[]
    nbatch = 0
    nbatch1 = 0
    nbatch2 = 0
    e1 = 0
    while (e1 < size):
        s1 = int(nsample*nbatch1)
        e1 = int(np.min([s1 + nsample, size]))
        nbatch2 = nbatch1
        nbatch1 += 1
        e2 = 0
        while (e2 <size):
            s2 = int(nsample*nbatch2)
            e2 = int(np.min([s2 + nsample, size]))
            block_id += [(s1, e1, s2, e2)]
            nbatch2 += 1
            nbatch += 1
    return block_id, nbatch

def partition_c(nsample, size, n_cpus):
    # ns = int(math.ceil(size/nsample/n_cpus)*n_cpus)
    nsample = int(math.ceil(size/n_cpus))
    # ns = int(math.ceil(size/nsample))
    block_id = []
    nbatch = 0
    e = 0
    while (e < size):
        s = nsample*nbatch
        e = np.min([s + nsample, size])
        block_id += [(s, e)]
        nbatch += 1
    return block_id, nbatch

def partition_update(nsample, size, old_size, n_cpus):

    ns = int(math.ceil(size/nsample))
    nproc = (size*3-old_size*3)*(ns+old_size*3)//2
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

    return block_id, nbatch



#######################################
##### KY MATRIX FUNCTIONS
#######################################

def get_ky_mat(hyps: np.ndarray, name: str,
               kernel, cutoffs=None, hyps_mask=None):

    """ Compute covariance matrix K by comparing training data with itself
    :param hyps: list of hyper-parameters
    :param name: name of the gp instance.
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
        if (hyps_mask.get('train_noise', True) is False):
            sigma_n = hyps_mask['original'][-1]

    size = len(_global_training_data[name])
    # matrix manipulation
    k_mat = get_ky_mat_pack(hyps, name, 0, size,
                            0, size, True, kernel,
                            cutoffs, hyps_mask)
    size3 = size*3
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return ky_mat

def get_ky_mat_pack(hyps: np.ndarray, name: str,
               s1: int, e1: int, s2: int, e2: int,
               same: bool, kernel, cutoffs, hyps_mask):
    """ Compute covariance matrix element between set1 and set2
    :param hyps: list of hyper-parameters
    :param name: name of the gp instance.
    :param same: whether the row and column are the same
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: covariance matrix
    """


    # initialize matrices
    training_data = _global_training_data[name]
    size1 = (e1-s1)*3
    size2 = (e2-s2)*3
    k_mat = np.zeros([size1, size2])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size1):
        x_1 = training_data[int(math.floor(m_index / 3))+s1]
        d_1 = ds[m_index % 3]
        if (same):
            lowbound = m_index
        else:
            lowbound = 0
        for n_index in range(lowbound, size2):
            x_2 = training_data[int(math.floor(n_index / 3))+s2]
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

def get_ky_mat_par(hyps: np.ndarray, name: str,
                   kernel, cutoffs=None, hyps_mask=None,
                   n_cpus=None, nsample=100):
    """ parallel version of get_ky_mat
    :param hyps: list of hyper-parameters
    :param name: name of the gp instance.
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: covariance matrix
    """

    if (n_cpus is None):
        n_cpus =mp.cpu_count()
    if (n_cpus == 1):
        return get_ky_mat(hyps, name,
                          kernel, cutoffs, hyps_mask)

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        if (hyps_mask.get('train_noise', True) is False):
                sigma_n = hyps_mask['original'][-1]

    # initialize matrices

    training_data = _global_training_data[name]
    size = len(training_data)

    block_id, nbatch = partition_cr(nsample, size, n_cpus)

    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        s1, e1, s2, e2 = block_id[wid]
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid,
                    get_ky_mat_pack,
                    (hyps, name, s1, e1, s2, e2,
                   s1==s2, kernel, cutoffs, hyps_mask
                    )
                )
            )
        )

    # Run child processes.
    for c in children:
        c.start()

    # Wait for all results to arrive.
    size3 = 3*size
    k_mat = np.zeros([size3, size3])
    for _ in range(nbatch):
        wid, result_chunk = result_queue.get(block=True)
        s1, e1, s2, e2 = block_id[wid]
        k_mat[s1*3:e1*3, s2*3:e2*3] = result_chunk
        if (s1 != s2):
            k_mat[s2*3:e2*3, s1*3:e1*3] = result_chunk.T

    # Join child processes (clean up zombies).
    for c in children:
        c.join()

    # matrix manipulation
    del result_queue
    del children

    ky_mat = k_mat
    ky_mat += sigma_n ** 2 * np.eye(size3)

    return ky_mat


def get_like_from_ky_mat(ky_mat):
    """ compute the likelihood from the covariance matrix

    :param ky_mat: the covariance matrix

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


def get_ky_and_hyp(hyps: np.ndarray, name: str,
                   kernel_grad, cutoffs=None, hyps_mask=None):

    """
    computes ky matrix and its derivative to hyper-parameter
    If the cpu set up is None, it uses as much as posible cpus

    :param hyps: list of hyper-parameters
    :param name: name of the gp instance.
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: hyp_mat, ky_mat
    """

    training_data = _global_training_data[name]
    size = len(training_data)
    hyp_mat0, k_mat = get_ky_and_hyp_pack(hyps, name, 0, size,
                                          0, size, True,
                                          kernel_grad,
                                          cutoffs=cutoffs,
                                          hyps_mask=hyps_mask)


    # obtain noise parameter
    train_noise = True
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        train_noise = hyps_mask.get('train_noise', True)
        if (train_noise is False):
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


def get_ky_and_hyp_pack(hyps: np.ndarray, name, s1, e1,
                   s2, e2, same: bool,
                   kernel_grad, cutoffs=None, hyps_mask=None):
    """
    computes a block of ky matrix and its derivative to hyper-parameter
    If the cpu set up is None, it uses as much as posible cpus

    :param hyps: list of hyper-parameters
    :param name: name of the gp instance.
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
        train_noise = hyps_mask.get('train_noise', True)
        if (train_noise is False):
            non_noise_hyps = len(hyps)

    # initialize matrices
    size1 = (e1-s1) * 3
    size2 = (e2-s2) * 3
    k_mat = np.zeros([size1, size2])
    hyp_mat = np.zeros([non_noise_hyps, size1, size2])

    ds = [1, 2, 3]

    training_data = _global_training_data[name]
    # calculate elements
    for m_index in range(size1):
        x_1 = training_data[int(math.floor(m_index / 3))+s1]
        d_1 = ds[m_index % 3]

        if (same):
            lowbound = m_index
        else:
            lowbound = 0
        for n_index in range(lowbound, size2):
            x_2 = training_data[int(math.floor(n_index / 3))+s2]
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


def get_ky_and_hyp_par(hyps: np.ndarray, name,
                       kernel_grad, cutoffs=None,
                       hyps_mask=None,
                       n_cpus=None, nsample=100):
    """
    parallel version of get_ky_and_hyp

    :param hyps: list of hyper-parameters
    :param name: name of the gp instance.
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
        return get_ky_and_hyp(hyps, name,
                              kernel_grad,
                              cutoffs, hyps_mask)

    # assume sigma_n is the final hyperparameter
    number_of_hyps = len(hyps)
    non_noise_hyps = len(hyps)-1
    sigma_n = hyps[-1]
    train_noise = True
    if (hyps_mask is not None):
        train_noise = hyps_mask.get('train_noise', True)
        if (train_noise is False):
            sigma_n = hyps_mask['original'][-1]
            non_noise_hyps = len(hyps)

    # initialize matrices
    training_data = _global_training_data[name]
    size = len(training_data)
    size3 = size*3
    k_mat = np.zeros([size3, size3])
    hyp_mat0 = np.zeros([non_noise_hyps, size3, size3])

    block_id, nbatch = partition_cr(nsample, size, n_cpus)

    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        s1, e1, s2, e2 = block_id[wid]
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid,
                    get_ky_and_hyp_pack,
                    (hyps, name, s1, e1, s2, e2,
                   s1==s2, kernel_grad, cutoffs, hyps_mask
                    )
                )
            )
        )

    # Run child processes.
    for c in children:
        c.start()

    for _ in range(nbatch):
        wid, result_chunk = result_queue.get(block=True)
        s1, e1, s2, e2 = block_id[wid]
        h_mat_block, k_mat_block = result_chunk
        k_mat[s1*3:e1*3, s2*3:e2*3] = k_mat_block
        hyp_mat0[:, s1*3:e1*3, s2*3:e2*3] = h_mat_block
        if (s1 != s2):
            k_mat[s2*3:e2*3, s1*3:e1*3] = k_mat_block.T
            for idx in range(hyp_mat0.shape[0]):
                hyp_mat0[idx, s2*3:e2*3, s1*3:e1*3] = h_mat_block[idx].T

    # Join child processes (clean up zombies).
    for c in children:
        c.join()

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


def get_neg_likelihood(hyps: np.ndarray, name,
                       kernel: Callable, output = None,
                       cutoffs=None, hyps_mask=None,
                       n_cpus=None, nsample=100):
    """compute the negative log likelihood

    :param hyps: list of hyper-parameters
    :type hyps: np.ndarray
    :param name: name of the gp instance.
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
        get_ky_mat_par(hyps, name, kernel,
                       cutoffs=cutoffs, hyps_mask=hyps_mask,
                       n_cpus=n_cpus, nsample=nsample)

    output.write_to_log(f"get_key_mat {time.time()-time0}\n", name="hyps")

    time0 = time.time()

    like = get_like_from_ky_mat(ky_mat, _global_training_labels[name])

    output.write_to_log(f"get_like_from_ky_mat {time.time()-time0}\n", name="hyps")

    if output is not None:
        output.write_to_log('like: ' + str(like)+'\n', name="hyps")

    return -like


def get_neg_like_grad(hyps: np.ndarray, name: str,
                      kernel_grad, output = None,
                      cutoffs=None, hyps_mask=None,
                      n_cpus=None, nsample=100):
    """compute the log likelihood and its gradients

    :param hyps: list of hyper-parameters
    :type hyps: np.ndarray
    :param name: name of the gp instance.
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
        ostring="hyps:"
        for hyp in hyps:
            ostring+=f" {hyp}"
        ostring+="\n"
        output.write_to_log(ostring, name="hyps")

    hyp_mat, ky_mat = \
        get_ky_and_hyp_par(hyps,
                           training_data,
                           kernel_grad,
                           cutoffs=cutoffs,
                           hyps_mask=hyps_mask,
                           n_cpus=n_cpus, nsample=nsample)

    if output is not None:
        output.write_to_log(f"get_ky_and_hyp {time.time()-time0}\n", name="hyps")

    time0 = time.time()

    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, _global_training_labels[name])

    print("like", like, like_grad)
    print("hyps", hyps)
    print("\n")

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


def get_like_grad_from_mats(ky_mat, hyp_mat, name):
    """compute the gradient of likelihood to hyper-parameters
    from covariance matrix and its gradient

    :param ky_mat: covariance matrix
    :type ky_mat: np.array
    :param hyp_mat: dky/d(hyper parameter) matrix
    :type hyp_mat: np.array
    :param name: name of the gp instance.

    :return: float, list. the likelihood and its gradients
    """

    number_of_hyps = hyp_mat.shape[0]

    # catch linear algebra errors
    try:
        ky_mat_inv = np.linalg.inv(ky_mat)
        l_mat = np.linalg.cholesky(ky_mat)
    except:
        return -1e8, np.zeros(number_of_hyps)

    labels = _global_training_labels[name]

    alpha = np.matmul(ky_mat_inv, labels)
    alpha_mat = np.matmul(alpha.reshape(-1, 1),
                          alpha.reshape(1, -1))
    like_mat = alpha_mat - ky_mat_inv

    # calculate likelihood
    like = (-0.5 * np.matmul(labels, alpha) -
            np.sum(np.log(np.diagonal(l_mat))) -
            math.log(2 * np.pi) * ky_mat.shape[1] / 2)

    # calculate likelihood gradient
    like_grad = np.zeros(number_of_hyps)
    for n in range(number_of_hyps):
        like_grad[n] = 0.5 * \
                       np.trace(np.matmul(like_mat, hyp_mat[n, :, :]))

    return like, like_grad



def get_ky_mat_update_par(ky_mat_old, hyps: np.ndarray, name: str,
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
        return get_ky_mat_update(ky_mat_old, hyps, name,
                                 kernel, cutoffs, hyps_mask)

    # assume sigma_n is the final hyperparameter
    sigma_n = hyps[-1]
    if (hyps_mask is not None):
        if (hyps_mask.get('train_noise', True) is False):
            sigma_n = hyps_mask['original'][-1]

    # initialize matrices
    old_size3 = ky_mat_old.shape[0]
    old_size = old_size3//3
    size = len(_global_training_data[name])
    size3 = 3*size
    ds = [1, 2, 3]

    block_id, nbatch = partition_update(nsample, size, old_size, n_cpus)

    # Send and Run child processes.
    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        s1, e1, s2, e2 = block_id[wid]
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid,
                    get_ky_mat_pack,
                    (hyps, name, s1, e1, s2, e2,
                   s1==s2, kernel, cutoffs, hyps_mask
                    )
                )
            )
        )
    for c in children:
        c.start()

    # Wait for all results to arrive.
    size3 = 3*size
    ky_mat = np.zeros([size3, size3])
    ky_mat[:old_size3, :old_size3] = ky_mat_old
    del ky_mat_old
    for _ in range(nbatch):
        wid, result_chunk = result_queue.get(block=True)
        s1, e1, s2, e2 = block_id[wid]
        ky_mat[s1*3:e1*3, s2*3:e2*3] = result_chunk
        if (s1 != s2):
            ky_mat[s2*3:e2*3, s1*3:e1*3] = result_chunk.T

    # Join child processes (clean up zombies).
    for c in children:
        c.join()

    # matrix manipulation
    ky_mat[old_size3:, old_size3:] += sigma_n ** 2 * np.eye(size3-old_size3)

    return ky_mat

def get_ky_mat_update(ky_mat_old, hyps: np.ndarray, name,
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

    training_data = _global_training_data[name]

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
        if (hyps_mask.get('train_noise', True) is False):
            sigma_n = hyps_mask['original'][-1]
    ky_mat[n:, n:] += sigma_n ** 2 * np.eye(size3-n)
    return ky_mat


def get_kernel_vector_unit(name, s, e, kernel, x, d_1, hyps,
                      cutoffs, hyps_mask):
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

    size = (e-s)*3
    ds = [1, 2, 3]
    s3 = s*3

    k_v = np.zeros(size, )
    for m_index in range(s, e):
        x_2 = _global_training_data[name][m_index]
        for d_2 in ds:
            if (hyps_mask is not None):
                k_v[m_index*3+d_2-1-s3] = kernel(x, x_2, d_1, d_2,
                                      hyps, cutoffs, hyps_mask=hyps_mask)
            else:
                k_v[m_index*3+d_2-1-s3] = kernel(x, x_2, d_1, d_2,
                                      hyps, cutoffs)

    return k_v


def get_kernel_vector_par(name, kernel,
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
        return get_kernel_vector(name, kernel,
                                 x, d_1, hyps,
                                 cutoffs, hyps_mask)

    size = len(_global_training_data[name])
    block_id, nbatch = partition_c(nsample, size, n_cpus)

    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        s, e = block_id[wid]
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid,
                    get_kernel_vector_unit,
                    (name, s, e, kernel, x, d_1, hyps,
                  cutoffs, hyps_mask
                    )
                )
            )
        )

    # Run child processes.
    for c in children:
        c.start()

    # Wait for all results to arrive.
    k12_v = np.zeros(size*3)
    for _ in range(nbatch):
        wid, result_chunk = result_queue.get(block=True)
        s, e = block_id[wid]
        k12_v[s*3:e*3] = result_chunk

    # Join child processes (clean up zombies).
    for c in children:
        c.join()

    return k12_v

def get_kernel_vector(name, kernel,
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

    training_data = _global_training_data[name]
    size = len(training_data) * 3

    k_v = np.zeros(size, )
    ds = [1, 2, 3]
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

def en_kern_vec_unit(name, s, e, kernel, x,
                     hyps, cutoffs=None, hyps_mask=None):
    """
    Compute energy kernel vector, comparing input environment to all environments
    in the GP's training set.
    :param training_data: Set of atomic environments to compare against
    :param kernel:
    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param hyps: list of hyper-parameters
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: kernel vector
    :rtype: np.ndarray
    """

    training_data = _global_training_data[name]

    ds = [1, 2, 3]
    size = (s-e) * 3
    k_v = np.zeros(size, )

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))+s]
        d_2 = ds[m_index % 3]
        if (hyps_mask is not None):
            k_v[m_index] = kernel(x_2, x, d_2,
                                  hyps, cutoffs, hyps_mask=hyps_mask)
        else:
            k_v[m_index] = kernel(x_2, x, d_2,
                                  hyps, cutoffs)
    return k_v

def en_kern_vec_par(name, kernel,
                    x, hyps,
                    cutoffs=None, hyps_mask=None,
                    n_cpus=None, nsample=100):
    """
    Compute kernel vector, comparing input environment to all environments
    in the GP's training set.

    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param hyps: list of hyper-parameters
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters
    :param n_cpus: number of cpus to use.
    :param nsample: the size of block for matrix to compute

    :return: kernel vector
    :rtype: np.ndarray
    """

    training_data = _global_training_data[name]

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return en_kern_vec(name, kernel,
                           x, hyps,
                           cutoffs, hyps_mask)

    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid,
                    en_kern_vec_unit,
                    (name,
                  block_id[wid][0],
                  block_id[wid][1],
                  kernel, x, hyps,
                  cutoffs, hyps_mask
                    )
                )
            )
        )

    # Run child processes.
    for c in children:
        c.start()

    # Wait for all results to arrive.
    k12_v = np.zeros(size*3)
    for _ in range(nbatch):
        wid, result_chunk = result_queue.get(block=True)
        s, e = block_id[wid]
        k12_v[s*3:e*3] = result_chunk

    # Join child processes (clean up zombies).
    for c in children:
        c.join()

    return k12_v

def en_kern_vec(name, kernel, x, hyps, cutoffs=None,
                hyps_mask=None):
    """
    Compute kernel vector, comparing input environment to all environments
    in the GP's training set.

    :param x: data point to compare against kernel matrix
    :type x: AtomicEnvironment
    :param hyps: list of hyper-parameters
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    """

    training_data = _global_training_data[name]

    ds = [1, 2, 3]
    size = len(training_data) * 3
    k_v = np.zeros(size, )

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))]
        d_2 = ds[m_index % 3]
        if (hyps_mask is not None):
            k_v[m_index] = kernel(x_2, x, d_2,
                                  hyps, cutoffs,
                                  hyps_mask=hyps_mask)
        else:
            k_v[m_index] = kernel(x_2, x, d_2,
                                  hyps, cutoffs)

    return k_v
