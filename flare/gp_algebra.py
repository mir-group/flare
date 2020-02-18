import math
import time
import numpy as np
import multiprocessing as mp

from typing import List, Callable
from flare.kernels.utils import from_mask_to_args, from_grad_to_mask

_global_training_data = {}
_global_training_labels = {}

def queue_wrapper(result_queue, wid,
                  func, args):
    """
    wrapper function for multiprocessing queue
    """
    result_queue.put((wid, func(*args)))

def partition_cr(n_sample, size, n_cpus):
    """
    partition the training data for matrix calculation
    the number of blocks are close to n_cpus
    since mp.Process does not allow to change the thread number
    """
    n_sample = int(math.ceil(np.sqrt(size*size/n_cpus/2.)))
    block_id=[]
    nbatch = 0
    nbatch1 = 0
    nbatch2 = 0
    e1 = 0
    while (e1 < size):
        s1 = int(n_sample*nbatch1)
        e1 = int(np.min([s1 + n_sample, size]))
        nbatch2 = nbatch1
        nbatch1 += 1
        e2 = 0
        while (e2 <size):
            s2 = int(n_sample*nbatch2)
            e2 = int(np.min([s2 + n_sample, size]))
            block_id += [(s1, e1, s2, e2)]
            nbatch2 += 1
            nbatch += 1
    return block_id, nbatch

def partition_c(n_sample, size, n_cpus):
    """
    partition the training data for vector calculation
    the number of blocks are the same as n_cpus
    since mp.Process does not allow to change the thread number
    """
    n_sample = int(math.ceil(size/n_cpus))
    block_id = []
    nbatch = 0
    e = 0
    while (e < size):
        s = n_sample*nbatch
        e = np.min([s + n_sample, size])
        block_id += [(s, e)]
        nbatch += 1
    return block_id, nbatch

def partition_update(n_sample, size, old_size, n_cpus):

    ns = int(math.ceil(size/n_sample))
    nproc = (size*3-old_size*3)*(ns+old_size*3)//2
    if (nproc < n_cpus):
        n_sample = int(math.ceil(size/np.sqrt(n_cpus*2)))
        ns = int(math.ceil(size/n_sample))

    ns_new = int(math.ceil((size-old_size)/n_sample))
    old_ns = int(math.ceil(old_size/n_sample))

    nbatch = 0
    block_id = []
    for ibatch1 in range(old_ns):
        s1 = int(n_sample*ibatch1)
        e1 = int(np.min([s1 + n_sample, old_size]))
        for ibatch2 in range(ns_new):
            s2 = int(n_sample*ibatch2)+old_size
            e2 = int(np.min([s2 + n_sample, size]))
            block_id += [(s1, e1, s2, e2)]
            nbatch += 1

    for ibatch1 in range(ns_new):
        s1 = int(n_sample*ibatch1)+old_size
        e1 = int(np.min([s1 + n_sample, size]))
        for ibatch2 in range(ns_new):
            s2 = int(n_sample*ibatch2)+old_size
            e2 = int(np.min([s2 + n_sample, size]))
            block_id += [(s1, e1, s2, e2)]
            nbatch += 1

    return block_id, nbatch


def obtain_noise_len(hyps, hyps_mask):
    """
    obtain the noise parameter from hyps and mask
    """

    # assume sigma_n is the final hyperparameter
    sigma_n = hyps[-1]
    # correct it if map is defined
    non_noise_hyps = len(hyps)-1
    train_noise = True
    if (hyps_mask is not None):
        train_noise = hyps_mask.get('train_noise', True)
        if (train_noise is False):
            sigma_n = hyps_mask['original'][-1]
            non_noise_hyps = len(hyps)

    return sigma_n, non_noise_hyps, train_noise


#######################################
##### KY MATRIX FUNCTIONS
#######################################

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
    args = from_mask_to_args(hyps, hyps_mask, cutoffs)

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
            kern_curr = kernel(x_1, x_2, d_1, d_2, *args)
            # store kernel value
            k_mat[m_index, n_index] = kern_curr
            if (same):
                k_mat[n_index, m_index] = kern_curr

    return k_mat

def get_ky_mat(hyps: np.ndarray, name: str,
               kernel, cutoffs=None, hyps_mask=None,
               n_cpus=1, n_sample=100):
    """ parallel version of get_ky_mat
    :param hyps: list of hyper-parameters
    :param name: name of the gp instance.
    :param kernel: function object of the kernel
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters

    :return: covariance matrix
    """

    training_data = _global_training_data[name]
    size = len(training_data)
    size3 = 3*size

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        k_mat = get_ky_mat_pack(
                hyps, name, 0, size, 0, size, True,
                kernel, cutoffs, hyps_mask)
    else:

        # initialize matrices
        block_id, nbatch = partition_cr(n_sample, size, n_cpus)

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

    sigma_n, _, __ = obtain_noise_len(hyps, hyps_mask)

    ky_mat = k_mat
    ky_mat += sigma_n ** 2 * np.eye(size3)

    return ky_mat


def get_like_from_ky_mat(ky_mat, name):
    """ compute the likelihood from the covariance matrix

    :param ky_mat: the covariance matrix

    :return: float, likelihood
    """
    # catch linear algebra errors
    try:
        ky_mat_inv = np.linalg.inv(ky_mat)
        l_mat = np.linalg.cholesky(ky_mat)
        alpha = np.matmul(ky_mat_inv, labels)
    except:
        return -1e8

    return get_like_from_mats(ky_mat, l_mat, alpha, name)

def get_like_from_mats(ky_mat, l_mat, alpha, name):
    """ compute the likelihood from the covariance matrix

    :param ky_mat: the covariance matrix

    :return: float, likelihood
    """
    # catch linear algebra errors
    labels = _global_training_labels[name]

    # calculate likelihood
    like = (-0.5 * np.matmul(labels, alpha) -
            np.sum(np.log(np.diagonal(l_mat))) -
            math.log(2 * np.pi) * ky_mat.shape[1] / 2)

    return like


#######################################
##### KY MATRIX FUNCTIONS and gradients
#######################################

def get_ky_and_hyp_pack(name, s1, e1, s2, e2, same: bool,
        hyps: np.ndarray, kernel_grad, cutoffs=None, hyps_mask=None):
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
    sigma_n, non_noise_hyps, _ = obtain_noise_len(hyps, hyps_mask)

    # initialize matrices
    size1 = (e1-s1) * 3
    size2 = (e2-s2) * 3
    k_mat = np.zeros([size1, size2])
    hyp_mat = np.zeros([non_noise_hyps, size1, size2])

    args = from_mask_to_args(hyps, hyps_mask, cutoffs)

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
            cov = kernel_grad(x_1, x_2, d_1, d_2, *args)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            grad = from_grad_to_mask(cov[1], hyps_mask)
            hyp_mat[:, m_index, n_index] = grad
            if (same):
                k_mat[n_index, m_index] = cov[0]
                hyp_mat[:, n_index, m_index] = grad

    return hyp_mat, k_mat


def get_ky_and_hyp(hyps: np.ndarray, name,
                   kernel_grad, cutoffs=None,
                   hyps_mask=None,
                   n_cpus=1, n_sample=100):
    """
    parallel version of get_ky_and_hyp

    :param hyps: list of hyper-parameters
    :param name: name of the gp instance.
    :param kernel_grad: function object of the kernel gradient
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters
    :param n_cpus: number of cpus to use.
    :param n_sample: the size of block for matrix to compute

    :return: hyp_mat, ky_mat
    """

    training_data = _global_training_data[name]
    size = len(training_data)
    size3 = size*3

    sigma_n, non_noise_hyps, train_noise = obtain_noise_len(hyps, hyps_mask)

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        hyp_mat0, k_mat = get_ky_and_hyp_pack(
                name, 0, size, 0, size, True,
                hyps, kernel_grad, cutoffs, hyps_mask)
    else:

        block_id, nbatch = partition_cr(n_sample, size, n_cpus)

        result_queue = mp.Queue()
        children = []
        for wid in range(nbatch):
            s1, e1, s2, e2 = block_id[wid]
            children.append(
                mp.Process(
                    target=queue_wrapper,
                    args=(result_queue, wid,
                        get_ky_and_hyp_pack,
                        (name, s1, e1, s2, e2, s1==s2,
                         hyps, kernel_grad, cutoffs, hyps_mask))))

        # Run child processes.
        for c in children:
            c.start()

        k_mat = np.zeros([size3, size3])
        hyp_mat0 = np.zeros([non_noise_hyps, size3, size3])
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
                       n_cpus=1, n_sample=100):
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
    :param n_sample: the size of block for matrix to compute

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
        get_ky_mat(hyps, name, kernel,
                   cutoffs=cutoffs, hyps_mask=hyps_mask,
                   n_cpus=n_cpus, n_sample=n_sample)

    output.write_to_log(f"get_key_mat {time.time()-time0}\n", name="hyps")

    time0 = time.time()

    like = get_like_from_ky_mat(ky_mat, name)

    output.write_to_log(f"get_like_from_ky_mat {time.time()-time0}\n", name="hyps")

    if output is not None:
        output.write_to_log('like: ' + str(like)+'\n', name="hyps")

    return -like


def get_neg_like_grad(hyps: np.ndarray, name: str,
                      kernel_grad, output = None,
                      cutoffs=None, hyps_mask=None,
                      n_cpus=1, n_sample=100):
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
    :param n_sample: the size of block for matrix to compute

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
        get_ky_and_hyp(hyps,
                       name,
                       kernel_grad,
                       cutoffs=cutoffs,
                       hyps_mask=hyps_mask,
                       n_cpus=n_cpus, n_sample=n_sample)

    if output is not None:
        output.write_to_log(f"get_ky_and_hyp {time.time()-time0}\n", name="hyps")

    time0 = time.time()

    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, name)

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



def get_ky_mat_update(ky_mat_old, hyps: np.ndarray, name: str,
                      kernel, cutoffs=None, hyps_mask=None,
                      n_cpus=1, n_sample=100):
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
    :param n_sample: the size of block for matrix to compute

    :return: updated covariance matrix

    '''

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return get_ky_mat_update_serial(
                ky_mat_old, hyps, name, kernel, cutoffs, hyps_mask)

    sigma_n, non_noise_hyps, _ = obtain_noise_len(hyps, hyps_mask)

    # initialize matrices
    old_size3 = ky_mat_old.shape[0]
    old_size = old_size3//3
    size = len(_global_training_data[name])
    size3 = 3*size
    ds = [1, 2, 3]

    block_id, nbatch = partition_update(n_sample, size, old_size, n_cpus)

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

def get_ky_mat_update_serial(\
        ky_mat_old, hyps: np.ndarray, name,
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

    args = from_mask_to_args(hyps, hyps_mask, cutoffs)

    # calculate elements
    for m_index in range(size3):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]
        low = int(np.max([m_index, n]))
        for n_index in range(low, size3):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]
            # calculate kernel
            kern_curr = kernel(x_1, x_2, d_1, d_2, *args)
            ky_mat[m_index, n_index] = kern_curr
            ky_mat[n_index, m_index] = kern_curr

    # matrix manipulation
    sigma_n, _, __ = obtain_noise_len(hyps, hyps_mask)
    ky_mat[n:, n:] += sigma_n ** 2 * np.eye(size3-n)
    return ky_mat


def get_kernel_vector_unit(name, s, e, x, d_1, kernel, hyps,
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

    size = (e-s)
    ds = [1, 2, 3]

    args = from_mask_to_args(hyps, hyps_mask, cutoffs)

    k_v = np.zeros(size*3, )
    for m_index in range(size):
        x_2 = _global_training_data[name][m_index+s]
        for d_2 in ds:
            k_v[m_index*3+d_2-1] = kernel(x, x_2, d_1, d_2, *args)

    return k_v


def get_kernel_vector(name, kernel, x, d_1, hyps,
                      cutoffs=None, hyps_mask=None,
                      n_cpus=1, n_sample=100):
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
    :param n_sample: the size of block for matrix to compute

    :return: kernel vector
    :rtype: np.ndarray
    """

    size = len(_global_training_data[name])

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return get_kernel_vector_unit(
                name, 0, size, x, d_1, kernel, hyps,
                cutoffs, hyps_mask)

    block_id, nbatch = partition_c(n_sample, size, n_cpus)

    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        s, e = block_id[wid]
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid,
                    get_kernel_vector_unit,
                    (name, s, e, x, d_1, kernel, hyps,
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


def en_kern_vec_unit(name, s, e, x, kernel,
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
    size = (e-s) * 3
    k_v = np.zeros(size, )

    args = from_mask_to_args(hyps, hyps_mask, cutoffs)

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))+s]
        d_2 = ds[m_index % 3]
        k_v[m_index] = kernel(x_2, x, d_2, *args)

    return k_v


def en_kern_vec(name, kernel,
                x, hyps,
                cutoffs=None, hyps_mask=None,
                n_cpus=1, n_sample=100):
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
    :param n_sample: the size of block for matrix to compute

    :return: kernel vector
    :rtype: np.ndarray
    """

    training_data = _global_training_data[name]
    size = len(training_data)

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return en_kern_vec_unit(
                name, 0, size, x, kernel, hyps,
                cutoffs, hyps_mask)

    block_id, nbatch = partition_c(n_sample, size, n_cpus)
    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid,
                    en_kern_vec_unit,
                    (name, block_id[wid][0], block_id[wid][1], x,
                     kernel, hyps, cutoffs, hyps_mask))))

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
