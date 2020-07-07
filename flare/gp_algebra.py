import logging
import math
import multiprocessing as mp
import numpy as np
import time

from typing import List, Callable
from flare.kernels.utils import from_mask_to_args, from_grad_to_mask

_global_training_data = {}
_global_training_labels = {}
_global_training_structures = {}
_global_energy_labels = {}


def queue_wrapper(result_queue, wid,
                  func, args):
    """
    wrapper function for multiprocessing queue
    """
    result_queue.put((wid, func(*args)))


def partition_matrix(n_sample, size, n_cpus):
    """
    partition the training data for matrix calculation
    the number of blocks are close to n_cpus
    since mp.Process does not allow to change the thread number
    """

    # divide the block by n_cpu partitions, with size n_sample0
    # if the divided chunk is smaller than the requested chunk n_sample
    # use the requested chunk size
    n_unique_elements = size * (size + 1) / 2
    n_sample0 = int(math.ceil(np.sqrt(n_unique_elements / n_cpus)))
    if (n_sample0 > n_sample):
        n_sample = n_sample0

    block_id = []
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
        while (e2 < size):
            s2 = int(n_sample * nbatch2)
            e2 = int(np.min([s2 + n_sample, size]))
            block_id += [(s1, e1, s2, e2)]
            nbatch2 += 1
            nbatch += 1

    return block_id, nbatch


def partition_matrix_custom(n_sample: int, start1, end1, start2, end2, n_cpus):
    """
    Partition a specified portion of a matrix.
    """

    n_rows = end1 - start1
    n_cols = end2 - start2
    n_unique_elements = n_rows * n_cols
    n_sample0 = int(math.ceil(np.sqrt(n_unique_elements / n_cpus)))
    if (n_sample0 > n_sample):
        n_sample = n_sample0

    block_id = []
    n_block = 0  # total number of blocks
    nbatch1 = 0
    nbatch2 = 0
    e1 = 0
    while (e1 < end1):
        s1 = start1 + n_sample * nbatch1
        e1 = np.min([s1 + n_sample, end1])
        nbatch1 += 1

        e2 = 0
        nbatch2 = 0
        while (e2 < end2):
            s2 = start2 + n_sample * nbatch2
            e2 = np.min([s2 + n_sample, end2])
            block_id += [(s1, e1, s2, e2)]
            nbatch2 += 1
            n_block += 1  # update block count

    return block_id, n_block


def partition_vector(n_sample, size, n_cpus):
    """
    partition the training data for vector calculation
    the number of blocks are the same as n_cpus
    since mp.Process does not allow to change the thread number
    """
    n_sample0 = int(math.ceil(size/n_cpus))
    if (n_sample0 > n_sample):
        n_sample = n_sample0

    block_id = []
    nbatch = 0
    e = 0
    while (e < size):
        s = n_sample*nbatch
        e = np.min([s + n_sample, size])
        block_id += [(s, e)]
        nbatch += 1
    return block_id, nbatch


def partition_force_energy_block(n_sample: int, size1: int, size2: int,
                                 n_cpus: int):
    """Special partition method for the force/energy block. Because the number
    of environments in a structure can vary, we only split up the environment
    list, which has length size1.

    Note that two sizes need to be specified: the size of the envionment
    list and the size of the structure list.

    Args:
        n_sample (int): Number of environments per processor.
        size1 (int): Size of the environment list.
        size2 (int): Size of the structure list.
        n_cpus (int): Number of cpus.
    """

    n_sample0 = int(math.ceil(size1/n_cpus))
    if (n_sample0 > n_sample):
        n_sample = n_sample0

    block_id = []
    nbatch = 0
    e = 0

    while (e < size1):
        s = n_sample * nbatch
        e = np.min([s + n_sample, size1])
        block_id += [(s, e, 0, size2)]
        nbatch += 1

    return block_id, nbatch


def partition_update(n_sample, size, old_size, n_cpus):

    n_unique_elements = (size - old_size) * size
    n_sample0 = int(math.ceil(np.sqrt(n_unique_elements / n_cpus)))
    if (n_sample0 > n_sample):
        n_sample = n_sample0

    ns_new = int(math.ceil((size - old_size) / n_sample))
    old_ns = int(math.ceil(old_size / n_sample))

    nbatch = 0
    block_id = []
    for ibatch1 in range(old_ns):
        s1 = int(n_sample*ibatch1)
        e1 = int(np.min([s1 + n_sample, old_size]))

        for ibatch2 in range(ns_new):
            s2 = int(n_sample * ibatch2) + old_size
            e2 = int(np.min([s2 + n_sample, size]))
            block_id += [(s1, e1, s2, e2)]
            nbatch += 1

    for ibatch1 in range(ns_new):
        s1 = int(n_sample * ibatch1) + old_size
        e1 = int(np.min([s1 + n_sample, size]))

        for ibatch2 in range(ns_new):
            s2 = int(n_sample * ibatch2) + old_size
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
            sigma_n = hyps_mask['original_hyps'][-1]
            non_noise_hyps = len(hyps)

    return sigma_n, non_noise_hyps, train_noise


# --------------------------------------------------------------------------
#                   Parallel matrix/vector construction
# --------------------------------------------------------------------------

def parallel_matrix_construction(pack_function, hyps, name, kernel, cutoffs,
                                 hyps_mask, block_id, nbatch, size1, size2,
                                 m1, m2, symm=True):
    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        s1, e1, s2, e2 = block_id[wid]
        children.append(mp.Process(target=queue_wrapper,
                                   args=(result_queue, wid, pack_function,
                                         (hyps, name, s1, e1, s2, e2, s1 == s2,
                                          kernel, cutoffs, hyps_mask))))

    # Run child processes.
    for c in children:
        c.start()

    # Wait for all results to arrive.
    matrix = np.zeros((size1, size2))
    for _ in range(nbatch):
        wid, result_chunk = result_queue.get(block=True)
        s1, e1, s2, e2 = block_id[wid]
        matrix[s1 * m1:e1 * m1,
               s2 * m2:e2 * m2] = result_chunk
        # Note that the force/energy block is not symmetric; in this case
        # symm is False.
        if ((s1 != s2) and (symm is True)):
            matrix[s2 * m2:e2 * m2, s1 * m1:e1 * m1] = result_chunk.T

    # Join child processes (clean up zombies).
    for c in children:
        c.join()

    # matrix manipulation
    del result_queue
    del children

    return matrix


def parallel_vector_construction(pack_function, name, x, kernel, hyps,
                                 cutoffs, hyps_mask, block_id, nbatch, size,
                                 mult, d_1=None):
    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        s, e = block_id[wid]
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid, pack_function,
                      (name, s, e, x, kernel, hyps, cutoffs, hyps_mask, d_1))))

    # Run child processes.
    for c in children:
        c.start()

    # Wait for all results to arrive.
    vector = np.zeros(size * mult)
    for _ in range(nbatch):
        wid, result_chunk = result_queue.get(block=True)
        s, e = block_id[wid]
        vector[s * mult:e * mult] = result_chunk

    # Join child processes (clean up zombies).
    for c in children:
        c.join()

    return vector


def multiple_array_construction(pack_function, name, x, kernel, hyps,
                                cutoffs, hyps_mask, block_id, nbatch, size,
                                mult, array_sizes):
    result_queue = mp.Queue()
    children = []
    for wid in range(nbatch):
        s, e = block_id[wid]
        children.append(
            mp.Process(
                target=queue_wrapper,
                args=(result_queue, wid, pack_function,
                      (name, s, e, x, kernel, hyps, cutoffs, hyps_mask))))

    # Run child processes.
    for c in children:
        c.start()

    # Initialize arrays.
    n_arrays = len(array_sizes)
    arrays = []
    for n in range(n_arrays):
        arrays.append(np.zeros(array_sizes[n]))

    # Wait for all results to arrive.
    for _ in range(nbatch):
        wid, result_chunk = result_queue.get(block=True)
        s, e = block_id[wid]
        for n in range(n_arrays):
            arrays[n][:, s * mult:e * mult] = result_chunk[n]

    # Join child processes (clean up zombies).
    for c in children:
        c.join()

    return arrays


# --------------------------------------------------------------------------
#                            Ky construction
# --------------------------------------------------------------------------

def get_force_block_pack(hyps: np.ndarray, name: str, s1: int, e1: int,
                         s2: int, e2: int, same: bool, kernel, cutoffs,
                         hyps_mask):
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
    force_block = np.zeros([size1, size2])

    ds = [1, 2, 3]

    # calculate elements
    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

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
            force_block[m_index, n_index] = kern_curr
            if (same):
                force_block[n_index, m_index] = kern_curr

    return force_block


def get_energy_block_pack(hyps: np.ndarray, name: str, s1: int, e1: int,
                          s2: int, e2: int, same: bool, kernel, cutoffs,
                          hyps_mask):

    # initialize matrices
    training_structures = _global_training_structures[name]
    size1 = e1 - s1
    size2 = e2 - s2
    energy_block = np.zeros([size1, size2])

    # calculate elements
    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    for m_index in range(size1):
        struc_1 = training_structures[m_index + s1]
        if (same):
            lowbound = m_index
        else:
            lowbound = 0

        for n_index in range(lowbound, size2):
            struc_2 = training_structures[n_index + s2]

            # Loop over environments in both structures to compute the
            # energy/energy kernel.
            kern_curr = 0
            for environment_1 in struc_1:
                for environment_2 in struc_2:
                    kern_curr += kernel(environment_1, environment_2, *args)

            # Store kernel value.
            energy_block[m_index, n_index] = kern_curr
            if (same):
                energy_block[n_index, m_index] = kern_curr

    return energy_block


def get_force_energy_block_pack(hyps: np.ndarray, name: str, s1: int,
                                e1: int, s2: int, e2: int, same: bool, kernel,
                                cutoffs, hyps_mask):
    # initialize matrices
    training_data = _global_training_data[name]
    training_structures = _global_training_structures[name]
    size1 = (e1 - s1) * 3
    size2 = e2 - s2
    force_energy_block = np.zeros([size1, size2])

    ds = [1, 2, 3]

    # calculate elements
    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    for m_index in range(size1):
        environment_1 = training_data[int(math.floor(m_index / 3)) + s1]
        d_1 = ds[m_index % 3]

        for n_index in range(size2):
            structure = training_structures[n_index + s2]

            # Loop over environments in the training structure.
            kern_curr = 0
            for environment_2 in structure:
                kern_curr += kernel(environment_1, environment_2, d_1, *args)

            # store kernel value
            force_energy_block[m_index, n_index] = kern_curr

    return force_energy_block


def get_force_block(hyps: np.ndarray, name: str, kernel, cutoffs=None,
                    hyps_mask=None, n_cpus=1, n_sample=100):
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
    size3 = 3 * size

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        k_mat = \
            get_force_block_pack(hyps, name, 0, size, 0, size, True, kernel,
                                 cutoffs, hyps_mask)
    else:
        # initialize matrices
        block_id, nbatch = partition_matrix(n_sample, size, n_cpus)
        mult = 3

        k_mat = \
            parallel_matrix_construction(get_force_block_pack, hyps, name,
                                         kernel, cutoffs, hyps_mask,
                                         block_id, nbatch, size3, size3,
                                         mult, mult)

    sigma_n, _, __ = obtain_noise_len(hyps, hyps_mask)
    force_block = k_mat
    force_block += sigma_n ** 2 * np.eye(size3)

    return force_block


def get_energy_block(hyps: np.ndarray, name: str, kernel, energy_noise,
                     cutoffs=None, hyps_mask=None, n_cpus=1, n_sample=100):
    training_structures = _global_training_structures[name]
    size = len(training_structures)

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        k_mat = \
            get_energy_block_pack(hyps, name, 0, size, 0, size, True, kernel,
                                  cutoffs, hyps_mask)
    else:
        block_id, nbatch = partition_matrix(n_sample, size, n_cpus)
        mult = 1

        k_mat = \
            parallel_matrix_construction(get_energy_block_pack, hyps, name,
                                         kernel, cutoffs, hyps_mask,
                                         block_id, nbatch, size, size, mult,
                                         mult)

    energy_block = k_mat
    energy_block += (energy_noise ** 2) * np.eye(size)

    return energy_block


def get_force_energy_block(hyps: np.ndarray, name: str, kernel, cutoffs=None,
                           hyps_mask=None, n_cpus=1, n_sample=100):
    training_data = _global_training_data[name]
    training_structures = _global_training_structures[name]
    size1 = len(training_data) * 3
    size2 = len(training_structures)
    size3 = len(training_data)

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        force_energy_block = \
            get_force_energy_block_pack(hyps, name, 0, size3, 0, size2, True,
                                        kernel, cutoffs, hyps_mask)
    else:
        # initialize matrices
        block_id, nbatch = \
            partition_force_energy_block(n_sample, size3, size2, n_cpus)
        m1 = 3  # 3 force components per environment
        m2 = 1  # 1 energy per structure

        force_energy_block = \
            parallel_matrix_construction(get_force_energy_block_pack, hyps,
                                         name, kernel, cutoffs, hyps_mask,
                                         block_id, nbatch, size1, size2, m1,
                                         m2, symm=False)

    return force_energy_block


def get_Ky_mat(hyps: np.ndarray, name: str, force_kernel: Callable,
               energy_kernel: Callable, force_energy_kernel: Callable,
               energy_noise, cutoffs=None, hyps_mask=None,
               n_cpus=1, n_sample=100):

    training_data = _global_training_data[name]
    training_structures = _global_training_structures[name]
    size1 = len(training_data) * 3
    size2 = len(training_structures)

    # Initialize Ky.
    ky_mat = np.zeros((size1 + size2, size1 + size2))

    # Assemble the full covariance matrix block-by-block.
    force_block = get_force_block(hyps, name, force_kernel, cutoffs, hyps_mask,
                                  n_cpus, n_sample)

    energy_block = get_energy_block(hyps, name, energy_kernel, energy_noise,
                                    cutoffs, hyps_mask, n_cpus, n_sample)

    force_energy_block = \
        get_force_energy_block(hyps, name, force_energy_kernel, cutoffs,
                               hyps_mask, n_cpus, n_sample)

    ky_mat[0:size1, 0:size1] = force_block
    ky_mat[size1:, size1:] = energy_block
    ky_mat[0:size1, size1:] = force_energy_block
    ky_mat[size1:, 0:size1] = force_energy_block.transpose()

    return ky_mat


# --------------------------------------------------------------------------
#                              Ky updates
# --------------------------------------------------------------------------

def update_force_block(ky_mat_old: np.ndarray, n_envs_prev: int,
                       hyps: np.ndarray, name: str, kernel, cutoffs=None,
                       hyps_mask=None, n_cpus=1, n_sample=100):

    old_size = n_envs_prev
    old_size3 = n_envs_prev * 3
    mult = 3
    size = len(_global_training_data[name])
    size3 = 3 * size

    if (n_cpus is None):
        n_cpus = mp.cpu_count()

    # serial version
    if (n_cpus == 1):
        force_block = np.zeros((size3, size3))
        new_mat = \
            get_force_block_pack(hyps, name, 0, size, old_size, size, False,
                                 kernel, cutoffs, hyps_mask)
        force_block[:, old_size3:] = new_mat
        force_block[old_size3:, :] = new_mat.transpose()

    # parallel version
    else:
        block_id, nbatch = partition_update(n_sample, size, old_size, n_cpus)

        force_block = \
            parallel_matrix_construction(get_force_block_pack, hyps, name,
                                         kernel, cutoffs, hyps_mask,
                                         block_id, nbatch, size3, size3,
                                         mult, mult)

    # insert previous covariance matrix
    force_block[:old_size3, :old_size3] = \
        ky_mat_old[:old_size3, :old_size3]

    # add the noise parameter
    sigma_n, _, _ = obtain_noise_len(hyps, hyps_mask)
    force_block[old_size3:, old_size3:] += \
        sigma_n ** 2 * np.eye(size3-old_size3)

    return force_block


def update_energy_block(ky_mat_old: np.ndarray, n_envs_prev: int,
                        hyps: np.ndarray, name: str, kernel,
                        energy_noise: float, cutoffs=None, hyps_mask=None,
                        n_cpus=1, n_sample=100):

    old_size = ky_mat_old.shape[0] - 3 * n_envs_prev
    mult = 1
    size = len(_global_training_structures[name])

    if (n_cpus is None):
        n_cpus = mp.cpu_count()

    # serial version
    if (n_cpus == 1):
        energy_block = np.zeros((size, size))
        new_mat = \
            get_energy_block_pack(hyps, name, 0, size, old_size, size, False,
                                  kernel, cutoffs, hyps_mask)
        energy_block[:, old_size:] = new_mat
        energy_block[old_size:, :] = new_mat.transpose()

    # parallel version
    else:
        block_id, nbatch = partition_update(n_sample, size, old_size, n_cpus)

        energy_block = \
            parallel_matrix_construction(get_energy_block_pack, hyps, name,
                                         kernel, cutoffs, hyps_mask,
                                         block_id, nbatch, size, size,
                                         mult, mult)

    # insert previous covariance matrix (if it has nonzero size)
    if old_size > 0:
        energy_block[:old_size, :old_size] = ky_mat_old[-old_size:, -old_size:]

    # add the noise parameter
    energy_block[old_size:, old_size:] += \
        (energy_noise ** 2) * np.eye(size - old_size)

    return energy_block


def update_force_energy_block(ky_mat_old: np.ndarray, n_envs_prev: int,
                              hyps: np.ndarray, name: str, kernel,
                              energy_noise: float, cutoffs=None,
                              hyps_mask=None, n_cpus=1, n_sample=100):

    n_strucs_prev = ky_mat_old.shape[0] - 3 * n_envs_prev
    n_envs = len(_global_training_data[name])
    n_strucs = len(_global_training_structures[name])
    force_energy_block = np.zeros((n_envs * 3, n_strucs))

    if (n_cpus is None):
        n_cpus = mp.cpu_count()

    # serial version
    if (n_cpus == 1):
        mat_1 = \
            get_force_energy_block_pack(hyps, name, 0, n_envs,
                                        n_strucs_prev, n_strucs,
                                        False, kernel, cutoffs, hyps_mask)

        mat_2 = \
            get_force_energy_block_pack(hyps, name, n_envs_prev, n_envs,
                                        0, n_strucs_prev,
                                        False, kernel, cutoffs, hyps_mask)

    # parallel version
    else:
        force_energy_block = np.zeros((n_envs * 3, n_strucs))

        block_id_1, nbatch_1 = \
            partition_matrix_custom(n_sample, 0, n_envs, n_strucs_prev,
                                    n_strucs, n_cpus)
        block_id_2, nbatch_2 = \
            partition_matrix_custom(n_sample, n_envs_prev, n_envs,
                                    0, n_strucs_prev, n_cpus)

        size1 = n_envs * 3
        size2 = n_strucs
        m1 = 3
        m2 = 1

        mat_1 = \
            parallel_matrix_construction(get_force_energy_block_pack,
                                         hyps, name, kernel, cutoffs,
                                         hyps_mask, block_id_1, nbatch_1,
                                         size1, size2, m1, m2, False)
        mat_2 = \
            parallel_matrix_construction(get_force_energy_block_pack,
                                         hyps, name, kernel, cutoffs,
                                         hyps_mask, block_id_2, nbatch_2,
                                         size1, size2, m1, m2, False)

        # reduce the size of the matrices
        mat_1 = mat_1[:n_envs * 3, n_strucs_prev:n_strucs]
        mat_2 = mat_2[n_envs_prev * 3:n_envs * 3, :n_strucs_prev]

    force_energy_block[:n_envs * 3, n_strucs_prev:n_strucs] = mat_1
    force_energy_block[n_envs_prev * 3:n_envs * 3, :n_strucs_prev] = mat_2

    # insert previous covariance matrix
    force_energy_block[:n_envs_prev * 3, :n_strucs_prev] = \
        ky_mat_old[:n_envs_prev * 3,
                   n_envs_prev * 3:n_envs_prev * 3 + n_strucs_prev]

    return force_energy_block


def get_ky_mat_update(ky_mat_old: np.ndarray, n_envs_prev: int,
                      hyps: np.ndarray, name: str, force_kernel: Callable,
                      energy_kernel: Callable, force_energy_kernel: Callable,
                      energy_noise: float, cutoffs=None, hyps_mask=None,
                      n_cpus=1, n_sample=100):

    n_envs = len(_global_training_data[name])
    n_strucs = len(_global_training_structures[name])
    size1 = n_envs * 3
    size2 = n_strucs
    ky_mat = np.zeros((size1 + size2, size1 + size2))

    force_block = \
        update_force_block(ky_mat_old, n_envs_prev, hyps, name, force_kernel,
                           cutoffs, hyps_mask, n_cpus, n_sample)

    energy_block = \
        update_energy_block(ky_mat_old, n_envs_prev, hyps, name, energy_kernel,
                            energy_noise, cutoffs, hyps_mask, n_cpus, n_sample)

    force_energy_block = \
        update_force_energy_block(ky_mat_old, n_envs_prev, hyps, name,
                                  force_energy_kernel, energy_noise, cutoffs,
                                  hyps_mask, n_cpus, n_sample)

    ky_mat[0:size1, 0:size1] = force_block
    ky_mat[size1:, size1:] = energy_block
    ky_mat[0:size1, size1:] = force_energy_block
    ky_mat[size1:, 0:size1] = force_energy_block.transpose()

    return ky_mat


# --------------------------------------------------------------------------
#                            Kernel vectors
# --------------------------------------------------------------------------

def energy_energy_vector_unit(name, s, e, x, kernel, hyps, cutoffs=None,
                              hyps_mask=None, d_1=None):
    """
    Gets part of the energy/energy vector.
    """

    training_structures = _global_training_structures[name]

    size = e - s
    energy_energy_unit = np.zeros(size, )

    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    for m_index in range(size):
        structure = training_structures[m_index + s]
        kern_curr = 0
        for environment in structure:
            kern_curr += kernel(x, environment, *args)

        energy_energy_unit[m_index] = kern_curr

    return energy_energy_unit


def energy_force_vector_unit(name, s, e, x, kernel, hyps, cutoffs=None,
                             hyps_mask=None, d_1=None):
    """
    Gets part of the energy/force vector.
    """

    training_data = _global_training_data[name]

    ds = [1, 2, 3]
    size = (e - s) * 3
    k_v = np.zeros(size, )

    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    for m_index in range(size):
        x_2 = training_data[int(math.floor(m_index / 3))+s]
        d_2 = ds[m_index % 3]
        k_v[m_index] = kernel(x_2, x, d_2, *args)

    return k_v


def force_energy_vector_unit(name, s, e, x, kernel, hyps, cutoffs, hyps_mask,
                             d_1):
    """
    Gets part of the force/energy vector.
    """

    size = e - s
    args = from_mask_to_args(hyps, cutoffs, hyps_mask)
    force_energy_unit = np.zeros(size,)

    for m_index in range(size):
        training_structure = _global_training_structures[name][m_index+s]
        kern_curr = 0
        for environment in training_structure:
            kern_curr += kernel(x, environment, d_1, *args)

        force_energy_unit[m_index] = kern_curr

    return force_energy_unit


def force_force_vector_unit(name, s, e, x, kernel, hyps, cutoffs, hyps_mask,
                            d_1):
    """
    Gets part of the force/force vector.
    """

    size = e - s
    ds = [1, 2, 3]

    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    k_v = np.zeros(size * 3)

    for m_index in range(size):
        x_2 = _global_training_data[name][m_index + s]
        for d_2 in ds:
            k_v[m_index * 3 + d_2 - 1] = kernel(x, x_2, d_1, d_2, *args)

    return k_v


def efs_force_vector_unit(name, s, e, x, efs_force_kernel, hyps, cutoffs,
                          hyps_mask):
    size = e - s

    k_ef = np.zeros((1, size * 3))
    k_ff = np.zeros((3, size * 3))
    k_sf = np.zeros((6, size * 3))

    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    for m_index in range(size):
        x_2 = _global_training_data[name][m_index + s]
        ef, ff, sf = efs_force_kernel(x, x_2, *args)

        ind1 = m_index * 3
        ind2 = (m_index + 1) * 3

        k_ef[:, ind1:ind2] = ef
        k_ff[:, ind1:ind2] = ff
        k_sf[:, ind1:ind2] = sf

    return k_ef, k_ff, k_sf


def efs_energy_vector_unit(name, s, e, x, efs_energy_kernel, hyps, cutoffs,
                           hyps_mask):

    size = e - s
    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

    k_ee = np.zeros((1, size))
    k_fe = np.zeros((3, size))
    k_se = np.zeros((6, size))

    for m_index in range(size):
        training_structure = _global_training_structures[name][m_index + s]

        ee_curr = 0
        fe_curr = np.zeros(3)
        se_curr = np.zeros(6)

        for environment in training_structure:
            ee, fe, se = efs_energy_kernel(x, environment, *args)
            ee_curr += ee
            fe_curr += fe
            se_curr += se

        k_ee[:, m_index] = ee_curr
        k_fe[:, m_index] = fe_curr
        k_se[:, m_index] = se_curr

    return k_ee, k_fe, k_se


def energy_energy_vector(name, kernel, x, hyps, cutoffs=None,
                         hyps_mask=None, n_cpus=1, n_sample=100):
    """
    Get a vector of covariances between the local energy of a test environment
    and the total energy labels in the training set.
    """

    size = len(_global_training_structures[name])

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return energy_energy_vector_unit(name, 0, size, x, kernel, hyps,
                                         cutoffs, hyps_mask)

    block_id, nbatch = partition_vector(n_sample, size, n_cpus)
    pack_function = energy_energy_vector_unit
    mult = 1

    force_energy_vector = \
        parallel_vector_construction(pack_function, name, x, kernel,
                                     hyps, cutoffs, hyps_mask, block_id,
                                     nbatch, size, mult)

    return force_energy_vector


def energy_force_vector(name, kernel, x, hyps, cutoffs=None, hyps_mask=None,
                        n_cpus=1, n_sample=100):
    """
    Get the vector of covariances between the local energy of a test
    environment and the force labels in the training set.
    """

    training_data = _global_training_data[name]
    size = len(training_data)

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return energy_force_vector_unit(name, 0, size, x, kernel, hyps,
                                        cutoffs, hyps_mask)

    block_id, nbatch = partition_vector(n_sample, size, n_cpus)
    pack_function = energy_force_vector_unit
    mult = 3

    k12_v = parallel_vector_construction(pack_function, name, x, kernel,
                                         hyps, cutoffs, hyps_mask, block_id,
                                         nbatch, size, mult)

    return k12_v


def force_energy_vector(name, kernel, x, d_1, hyps, cutoffs=None,
                        hyps_mask=None, n_cpus=1, n_sample=100):
    """
    Get a vector of covariances between a force component of a test environment
    and the total energy labels in the training set.
    """

    size = len(_global_training_structures[name])

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return force_energy_vector_unit(name, 0, size, x, kernel, hyps,
                                        cutoffs, hyps_mask, d_1)

    block_id, nbatch = partition_vector(n_sample, size, n_cpus)
    pack_function = force_energy_vector_unit
    mult = 1

    force_energy_vector = \
        parallel_vector_construction(pack_function, name, x, kernel,
                                     hyps, cutoffs, hyps_mask, block_id,
                                     nbatch, size, mult, d_1)

    return force_energy_vector


def force_force_vector(name, kernel, x, d_1, hyps, cutoffs=None,
                       hyps_mask=None, n_cpus=1, n_sample=100):
    """
    Get a vector of covariances between a force component of a test environment
    and the force labels in the training set.
    """

    size = len(_global_training_data[name])

    if (n_cpus is None):
        n_cpus = mp.cpu_count()
    if (n_cpus == 1):
        return force_force_vector_unit(name, 0, size, x, kernel, hyps, cutoffs,
                                       hyps_mask, d_1)

    block_id, nbatch = partition_vector(n_sample, size, n_cpus)
    pack_function = force_force_vector_unit
    mult = 3

    k12_v = parallel_vector_construction(pack_function, name, x, kernel,
                                         hyps, cutoffs, hyps_mask, block_id,
                                         nbatch, size, mult, d_1)

    return k12_v


def efs_force_vector(name, efs_force_kernel, x, hyps, cutoffs=None,
                     hyps_mask=None, n_cpus=1, n_sample=100):
    """
    Returns covariances between the local eneregy, force components, and
    partial stresses of a test environment and the force labels in the
    training set.
    """

    size = len(_global_training_data[name])

    if (n_cpus is None):
        n_cpus = mp.cpu_count()

    # Perform serial calculation if n_cpus = 1.
    if (n_cpus == 1):
        return efs_force_vector_unit(name, 0, size, x, efs_force_kernel, hyps,
                                     cutoffs, hyps_mask)

    # Otherwise, perform parallel calculation.
    block_id, nbatch = partition_vector(n_sample, size, n_cpus)
    pack_function = efs_force_vector_unit
    mult = 3
    n_comps = size * mult
    array_sizes = [(1, n_comps), (3, n_comps), (6, n_comps)]

    efs_arrays = \
        multiple_array_construction(pack_function, name, x, efs_force_kernel,
                                    hyps, cutoffs, hyps_mask, block_id, nbatch,
                                    size, mult, array_sizes)

    return efs_arrays


def efs_energy_vector(name, efs_energy_kernel, x, hyps, cutoffs=None,
                      hyps_mask=None, n_cpus=1, n_sample=100):
    """
    Returns covariances between the local eneregy, force components, and
    partial stresses of a test environment and the total energy labels in the
    training set.
    """

    size = len(_global_training_structures[name])

    if (n_cpus is None):
        n_cpus = mp.cpu_count()

    # Perform serial calculation if n_cpus = 1.
    if (n_cpus == 1):
        return efs_energy_vector_unit(name, 0, size, x, efs_energy_kernel,
                                      hyps, cutoffs, hyps_mask)

    # Otherwise, perform parallel calculation.
    block_id, nbatch = partition_vector(n_sample, size, n_cpus)
    pack_function = efs_energy_vector_unit
    mult = 1
    array_sizes = [(1, size), (3, size), (6, size)]

    efs_arrays = \
        multiple_array_construction(pack_function, name, x, efs_energy_kernel,
                                    hyps, cutoffs, hyps_mask, block_id, nbatch,
                                    size, mult, array_sizes)

    return efs_arrays


def get_kernel_vector(name, force_force_kernel: Callable,
                      force_energy_kernel: Callable, x, d_1, hyps,
                      cutoffs=None, hyps_mask=None, n_cpus=1, n_sample=100):

    size1 = len(_global_training_data[name])
    size2 = len(_global_training_structures[name])
    kernel_vector = np.zeros(size1 * 3 + size2)

    force_vector = \
        force_force_vector(name, force_force_kernel, x, d_1, hyps, cutoffs,
                           hyps_mask, n_cpus, n_sample)
    energy_vector = \
        force_energy_vector(name, force_energy_kernel, x, d_1, hyps, cutoffs,
                            hyps_mask, n_cpus, n_sample)

    kernel_vector[0:size1*3] = force_vector
    kernel_vector[size1*3:] = energy_vector

    return kernel_vector


def en_kern_vec(name, energy_force_kernel, energy_energy_kernel, x, hyps,
                cutoffs=None, hyps_mask=None, n_cpus=1, n_sample=100):

    size1 = len(_global_training_data[name])
    size2 = len(_global_training_structures[name])
    kernel_vector = np.zeros(size1 * 3 + size2)

    force_vector = \
        energy_force_vector(name, energy_force_kernel, x, hyps, cutoffs,
                            hyps_mask, n_cpus, n_sample)
    energy_vector = \
        energy_energy_vector(name, energy_energy_kernel, x, hyps, cutoffs,
                             hyps_mask, n_cpus, n_sample)

    kernel_vector[0:size1*3] = force_vector
    kernel_vector[size1*3:] = energy_vector

    return kernel_vector


def efs_kern_vec(name, efs_force_kernel, efs_energy_kernel, x, hyps,
                 cutoffs=None, hyps_mask=None, n_cpus=1, n_sample=100):

    size1 = len(_global_training_data[name])
    size2 = len(_global_training_structures[name])
    tot_size = size1 * 3 + size2

    # Initialize arrays.
    energy_vector = np.zeros(tot_size)
    force_array = np.zeros((3, tot_size))
    stress_array = np.zeros((6, tot_size))

    # Compute force and energy arrays.
    force_arrays = \
        efs_force_vector(name, efs_force_kernel, x, hyps, cutoffs,
                         hyps_mask, n_cpus, n_sample)

    energy_arrays = \
        efs_energy_vector(name, efs_energy_kernel, x, hyps, cutoffs,
                          hyps_mask, n_cpus, n_sample)

    # Populate arrays.
    energy_vector[0:size1 * 3] = force_arrays[0]
    energy_vector[size1 * 3:] = energy_arrays[0]
    force_array[:, 0:size1 * 3] = force_arrays[1]
    force_array[:, size1 * 3:] = energy_arrays[1]
    stress_array[:, 0:size1 * 3] = force_arrays[2]
    stress_array[:, size1 * 3:] = energy_arrays[2]

    return energy_vector, force_array, stress_array


# --------------------------------------------------------------------------
#                      Ky hyperparameter gradient
# --------------------------------------------------------------------------

def get_ky_and_hyp_pack(name, s1, e1, s2, e2, same: bool, hyps: np.ndarray,
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
    _, non_noise_hyps, _ = obtain_noise_len(hyps, hyps_mask)

    # initialize matrices
    size1 = (e1-s1) * 3
    size2 = (e2-s2) * 3
    k_mat = np.zeros([size1, size2])
    hyp_mat = np.zeros([non_noise_hyps, size1, size2])

    args = from_mask_to_args(hyps, cutoffs, hyps_mask)

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


def get_ky_and_hyp(hyps: np.ndarray, name, kernel_grad, cutoffs=None,
                   hyps_mask=None, n_cpus=1, n_sample=100):
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

        block_id, nbatch = partition_matrix(n_sample, size, n_cpus)

        result_queue = mp.Queue()
        children = []
        for wid in range(nbatch):
            s1, e1, s2, e2 = block_id[wid]
            children.append(
                mp.Process(
                    target=queue_wrapper,
                    args=(result_queue, wid, get_ky_and_hyp_pack,
                          (name, s1, e1, s2, e2, s1 == s2,
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
        hyp_mat = np.zeros([hyp_mat0.shape[0]+1, hyp_mat0.shape[1],
                            hyp_mat0.shape[2]])
        hyp_mat[:-1, :, :] = hyp_mat0
        hyp_mat[-1, :, :] = np.eye(size3) * 2 * sigma_n
    else:
        hyp_mat = hyp_mat0

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size3)

    return hyp_mat, ky_mat


# --------------------------------------------------------------------------
#                        Computing the likelihood
# --------------------------------------------------------------------------

def get_like_from_mats(ky_mat, l_mat, alpha, name):
    """ compute the likelihood from the covariance matrix

    :param ky_mat: the covariance matrix

    :return: float, likelihood
    """
    # catch linear algebra errors
    force_labels = _global_training_labels[name]
    energy_labels = _global_energy_labels[name]
    labels = np.concatenate((force_labels, energy_labels))

    # calculate likelihood
    like = (-0.5 * np.matmul(labels, alpha) -
            np.sum(np.log(np.diagonal(l_mat))) -
            math.log(2 * np.pi) * ky_mat.shape[1] / 2)

    return like


def get_neg_like_grad(hyps: np.ndarray, name: str,
                      kernel_grad, logger_name: str=None,
                      cutoffs=None, hyps_mask=None,
                      n_cpus=1, n_sample=100):
    """compute the log likelihood and its gradients

    :param hyps: list of hyper-parameters
    :type hyps: np.ndarray
    :param name: name of the gp instance.
    :param kernel_grad: function object of the kernel gradient
    :param logger_name: name of logger object for dumping every hyper-parameter
                   sets computed
    :type logger_name: str
    :param cutoffs: The cutoff values used for the atomic environments
    :type cutoffs: list of 2 float numbers
    :param hyps_mask: dictionary used for multi-group hyperparmeters
    :param n_cpus: number of cpus to use.
    :param n_sample: the size of block for matrix to compute

    :return: float, np.array
    """

    time0 = time.time()

    hyp_mat, ky_mat = \
        get_ky_and_hyp(hyps, name, kernel_grad, cutoffs=cutoffs,
                       hyps_mask=hyps_mask, n_cpus=n_cpus, n_sample=n_sample)

    logger = logging.getLogger(logger_name)
    logger.debug(f"get_ky_and_hyp {time.time()-time0}")

    time0 = time.time()

    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, name)

    logger.debug(f"get_like_grad_from_mats {time.time()-time0}")

    logger.debug('')
    logger.info(f'Hyperparameters: {list(hyps)}')
    logger.info(f'Likelihood: {like}')
    logger.info(f'Likelihood Gradient: {list(like_grad)}')

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
    except np.linalg.LinAlgError:
        return -1e8, np.zeros(number_of_hyps)

    labels = _global_training_labels[name]

    alpha = np.matmul(ky_mat_inv, labels)
    alpha_mat = np.matmul(alpha.reshape(-1, 1), alpha.reshape(1, -1))
    like_mat = alpha_mat - ky_mat_inv

    # calculate likelihood
    like = (-0.5 * np.matmul(labels, alpha) -
            np.sum(np.log(np.diagonal(l_mat))) -
            math.log(2 * np.pi) * ky_mat.shape[1] / 2)

    # calculate likelihood gradient
    like_grad = np.zeros(number_of_hyps)
    for n in range(number_of_hyps):
        like_grad[n] = 0.5 * np.trace(np.matmul(like_mat, hyp_mat[n, :, :]))

    return like, like_grad
