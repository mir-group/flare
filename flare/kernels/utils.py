import numpy as np

import flare.kernels.map_3b_kernel as map_3b

from flare.kernels import sc, mc_simple, mc_sephyps
from flare.parameters import Parameters

"""
This module includes interface functions between kernels and gp/gp_algebra

str_to_kernel_set is used in GaussianProcess class to search for a kernel
    function based on a string name.

from_mask_to_args converts the hyperparameter vector and the dictionary of
    hyps_mask to the list of arguments needed by the kernel function.

from_grad_to_mask(grad, hyps_mask) converts the gradient matrix to the actual
    gradient matrix by removing the fixed dimensions.
"""


def str_to_kernel_set(name: str, hyps_mask: dict = None):
    """
    return kernels and kernel gradient function base on a string.
    If it contains 'sc', it will use the kernel in sc module;
    otherwise, it uses the kernel in mc_simple;
    if sc is not included and multihyps is True,
    it will use the kernel in mc_sephyps module
    otherwise, it will use the kernel in the sc module

    Args:

    name (str): name for kernels. example: "2+3mc"
    multihyps (bool, optional): True for using multiple hyperparameter groups

    :return: kernel function, kernel gradient, energy kernel,
             energy_and_force kernel

    """

    #  kernel name should be replace with kernel array

    if 'sc' in name:
        stk = sc._str_to_kernel
    else:
        if hyps_mask is None:
            stk = mc_simple._str_to_kernel
        else:
            # In the future, this should be nbond >1, use sephyps bond...
            if hyps_mask['nspecie'] > 1:
                stk = mc_sephyps._str_to_kernel
            else:
                stk = mc_simple._str_to_kernel

    # b2 = Two body in use, b3 = Three body in use
    b2 = False
    b3 = False
    many = False

    for s in ['2', 'two', 'Two', 'TWO']:
        if s in name:
            b2 = True
    for s in ['3', 'three', 'Three', 'THREE']:
        if s in name:
            b3 = True
    for s in ['mb', 'manybody', 'many', 'Many', 'ManyBody']:
        if s in name:
            many = True

    prefix = ''
    str_term = {'2': b2, '3': b3, 'many': many}
    for term in str_term:
        if str_term[term]:
            if len(prefix) > 0:
                prefix += '+'
            prefix += term
    if len(prefix) == 0:
        raise RuntimeError(
            f"the name has to include at least one number {name}")

    for suffix in ['', '_grad', '_en', '_force_en']:
        if prefix+suffix not in stk:
            raise RuntimeError(
                f"cannot find kernel function of {prefix}{suffix}")

    return stk[prefix], stk[prefix+'_grad'], stk[prefix+'_en'], \
        stk[prefix+'_force_en']


def str_to_mapped_kernel(name: str, hyps_mask: dict = None, energy=False):
    """
    return kernels and kernel gradient function base on a string.
    If it contains 'sc', it will use the kernel in sc module;
    otherwise, it uses the kernel in mc_simple;
    if sc is not included and multihyps is True,
    it will use the kernel in mc_sephyps module
    otherwise, it will use the kernel in the sc module

    Args:

    name (str): name for kernels. example: "2+3mc"
    multihyps (bool, optional): True for using multiple hyperparameter groups
    energy (bool, optional): True for mapping energy/energy kernel

    :return: mapped kernel function, kernel gradient, energy kernel,
             energy_and_force kernel

    """

    if 'sc' in name:
        raise NotImplementedError("mapped kernel for single component "
                                  "is not implemented")

    if hyps_mask is None:
        multihyps = False
    else:
        # In the future, this should be nbond >1, use sephyps bond...
        if hyps_mask['ntriplet'] > 1:
            multihyps = True
        else:
            multihyps = False

    # b2 = Two body in use, b3 = Three body in use
    b2 = False
    many = False
    b3 = False
    for s in ['3', 'three', 'Three', 'THREE']:
        if s in name:
            b3 = True

    if b3 == True and energy == False:
        if multihyps:
            tbmfe = map_3b.three_body_mc_en_force_sephyps
            tbme = map_3b.three_body_mc_en_sephyps
        else:
            tbmfe = map_3b.three_body_mc_en_force
            tbme = map_3b.three_body_mc_en
    else:
        raise NotImplementedError("mapped kernel for two-body and manybody kernels "
                                  "are not implemented")

    return tbmfe, tbme


def from_mask_to_args(hyps, hyps_mask: dict, cutoffs):
    """ Return the tuple of arguments needed for kernel function.
    The order of the tuple has to be exactly the same as the one taken by
        the kernel function.

    :param hyps: list of hyperparmeter values
    :type hyps: nd.array
    :param hyps_mask: all the remaining parameters needed
    :type hyps_mask: dictionary
    :param cutoffs: cutoffs used

    :return: args
    """

    cutoffs_array = [0, 0, 0]
    cutoffs_array[0] = cutoffs.get('bond', 0)
    cutoffs_array[1] = cutoffs.get('triplet', 0)
    cutoffs_array[2] = cutoffs.get('mb', 0)

    # no special setting
    if (hyps_mask is None):
        return (hyps, cutoffs_array)

    # setting for mc_sephyps
    nspecie = hyps_mask['nspecie']
    n2b = hyps_mask.get('nbond', 0)

    n3b = hyps_mask.get('ntriplet', 0)
    nmb = hyps_mask.get('nmb', 0)
    ncut3b = hyps_mask.get('ncut3b', 0)

    bond_mask = hyps_mask.get('bond_mask', None)
    triplet_mask = hyps_mask.get('triplet_mask', None)
    mb_mask = hyps_mask.get('mb_mask', None)
    cut3b_mask = hyps_mask.get('cut3b_mask', None)

    # TO DO , should instead use the non-sephyps kernel
    if (n2b == 1):
        bond_mask = np.zeros(nspecie**2, dtype=int)
    if (n3b == 1):
        triplet_mask = np.zeros(nspecie**3, dtype=int)
    if (nmb == 1):
        mb_mask = np.zeros(nspecie**2, dtype=int)

    cutoff_2b = cutoffs.get('bond', 0)
    cutoff_3b = cutoffs.get('triplet', 0)
    cutoff_mb = cutoffs.get('mb', 0)

    if 'bond_cutoff_list' in hyps_mask:
        cutoff_2b = hyps_mask['bond_cutoff_list']
    else:
        cutoff_2b = np.ones(nspecie**2, dtype=float)*cutoff_2b

    if 'triplet_cutoff_list' in hyps_mask:
        cutoff_3b = hyps_mask['triplet_cutoff_list']
    if 'mb_cutoff_list' in hyps_mask:
        cutoff_mb = hyps_mask['mb_cutoff_list']

    (sig2, ls2) = Parameters.get_component_hyps(hyps_mask, 'bond', hyps=hyps)
    (sig3, ls3) = Parameters.get_component_hyps(hyps_mask, 'triplet', hyps=hyps)
    (sigm, lsm) = Parameters.get_component_hyps(hyps_mask, 'mb', hyps=hyps)

    return (cutoff_2b, cutoff_3b, cutoff_mb,
            nspecie,
            np.array(hyps_mask['specie_mask']),
            n2b, bond_mask,
            n3b, triplet_mask,
            ncut3b, cut3b_mask,
            nmb, mb_mask,
            sig2, ls2, sig3, ls3, sigm, lsm)


def from_grad_to_mask(grad, hyps_mask):
    """
    Return gradient which only includes hyperparameters
    which are meant to vary

    :param grad: original gradient vector
    :param hyps_mask: dictionary for hyper-parameters

    :return: newgrad
    """

    # no special setting
    if hyps_mask is None:
        return grad

    # setting for mc_sephyps
    # no constrained optimization
    if 'map' not in hyps_mask:
        return grad

    # setting for mc_sephyps
    # if the last element is not sigma_noise
    if hyps_mask['map'][-1] == len(grad):
        hm = hyps_mask['map'][:-1]
    else:
        hm = hyps_mask['map']

    newgrad = np.zeros(len(hm), dtype=np.float64)
    for i, mapid in enumerate(hm):
        newgrad[i] = grad[mapid]
    return newgrad
