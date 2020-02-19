import numpy as np

from flare.kernels.kernels import str_to_kernel
from flare.kernels.mc_simple import str_to_mc_kernel
from flare.kernels.mc_sephyps import str_to_mc_kernel as str_to_mc_sephyps_kernel

"""
This module includes interface functions between kernels and gp/gp_algebra

str_to_kernel_set is used in GaussianProcess class to search for a kernel function
based on a string name.

from_mask_to_args converts the hyperparameter vector and the dictionary of hyps_mask
to the list of arguments needed by the kernel function.

from_grad_to_mask(grad, hyps_mask) converts the gradient matrix to the actual gradient
matrix by removing the fixed dimensions.
"""

def str_to_kernel_set(name: str, multihyps: bool =False):
    """
    return kernels and kernel gradient function base on a string

    Args:

    name (str): name for kernels. example: "2+3mc"
    multihyps (bool): True for using multiple hyperparameter groups

    :return: kernel function, kernel gradient, energy kernel,
             energy_and_force kernel

    """

    if 'mc' in name:
        if (multihyps is False):
            stk = str_to_mc_kernel
        else:
            stk = str_to_mc_sephyps_kernel
    else:
        stk = str_to_kernel

    # b2 = Two body in use, b3 = Three body in use
    b2 = False
    b3 = False

    for s in ['2', 'two']:
        if (s in name):
            b2 = True
    for s in ['3', 'three']:
        if (s in name):
            b3 = True
    if (b2 and b3):
        prefix='2+3'
    elif (b2):
        prefix='2'
    elif (b3):
        prefix='3'
    else:
        raise RuntimeError(f"the name has to include at least one number {name}")

    return stk(prefix), stk(prefix+'_grad'), stk(prefix+'_en'), \
            stk(prefix+'_force_en')


def from_mask_to_args(hyps, hyps_mask: dict, cutoffs):
    """ return the tuple of arguments needed for kernel function
    the order of the tuple has to be exactly the same as the one
    taken by the kernel function

    :param hyps: list of hyperparmeter values
    :type hyps: nd.array
    :param hyps_mask: all the remaining parameters needed
    :type hyps_mask: dictionary
    :param cutoffs: cutoffs used

    :return: args
    """

    # no special setting
    if (hyps_mask is None):
        return (hyps, cutoffs)

    # setting for mc_sephyps
    n2b = hyps_mask.get('nbond', 0)
    n3b = hyps_mask.get('ntriplet', 0)
    triplet_mask = hyps_mask.get('triplet_mask', None)
    bond_mask = hyps_mask.get('bond_mask', None)
    sig2 = None
    ls2 = None
    sig3 = None
    ls3 = None

    if ('map' in hyps_mask.keys()):
        orig_hyps = hyps_mask['original']
        hm = hyps_mask['map']
        for i, h in enumerate(hyps):
            orig_hyps[hm[i]] = h
    else:
        orig_hyps = hyps

    if (n2b != 0):
        sig2 = orig_hyps[:n2b]
        ls2 = orig_hyps[n2b:n2b * 2]
    if (n3b !=0):
        sig3 = orig_hyps[n2b * 2:n2b * 2 + n3b]
        ls3 = orig_hyps[n2b * 2 + n3b:n2b * 2 + n3b * 2]
    if (n2b == 0) and (n3b == 0):
        raise NameError("Hyperparameter mask missing nbond and/or"
                        "ntriplet key")

    return (np.array(cutoffs), hyps_mask['nspec'], np.array(hyps_mask['spec_mask']),
            n2b, np.array(bond_mask), n3b, np.array(triplet_mask),
            np.array(sig2), np.array(ls2), np.array(sig3), np.array(ls3))


def from_grad_to_mask(grad, hyps_mask):
    """
    Return gradient which only includes hyperparameters
    which are meant to vary

    :param grad: original gradient vector
    :param hyps_mask: dictionary for hyper-parameters

    :return: newgrad
    """

    # no special setting
    if (hyps_mask is None):
        return grad

    # setting for mc_sephyps
    # no constrained optimization
    if 'map' not in hyps_mask.keys():
        return grad

    # setting for mc_sephyps
    # if the last element is not sigma_noise
    if (hyps_mask['map'][-1] == len(grad)):
        hm = hyps_mask['map'][:-1]
    else:
        hm = hyps_mask['map']

    newgrad = np.zeros(len(hm))
    for i, mapid in enumerate(hm):
        newgrad[i] = grad[mapid]
    return newgrad

