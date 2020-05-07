import numpy as np

from flare.kernels import sc, mc_simple, mc_sephyps

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

    if 'sc' in name:
        stk = sc._str_to_kernel
    else:
        if (multihyps is False):
            stk = mc_simple._str_to_kernel
        else:
            stk = mc_sephyps._str_to_kernel

    # b2 = Two body in use, b3 = Three body in use
    b2 = False
    b3 = False
    many = False

    for s in ['2', 'two', 'Two', 'TWO']:
        if (s in name):
            b2 = True
    for s in ['3', 'three', 'Three', 'THREE']:
        if (s in name):
            b3 = True
    for s in ['mb', 'manybody', 'many', 'Many', 'ManyBody']:
        if (s in name):
            many = True

    prefix=''
    str_term={'2':b2, '3':b3, 'many':many}
    for term in str_term:
        if str_term[term]:
            if (len(prefix)>0):
                prefix += '+'
            prefix += term
    if len(prefix)==0:
        raise RuntimeError(f"the name has to include at least one number {name}")

    for suffix in ['', '_grad', '_en', '_force_en']:
        if prefix+suffix not in stk:
            raise RuntimeError(f"cannot find kernel function of {prefix}{suffix}")

    return stk[prefix], stk[prefix+'_grad'], stk[prefix+'_en'], \
            stk[prefix+'_force_en']


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
    sigm = None
    lsm = None

    if ('map' in hyps_mask.keys()):
        orig_hyps = hyps_mask['original']
        hm = hyps_mask['map']
        for i, h in enumerate(hyps):
            orig_hyps[hm[i]] = h
    else:
        orig_hyps = hyps

    if (len(cutoffs)<=2):
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

    elif (len(cutoffs)==3):
        if (n2b != 0):
            sig2 = orig_hyps[:n2b]
            ls2 = orig_hyps[n2b:n2b * 2]
        if (n3b !=0):
            sig3 = orig_hyps[n2b * 2:n2b * 2 + n3b]
            ls3 = orig_hyps[n2b * 2 + n3b:n2b * 2 + n3b * 2]
        sigm = orig_hyps[n2b*2+n3b*2]
        lsm = orig_hyps[n2b*2+n3b*2+1]

        return (np.array(cutoffs), hyps_mask['nspec'], np.array(hyps_mask['spec_mask']),
                n2b, np.array(bond_mask), n3b, np.array(triplet_mask),
                np.array(sig2), np.array(ls2), np.array(sig3), np.array(ls3), sigm, lsm)
    else:
        raise RuntimeError("only support up to 3 cutoffs")


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

