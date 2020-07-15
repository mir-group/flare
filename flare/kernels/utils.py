import numpy as np

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


def str_to_kernel_set(kernels: list = ['twobody', 'threebody'],
                      component: str = "mc",
                      hyps_mask: dict = None):
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
    if component == 'sc':
        stk = sc._str_to_kernel
    else:
        multihyps = True
        if hyps_mask is None:
            multihyps = False
        elif hyps_mask['nspecie'] == 1:
            multihyps = False
        if multihyps:
            stk = mc_sephyps._str_to_kernel
        else:
            stk = mc_simple._str_to_kernel

    # b2 = Two body in use, b3 = Three body in use
    str_terms = {'2': ['2', 'two', 'twobody'],
                 '3': ['3', 'three', 'threebody'],
                 'many': ['mb', 'manybody', 'many']}

    if isinstance(kernels, str):
        kernels = [kernels]

    prefix = ''
    for term in str_terms:
        add = False
        for s in str_terms[term]:
            for k in kernels:
                if s in k.lower():
                    add = True
        if add:
            if len(prefix) > 0:
                prefix += '+'
            prefix += term

    if len(prefix) == 0:
        raise RuntimeError(
            f"the name has to include at least one number {kernels}")

    for suffix in ['', '_grad', '_en', '_force_en', '_efs_energy',
                   '_efs_force', '_efs_self']:
        if prefix+suffix not in stk:
            raise RuntimeError(
                f"cannot find kernel function of {prefix}{suffix}")

    return stk[prefix], stk[prefix + '_grad'], stk[prefix + '_en'], \
        stk[prefix + '_force_en'], stk[prefix+'_efs_energy'], \
        stk[prefix + '_efs_force'], stk[prefix + '_efs_self']



def from_mask_to_args(hyps, cutoffs, hyps_mask=None):
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

    # no special setting
    multihyps = True
    if hyps_mask is None:
        multihyps = False
    elif hyps_mask['nspecie'] == 1:
        multihyps = False

    if not multihyps:

        cutoffs_array = [0, 0, 0]
        cutoffs_array[0] = cutoffs.get('twobody', 0)
        cutoffs_array[1] = cutoffs.get('threebody', 0)
        cutoffs_array[2] = cutoffs.get('manybody', 0)
        return (hyps, cutoffs_array)

    # setting for mc_sephyps
    nspecie = hyps_mask['nspecie']
    n2b = hyps_mask.get('ntwobody', 0)

    n3b = hyps_mask.get('nthreebody', 0)
    nmanybody = hyps_mask.get('nmanybody', 0)
    ncut3b = hyps_mask.get('ncut3b', 0)

    twobody_mask = hyps_mask.get('twobody_mask', None)
    threebody_mask = hyps_mask.get('threebody_mask', None)
    manybody_mask = hyps_mask.get('manybody_mask', None)
    cut3b_mask = hyps_mask.get('cut3b_mask', None)

    # TO DO , should instead use the non-sephyps kernel
    if (n2b == 1):
        twobody_mask = np.zeros(nspecie**2, dtype=int)
    if (n3b == 1):
        threebody_mask = np.zeros(nspecie**3, dtype=int)
    if (nmanybody == 1):
        manybody_mask = np.zeros(nspecie**2, dtype=int)

    cutoff_2b = cutoffs.get('twobody', 0)
    cutoff_3b = cutoffs.get('threebody', 0)
    cutoff_mb = cutoffs.get('manybody', 0)

    if 'bond_cutoff_list' in hyps_mask:
        cutoff_2b = hyps_mask['bond_cutoff_list']
    else:
        cutoff_2b = np.ones(nspecie**2, dtype=float)*cutoff_2b

    if 'threebody_cutoff_list' in hyps_mask:
        cutoff_3b = hyps_mask['threebody_cutoff_list']
    if 'manybody_cutoff_list' in hyps_mask:
        cutoff_mb = hyps_mask['manybody_cutoff_list']

    (sig2, ls2) = Parameters.get_component_hyps(hyps_mask, 'twobody', hyps=hyps)
    (sig3, ls3) = Parameters.get_component_hyps(
        hyps_mask, 'threebody', hyps=hyps)
    (sigm, lsm) = Parameters.get_component_hyps(
        hyps_mask, 'manybody', hyps=hyps)

    return (cutoff_2b, cutoff_3b, cutoff_mb,
            nspecie,
            np.array(hyps_mask['specie_mask']),
            n2b, twobody_mask,
            n3b, threebody_mask,
            ncut3b, cut3b_mask,
            nmanybody, manybody_mask,
            sig2, ls2, sig3, ls3, sigm, lsm)


def from_grad_to_mask(grad, hyps_mask=None):
    """
    Return gradient which only includes hyperparameters
    which are meant to vary

    :param grad: original gradient vector
    :param hyps_mask: dictionary for hyper-parameters

    :return: newgrad
    """

    constrain = True
    if hyps_mask is None:
        constrain = False
    elif 'map' not in hyps_mask:
        constrain = False
    if not constrain:
        return grad

    hyp_index = hyps_mask['map']

    # setting for mc_sephyps
    # if the last element is not sigma_noise
    if hyp_index[-1] == len(grad):
        hm = hyp_index[:-1]
    else:
        hm = hyp_index

    newgrad = np.zeros(len(hm), dtype=np.float64)
    for i, mapid in enumerate(hm):
        newgrad[i] = grad[mapid]

    return newgrad


def kernel_str_to_array(kernel_name: str):
    """
    Args:

    name (str): name for kernels. example: "2+3mc"

    :return: kernel function, kernel gradient, energy kernel,
             energy_and_force kernel
    """

    #  kernel name should be replace with kernel array
    str_terms = {'twobody': ['2', 'two', 'twobody'],
                 'threebody': ['3', 'three', 'threebody'],
                 'manybody': ['mb', 'manybody', 'many']}

    array = []
    for term in str_terms:
        add = False
        for s in str_terms[term]:
            if s in kernel_name.lower():
                add = True
        if add:
            array += [term]
    return array
