import numpy as np

def from_mask_to_args(hyps, hyps_mask: dict, cutoffs):
    """
    :param hyps:
    :param hyps_mask:
    :return:
    """

    if (hyps_mask is None):
        return (hyps, cutoffs)

    n2b = hyps_mask.get('nbond', 0)
    n3b = hyps_mask.get('ntriplet', 0)
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
            np.array(n2b, hyps_mask['bond_mask']), n3b, np.array(hyps_mask['triplet_mask']),
            np.array(sig2), np.array(ls2), np.array(sig3), np.array(ls3))

def from_mask_to_hyps(hyps, hyps_mask: dict = {}):
    """
    :param hyps:
    :param hyps_mask:
    :return:
    """

    n2b = hyps_mask.get('nbond', 0)
    n3b = hyps_mask.get('ntriplet', 0)
    if ('map' in hyps_mask.keys()):
        orig_hyps = hyps_mask['original']
        hm = hyps_mask['map']
        for i, h in enumerate(hyps):
            orig_hyps[hm[i]] = h
    else:
        orig_hyps = hyps

    if (n2b != 0) and (n3b != 0):
        sig2 = orig_hyps[:n2b]
        ls2 = orig_hyps[n2b:n2b * 2]
        sig3 = orig_hyps[n2b * 2:n2b * 2 + n3b]
        ls3 = orig_hyps[n2b * 2 + n3b:n2b * 2 + n3b * 2]
        return n2b, n3b, sig2, ls2, sig3, ls3

    elif (n2b == 0) and (n3b != 0):
        sig = orig_hyps[:n3b]
        ls = orig_hyps[n3b:n3b * 2]
        return 0, n3b, None, None, sig, ls

    elif (n2b != 0) and (n3b == 0):
        sig = orig_hyps[:n2b]
        ls = orig_hyps[n2b:n2b * 2]
        return n2b, 0, sig, ls, None, None

    elif (n2b == 0) and (n3b == 0):
        raise NameError("Hyperparameter mask missing nbond and/or"
                        "ntriplet key")


def from_grad_to_mask(grad, hyps_mask):
    """
    Return gradient which only includes hyperparameters
    which are meant to vary
    :param grad:
    :param hyps_mask:
    :return:
    """
    if (hyps_mask is None):
        return grad

    if 'map' not in hyps_mask.keys():
        return grad

    # if the last element is not sigma_noise
    if (hyps_mask['map'][-1] == len(grad)):
        hm = hyps_mask['map'][:-1]
    else:
        hm = hyps_mask['map']

    newgrad = np.zeros(len(hm))
    for i, mapid in enumerate(hm):
        newgrad[i] = grad[mapid]
    return newgrad

