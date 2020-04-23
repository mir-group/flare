import time
import math
import pickle
import inspect
import json

import numpy as np
from numpy.random import random
from numpy import array as nparray
from numpy import max as npmax
from typing import List, Callable, Union


class ParameterMasking():

    def __init__(self, hyps_mask=None):
        pass

    @staticmethod
    def check_instantiation(hyps_mask):
        """
        Runs a series of checks to ensure that the user has not supplied
        contradictory arguments which will result in undefined behavior
        with multiple hyperparameters.
        :return:
        """

        assert isinstance(hyps_mask, dict)

        assert 'nspec' in hyps_mask, "nspec key missing in " \
                                                 "hyps_mask dictionary"
        assert 'spec_mask' in hyps_mask, "spec_mask key " \
                                                     "missing " \
                                                     "in hyps_mask dicticnary"

        nspec = hyps_mask['nspec']
        hyps_mask['spec_mask'] = nparray(hyps_mask['spec_mask'], dtype=int)

        if 'nbond' in hyps_mask:
            n2b = hyps_mask['nbond']
            assert n2b>0
            assert isinstance(n2b, int)
            hyps_mask['bond_mask'] = nparray(hyps_mask['bond_mask'], dtype=int)
            if n2b > 0:
                bmask = hyps_mask['bond_mask']
                assert (npmax(bmask) < n2b)
                assert len(bmask) == nspec ** 2, \
                    f"wrong dimension of bond_mask: " \
                    f" {len(bmask)} != nspec^2 {nspec**2}"
                for t2b in range(nspec):
                    for t2b_2 in range(t2b, nspec):
                        assert bmask[t2b*nspec+t2b_2] == bmask[t2b_2*nspec+t2b], \
                                'bond_mask has to be symmetric'
        else:
            n2b = 0

        if 'ntriplet' in hyps_mask:
            n3b = hyps_mask['ntriplet']
            assert n3b>0
            assert isinstance(n3b, int)
            hyps_mask['triplet_mask'] = nparray(hyps_mask['triplet_mask'], dtype=int)
            if n3b > 0:
                tmask = hyps_mask['triplet_mask']
                assert (npmax(tmask) < n3b)
                assert len(tmask) == nspec ** 3, \
                    f"wrong dimension of bond_mask: " \
                    f" {len(tmask)} != nspec^3 {nspec**3}"

                for t3b in range(nspec):
                    for t3b_2 in range(t3b, nspec):
                        for t3b_3 in range(t3b_2, nspec):
                            assert tmask[t3b*nspec*nspec+t3b_2*nspec+t3b_3] \
                                    == tmask[t3b*nspec*nspec+t3b_3*nspec+t3b_2], \
                                    'bond_mask has to be symmetric'
                            assert tmask[t3b*nspec*nspec+t3b_2*nspec+t3b_3] \
                                    == tmask[t3b_2*nspec*nspec+t3b*nspec+t3b_3], \
                                    'bond_mask has to be symmetric'
                            assert tmask[t3b*nspec*nspec+t3b_2*nspec+t3b_3] \
                                    == tmask[t3b_2*nspec*nspec+t3b_3*nspec+t3b], \
                                    'bond_mask has to be symmetric'
                            assert tmask[t3b*nspec*nspec+t3b_2*nspec+t3b_3] \
                                    == tmask[t3b_3*nspec*nspec+t3b*nspec+t3b_2], \
                                    'bond_mask has to be symmetric'
                            assert tmask[t3b*nspec*nspec+t3b_2*nspec+t3b_3] \
                                    == tmask[t3b_3*nspec*nspec+t3b_2*nspec+t3b], \
                                    'bond_mask has to be symmetric'
        else:
            n3b = 0

        if 'nmb' in hyps_mask:
            nmb = hyps_mask['nmb']
            assert nmb>0
            assert isinstance(nmb, int)
            hyps_mask['mb_mask'] = nparray(hyps_mask['mb_mask'], dtype=int)
            if nmb > 0:
                bmask = hyps_mask['mb_mask']
                assert (npmax(bmask) < nmb)
                assert len(bmask) == nspec ** 2, \
                    f"wrong dimension of mb_mask: " \
                    f" {len(bmask)} != nspec^2 {nspec**2}"
                for tmb in range(nspec):
                    for tmb_2 in range(tmb, nspec):
                        assert bmask[tmb*nspec+tmb_2] == bmask[tmb_2*nspec+tmb], \
                                'mb_mask has to be symmetric'
        else:
            nmb = 1
            hyps_mask['mb_mask'] = np.zeros(nspec**2)

        if 'map' in hyps_mask:
            assert ('original' in hyps_mask), \
                "original hyper parameters have to be defined"
            # Ensure typed correctly as numpy array
            hyps_mask['original'] = nparray(hyps_mask['original'], dtype=np.float)

            if (len(hyps_mask['original']) - 1) not in hyps_mask['map']:
                assert hyps_mask['train_noise'] is False, \
                    "train_noise should be False when noise is not in hyps"
        else:
            assert hyps_mask['train_noise'] is True, \
                "train_noise should be True when map is not used"

        if 'cutoff_2b' in hyps_mask:
            c2b = hyps_mask['cutoff_2b']
            assert len(c2b) == n2b, \
                    f'number of 2b cutoff should be the same as n2b {n2b}'

        if 'cutoff_3b' in hyps_mask:
            c3b = hyps_mask['cutoff_3b']
            assert nc3b>0
            assert isinstance(nc3b, int)
            hyps_mask['cut3b_mask'] = nparray(hyps_mask['cut3b_mask'], dtype=int)
            assert len(c3b) == hyps_mask['ncut3b'], \
                    f'number of 3b cutoff should be the same as ncut3b {ncut3b}'
            assert len(hyps_mask['cut3b_mask']) == nspec ** 2, \
                f"wrong dimension of cut3b_mask: " \
                f" {len(bmask)} != nspec^2 {nspec**2}"
            assert npmax(hyps_mask['cut3b_mask']) < hyps_mask['ncut3b'], \
                f"wrong dimension of cut3b_mask: " \
                f" {len(bmask)} != nspec^2 {nspec**2}"

        if 'cutoff_mb' in hyps_mask:
            cmb = hyps_mask['cutoff_mb']
            assert len(cmb) == nmb, \
                    f'number of mb cutoff should be the same as nmb {nmb}'
        return hyps_mask

    @staticmethod
    def check_matching(hyps_mask, hyps, cutoffs):

        n2b = hyps_mask.get('nbond', 0)
        n3b = hyps_mask.get('ntriplet', 0)
        nmb = hyps_mask.get('nmb', 1)

        if (len(cutoffs)<=2):
            assert ((n2b + n3b) > 0)
        else:
            assert ((n2b + n3b + nmb) > 0)

        if 'map' in hyps_mask:
            if (len(cutoffs)<=2):
                assert (n2b * 2 + n3b * 2 + 1) == len(hyps_mask['original']), \
                    "the hyperparmeter length is inconsistent with the mask"
            else:
                assert (n2b * 2 + n3b * 2 + nmb * 2 + 1) == len(hyps_mask['original']), \
                    "the hyperparmeter length is inconsistent with the mask"
            assert len(hyps_mask['map']) == len(hyps), \
                "the hyperparmeter length is inconsistent with the mask"
        else:
            if (len(cutoffs)<=2):
                assert (n2b * 2 + n3b * 2 + 1) == len(hyps), \
                    "the hyperparmeter length is inconsistent with the mask"
            else:
                assert (n2b * 2 + n3b * 2 + nmb*2 + 1) == len(hyps), \
                    "the hyperparmeter length is inconsistent with the mask"

        if 'cutoff_2b' in hyps_mask:
            assert cutoffs[0] > npmax(hyps_mask['cutoff_2b']), \
                    'general cutoff should be larger than all cutoffs listed in hyps_mask'

        if 'cutoff_3b' in hyps_mask:
            assert cutoffs[0] > npmax(hyps_mask['cutoff_3b']), \
                    'general cutoff should be larger than all cutoffs listed in hyps_mask'

        if 'cutoff_mb' in hyps_mask:
            assert cutoffs[0] > npmax(hyps_mask['cutoff_mb']), \
                    'general cutoff should be larger than all cutoffs listed in hyps_mask'


    def __str__(self):
        """String representation of the GP model."""
        pass


    def as_dict(self):
        """Dictionary representation of the GP model."""
        pass
