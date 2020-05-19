import time
import math
import pickle
import inspect
import json
import logging

import numpy as np
from copy import deepcopy
from numpy.random import random
from numpy import array as nparray
from numpy import max as npmax
from typing import List, Callable, Union
from warnings import warn
from sys import stdout
from os import devnull

from flare.utils.element_coder import element_to_Z, Z_to_element


class Parameters():

    def __init__(self, hyps_mask=None, species=None, kernels={},
                 cutoff_group={}, parameters=None,
                 constraints={}, allseparate=False, random=False, verbose=False):

        self.nspecies = 0

        self.n = {}
        for kernel_type in ['bonds', 'triplets', 'many2b', 'cut3b']:
            self.n[kernel_type] = 0

        self.mask = {}
        for mask_type in ['bond_mask', 'triplet_mask', 'cut3b_mask', 'mb_mask']:
            self.mask[mask_type] = None


        hyps = []
        hyps_label = []
        opt = []
        for group in ['bond', 'triplet', 'mb']:
            if (self.n[group] >= 1):
                # copy the mask
                hyps_mask['n'+group] = self.n[group]
                hyps_mask[group+'_mask'] = self.mask[group]
                hyps += [self.hyps_sig[group]]
                hyps += [self.hyps_ls[group]]
                # check parameters
                opt += [self.hyps_opt[group]]
                aeg = self.all_group_names[group]
                for idt in range(self.n[group]):
                    hyps_label += ['Signal_Var._'+aeg[idt]]
                for idt in range(self.n[group]):
                    hyps_label += ['Length_Scale_'+group]
        opt += [self.opt['noise']]
        hyps_label += ['Noise_Var.']
        hyps_mask['hyps_label'] = hyps_label
        hyps += [self.noise]

        # handle partial optimization if any constraints are defined
        hyps_mask['original'] = np.hstack(hyps)

        opt = np.hstack(opt)
        hyps_mask['train_noise'] = self.opt['noise']
        if (not opt.all()):
            nhyps = len(hyps_mask['original'])
            hyps_mask['original_labels'] = hyps_mask['hyps_label']
            mapping = []
            hyps_mask['hyps_label'] = []
            for i in range(nhyps):
                if (opt[i]):
                    mapping += [i]
                    hyps_mask['hyps_label'] += [hyps_label[i]]
            newhyps = hyps_mask['original'][mapping]
            hyps_mask['map'] = np.array(mapping, dtype=np.int)
        elif (opt.any()):
            newhyps = hyps_mask['original']
        else:
            raise RuntimeError("hyps has length zero."
                               "at least one component of the hyper-parameters"
                               "should be allowed to be optimized. \n")
        hyps_mask['hyps'] = newhyps

        # checkout universal cutoffs and seperate cutoffs
        nbond = hyps_mask.get('nbond', 0)
        ntriplet = hyps_mask.get('ntriplet', 0)
        nmb = hyps_mask.get('nmb', 0)
        if len(self.cutoff_list.get('bond', [])) > 0 \
                and nbond > 0:
            hyps_mask['cutoff_2b'] = np.array(
                self.cutoff_list['bond'], dtype=np.float)
        if len(self.cutoff_list.get('cut3b', [])) > 0 \
                and ntriplet > 0:
            hyps_mask['cutoff_3b'] = np.array(
                self.cutoff_list['cut3b'], dtype=np.float)
            hyps_mask['ncut3b'] = self.n['cut3b']
            hyps_mask['cut3b_mask'] = self.mask['cut3b']
        if len(self.cutoff_list.get('mb', [])) > 0 \
                and nmb > 0:
            hyps_mask['cutoff_mb'] = np.array(
                self.cutoff_list['mb'], dtype=np.float)

        self.hyps_mask = hyps_mask
        if (self.cutoffs_array[2] > 0) and nmb > 0:
            hyps_mask['cutoffs'] = self.cutoffs_array
        else:
            hyps_mask['cutoffs'] = self.cutoffs_array[:2]

        if self.n['specie'] < 2:
            print("only one type of elements was defined. Please use multihyps=False",
                  file=self.fout)

        return hyps_mask

    @staticmethod
    def check_instantiation(hyps_mask):
        """
        Runs a series of checks to ensure that the user has not supplied
        contradictory arguments which will result in undefined behavior
        with multiple hyperparameters.
        :return:
        """

        # backward compatability
        if ('nspec' in hyps_mask):
            hyps_mask['nspecie'] = hyps_mask['nspec']
        if ('spec_mask' in hyps_mask):
            hyps_mask['specie_mask'] = hyps_mask['spec_mask']
        if ('train_noise' not in hyps_mask):
            hyps_mask['train_noise'] = True

        assert isinstance(hyps_mask, dict)

        assert 'nspecie' in hyps_mask, "nspecie key missing in " \
            "hyps_mask dictionary"
        assert 'specie_mask' in hyps_mask, "specie_mask key " \
            "missing " \
            "in hyps_mask dicticnary"

        nspecie = hyps_mask['nspecie']
        hyps_mask['specie_mask'] = nparray(
            hyps_mask['specie_mask'], dtype=np.int)

        if 'nbond' in hyps_mask:
            n2b = hyps_mask['nbond']
            assert n2b > 0
            assert isinstance(n2b, int)
            hyps_mask['bond_mask'] = nparray(
                hyps_mask['bond_mask'], dtype=np.int)
            if n2b > 0:
                bmask = hyps_mask['bond_mask']
                assert (npmax(bmask) < n2b)
                assert len(bmask) == nspecie ** 2, \
                    f"wrong dimension of bond_mask: " \
                    f" {len(bmask)} != nspecie^2 {nspecie**2}"
                for t2b in range(nspecie):
                    for t2b_2 in range(t2b, nspecie):
                        assert bmask[t2b*nspecie+t2b_2] == bmask[t2b_2*nspecie+t2b], \
                            'bond_mask has to be symmetric'
        else:
            n2b = 0

        if 'ntriplet' in hyps_mask:
            n3b = hyps_mask['ntriplet']
            assert n3b > 0
            assert isinstance(n3b, int)
            hyps_mask['triplet_mask'] = nparray(
                hyps_mask['triplet_mask'], dtype=np.int)
            if n3b > 0:
                tmask = hyps_mask['triplet_mask']
                assert (npmax(tmask) < n3b)
                assert len(tmask) == nspecie ** 3, \
                    f"wrong dimension of bond_mask: " \
                    f" {len(tmask)} != nspecie^3 {nspecie**3}"

                for t3b in range(nspecie):
                    for t3b_2 in range(t3b, nspecie):
                        for t3b_3 in range(t3b_2, nspecie):
                            assert tmask[t3b*nspecie*nspecie+t3b_2*nspecie+t3b_3] \
                                == tmask[t3b*nspecie*nspecie+t3b_3*nspecie+t3b_2], \
                                'bond_mask has to be symmetric'
                            assert tmask[t3b*nspecie*nspecie+t3b_2*nspecie+t3b_3] \
                                == tmask[t3b_2*nspecie*nspecie+t3b*nspecie+t3b_3], \
                                'bond_mask has to be symmetric'
                            assert tmask[t3b*nspecie*nspecie+t3b_2*nspecie+t3b_3] \
                                == tmask[t3b_2*nspecie*nspecie+t3b_3*nspecie+t3b], \
                                'bond_mask has to be symmetric'
                            assert tmask[t3b*nspecie*nspecie+t3b_2*nspecie+t3b_3] \
                                == tmask[t3b_3*nspecie*nspecie+t3b*nspecie+t3b_2], \
                                'bond_mask has to be symmetric'
                            assert tmask[t3b*nspecie*nspecie+t3b_2*nspecie+t3b_3] \
                                == tmask[t3b_3*nspecie*nspecie+t3b_2*nspecie+t3b], \
                                'bond_mask has to be symmetric'
        else:
            n3b = 0

        if 'nmb' in hyps_mask:
            nmb = hyps_mask['nmb']
            assert nmb > 0
            assert isinstance(nmb, int)
            hyps_mask['mb_mask'] = nparray(hyps_mask['mb_mask'], dtype=np.int)
            if nmb > 0:
                bmask = hyps_mask['mb_mask']
                assert (npmax(bmask) < nmb)
                assert len(bmask) == nspecie ** 2, \
                    f"wrong dimension of mb_mask: " \
                    f" {len(bmask)} != nspecie^2 {nspecie**2}"
                for tmb in range(nspecie):
                    for tmb_2 in range(tmb, nspecie):
                        assert bmask[tmb*nspecie+tmb_2] == bmask[tmb_2*nspecie+tmb], \
                            'mb_mask has to be symmetric'
        # else:
        #     nmb = 1
        #     hyps_mask['mb_mask'] = np.zeros(nspecie**2, dtype=np.int)

        if 'map' in hyps_mask:
            assert ('original' in hyps_mask), \
                "original hyper parameters have to be defined"
            # Ensure typed correctly as numpy array
            hyps_mask['original'] = nparray(
                hyps_mask['original'], dtype=np.float)

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
            assert nc3b > 0
            assert isinstance(nc3b, int)
            hyps_mask['cut3b_mask'] = nparray(
                hyps_mask['cut3b_mask'], dtype=int)
            assert len(c3b) == hyps_mask['ncut3b'], \
                f'number of 3b cutoff should be the same as ncut3b {ncut3b}'
            assert len(hyps_mask['cut3b_mask']) == nspecie ** 2, \
                f"wrong dimension of cut3b_mask: " \
                f" {len(bmask)} != nspecie^2 {nspecie**2}"
            assert npmax(hyps_mask['cut3b_mask']) < hyps_mask['ncut3b'], \
                f"wrong dimension of cut3b_mask: " \
                f" {len(bmask)} != nspecie^2 {nspecie**2}"

        if 'cutoff_mb' in hyps_mask:
            cmb = hyps_mask['cutoff_mb']
            assert len(cmb) == nmb, \
                f'number of mb cutoff should be the same as nmb {nmb}'

        return hyps_mask

    @staticmethod
    def check_matching(hyps_mask, hyps, cutoffs):
        """
        check whether hyps_mask, hyps and cutoffs are compatible
        used in GaussianProcess
        """

        n2b = hyps_mask.get('nbond', 0)
        n3b = hyps_mask.get('ntriplet', 0)
        nmb = hyps_mask.get('nmb', 0)

        if (len(cutoffs) <= 2):
            assert ((n2b + n3b) > 0)
        else:
            assert ((n2b + n3b + nmb) > 0)

        if 'map' in hyps_mask:
            if (len(cutoffs) <= 2):
                assert (n2b * 2 + n3b * 2 + 1) == len(hyps_mask['original']), \
                    "the hyperparmeter length is inconsistent with the mask"
            else:
                if (nmb == 0):
                    nmb = 1
                    hyps_mask['mb_mask'] = np.zeros(hyps_mask['nspecie']**2, dtype=np.int)
                assert (n2b * 2 + n3b * 2 + nmb * 2 + 1) == len(hyps_mask['original']), \
                    "the hyperparmeter length is inconsistent with the mask"
            assert len(hyps_mask['map']) == len(hyps), \
                "the hyperparmeter length is inconsistent with the mask"
        else:
            if (len(cutoffs) <= 2):
                assert (n2b * 2 + n3b * 2 + 1) == len(hyps), \
                    "the hyperparmeter length is inconsistent with the mask"
            else:
                if (nmb == 0):
                    nmb = 1
                    hyps_mask['mb_mask'] = np.zeros(hyps_mask['nspecie']**2, dtype=np.int)
                assert (n2b * 2 + n3b * 2 + nmb*2 + 1) == len(hyps), \
                    "the hyperparmeter length is inconsistent with the mask"

        if 'cutoff_2b' in hyps_mask:
            assert cutoffs[0] >= npmax(hyps_mask['cutoff_2b']), \
                'general cutoff should be larger than all cutoffs listed in hyps_mask'

        if 'cutoff_3b' in hyps_mask:
            assert cutoffs[0] >= npmax(hyps_mask['cutoff_3b']), \
                'general cutoff should be larger than all cutoffs listed in hyps_mask'

        if 'cutoff_mb' in hyps_mask:
            assert cutoffs[0] >= npmax(hyps_mask['cutoff_mb']), \
                'general cutoff should be larger than all cutoffs listed in hyps_mask'

    @staticmethod
    def mask2cutoff(cutoffs, cutoffs_mask):
        """use in flare.env AtomicEnvironment to resolve what cutoff to use"""

        ncutoffs = len(cutoffs)
        scalar_cutoff_2 = cutoffs[0]
        scalar_cutoff_3 = 0
        scalar_cutoff_mb = 0
        if (ncutoffs > 1):
            scalar_cutoff_3 = cutoffs[1]
        if (ncutoffs > 2):
            scalar_cutoff_mb = cutoffs[2]

        if (scalar_cutoff_2 == 0):
            scalar_cutoff_2 = np.max([scalar_cutoff_3, scalar_cutoff_mb])

        if (cutoffs_mask is None):
            return scalar_cutoff_2, scalar_cutoff_3, scalar_cutoff_mb, \
                None, None, None, \
                1, 1, 1, 1, None, None, None, None

        nspecie = cutoffs_mask.get('nspecie', 1)
        nspecie = nspecie
        if (nspecie == 1):
            return scalar_cutoff_2, scalar_cutoff_3, scalar_cutoff_mb, \
                None, None, None, \
                1, 1, 1, 1, None, None, None, None

        n2b = cutoffs_mask.get('nbond', 1)
        n3b = cutoffs_mask.get('ncut3b', 1)
        nmb = cutoffs_mask.get('nmb', 1)
        specie_mask = cutoffs_mask.get('specie_mask', None)
        bond_mask = cutoffs_mask.get('bond_mask', None)
        cut3b_mask = cutoffs_mask.get('cut3b_mask', None)
        mb_mask = cutoffs_mask.get('mb_mask', None)
        cutoff_2b = cutoffs_mask.get('cutoff_2b', None)
        cutoff_3b = cutoffs_mask.get('cutoff_3b', None)
        cutoff_mb = cutoffs_mask.get('cutoff_mb', None)

        if cutoff_2b is not None:
            scalar_cutoff_2 = np.max(cutoff_2b)
        else:
            n2b = 1

        if cutoff_3b is not None:
            scalar_cutoff_3 = np.max(cutoff_3b)
        else:
            n3b = 1

        if cutoff_mb is not None:
            scalar_cutoff_mb = np.max(cutoff_mb)
        else:
            nmb = 1

        return scalar_cutoff_2, scalar_cutoff_3, scalar_cutoff_mb, \
            cutoff_2b, cutoff_3b, cutoff_mb, \
            nspecie, n2b, n3b, nmb, specie_mask, bond_mask, cut3b_mask, mb_mask

    @staticmethod
    def get_2b_hyps(hyps, hyps_mask, multihyps=False):

        original_hyps = np.copy(hyps)
        if (multihyps is True):
            new_hyps = Parameters.get_hyps(hyps_mask, hyps)
            n2b = hyps_mask['nbond']
            new_hyps = np.hstack([new_hyps[:n2b*2], new_hyps[-1]])
            new_hyps_mask = {'nbond': n2b, 'ntriplet': 0,
                             'nspecie': hyps_mask['nspecie'],
                             'specie_mask': hyps_mask['specie_mask'],
                             'bond_mask': hyps_mask['bond_mask']}
            if ('cutoff_2b' in hyps_mask):
                new_hyps_mask['cutoff_2b'] = hyps_mask['cutoff_2b']
        else:
            new_hyps = [hyps[0], hyps[1], hyps[-1]]
            new_hyps_mask = None

        return new_hyps, new_hyps_mask

    @staticmethod
    def get_3b_hyps(hyps, hyps_mask, multihyps=False):

        if (multihyps is True):
            new_hyps = Parameters.get_hyps(hyps_mask, hyps)
            n2b = hyps_mask.get('nbond', 0)
            n3b = hyps_mask['ntriplet']
            new_hyps = np.hstack([new_hyps[n2b*2:n2b*2+n3b*2], new_hyps[-1]])
            new_hyps_mask = {'ntriplet': n3b, 'nbond': 0,
                             'nspecie': hyps_mask['nspecie'],
                             'specie_mask': hyps_mask['specie_mask'],
                             'triplet_mask': hyps_mask['triplet_mask']}
            ncut3b = hyps_mask.get('ncut3b', 0)
            if (ncut3b > 0):
                new_hyps_mask['ncut3b'] = hyps_mask['cut3b_mask']
                new_hyps_mask['cut3b_mask'] = hyps_mask['cut3b_mask']
                new_hyps_mask['cutoff_3b'] = hyps_mask['cutoff_3b']
        else:
            # kind of assuming that 2-body is there
            base = 2
            new_hyps = np.hstack([hyps[0+base], hyps[1+base], hyps[-1]])
            new_hyps_mask = None

        return hyps, hyps_mask

    @staticmethod
    def get_mb_hyps(hyps, hyps_mask, multihyps=False):

        if (multihyps is True):
            new_hyps = Parameters.get_hyps(hyps_mask, hyps)
            n2b = hyps_mask.get('n2b', 0)
            n3b = hyps_mask.get('n3b', 0)
            n23b2 = (n2b+n3b)*2
            nmb = hyps_mask['nmb']

            new_hyps = np.hstack([new_hyps[n23b2:n23b2+nmb*2], new_hyps[-1]])

            new_hyps_mask = {'nmb': nmb, 'nbond': 0, 'ntriplet':0,
                             'nspecie': hyps_mask['nspecie'],
                             'specie_mask': hyps_mask['specie_mask'],
                             'mb_mask': hyps_mask['mb_mask']}

            if ('cutoff_mb' in hyps_mask):
                new_hyps_mask['cutoff_mb'] = hyps_mask['cutoff_mb']
        else:
            # kind of assuming that 2+3 are there
            base = 4
            new_hyps = np.hstack([hyps[0+base], hyps[1+base], hyps[-1]])
            new_hyps_mask = None

        return new_hyps, new_hyps_mask

    @staticmethod
    def get_cutoff(coded_species, cutoff, hyps_mask):

        if (len(coded_species)==2):
            if (hyps_mask is None):
                return cutoff[0]
            elif ('cutoff_2b' not in hyps_mask):
                return cutoff[0]

            ele1 = hyps_mask['species_mask'][coded_species[0]]
            ele2 = hyps_mask['species_mask'][coded_species[1]]
            bond_type = hyps_mask['bond_mask'][ \
                    hyps_mask['nspecie']*ele1 + ele2]
            return hyps_mask['cutoff_2b'][bond_type]

        elif (len(coded_species)==3):
            if (hyps_mask is None):
                return np.ones(3)*cutoff[1]
            elif ('cutoff_3b' not in hyps_mask):
                return np.ones(3)*cutoff[1]

            ele1 = hyps_mask['species_mask'][coded_species[0]]
            ele2 = hyps_mask['species_mask'][coded_species[1]]
            ele3 = hyps_mask['species_mask'][coded_species[2]]
            bond1 = hyps_mask['cut3b_mask'][ \
                        hyps_mask['nspecie']*ele1 + ele2]
            bond2 = hyps_mask['cut3b_mask'][ \
                        hyps_mask['nspecie']*ele1 + ele3]
            bond12 = hyps_mask['cut3b_mask'][ \
                        hyps_mask['nspecie']*ele2 + ele3]
            return np.array([hyps_mask['cutoff_3b'][bond1],
                             hyps_mask['cutoff_3b'][bond2],
                             hyps_mask['cutoff_3b'][bond12]])
        else:
            raise NotImplementedError

    @staticmethod
    def get_hyps(hyps_mask, hyps):
        if 'map' in hyps_mask:
            newhyps = np.copy(hyps_mask['original'])
            for i, ori in enumerate(hyps_mask['map']):
                newhyps[ori] = hyps[i]
            return newhyps
        else:
            return hyps

    @staticmethod
    def compare_dict(dict1, dict2):

        if type(dict1) != type(dict2):
            return False

        if dict1 is None:
            return True

        for k in ['nspecie', 'specie_mask', 'nbond', 'bond_mask',
                  'cutoff_2b', 'ntriplet', 'triplet_mask',
                  'n3b', 'cut3b_mask', 'nmb', 'mb_mask',
                  'cutoff_mb', 'map']: #, 'train_noise']:
            if (k in dict1) != (k in dict2):
                return False
            elif (k in dict1):
                if not (np.isclose(dict1[k], dict2[k]).all()):
                    return False

        for k in ['train_noise']:
            if (k in dict1) != (k in dict2):
                return False
            elif (k in dict1):
                if dict1[k] !=dict2[k]:
                    return False
        return True
