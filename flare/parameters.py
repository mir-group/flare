import inspect
import json
import logging
import math
import numpy as np
import pickle
import time

from copy import deepcopy
from itertools import combinations_with_replacement, permutations
from numpy.random import random
from numpy import array as nparray
from numpy import max as npmax
from typing import List, Callable, Union
from warnings import warn
from sys import stdout
from os import devnull

from flare.utils.element_coder import element_to_Z, Z_to_element


class Parameters():

    all_kernel_types = ['bond', 'triplet', 'mb']
    ndim = {'bond': 2, 'triplet': 3, 'mb': 2, 'cut3b': 2}

    def __init__(self):

        self.nspecie = 0
        self.specie_mask = None

        # if nxx > 1, the kernel array should also be there
        self.nbond = 0
        self.ntriplet = 0
        self.nmb = 0

        self.bond_mask = None
        self.bond_start = 0

        self.triplet_mask = None
        self.triplet_start = 0

        self.mb_mask = None
        self.mb_start = 0

    @staticmethod
    def check_instantiation(param_dict):
        """
        Runs a series of checks to ensure that the user has not supplied
        contradictory arguments which will result in undefined behavior
        with multiple hyperparameters.
        :return:
        """

        assert isinstance(param_dict, dict)

        # backward compatability
        if 'nspec' in param_dict:
            param_dict['nspecie'] = param_dict['nspec']
        if 'spec_mask' in param_dict:
            param_dict['specie_mask'] = param_dict['spec_mask']
        if 'train_noise' not in param_dict:
            param_dict['train_noise'] = True

        assert 'nspecie' in param_dict, "nspecie key missing in " \
            "param_dict dictionary"

        nspecie = param_dict['nspecie']
        kernels = param_dict['kernels']
        if nspecie > 1:
            assert 'specie_mask' in param_dict, "specie_mask key " \
                "missing " \
                "in param_dict dictionary"
            param_dict['specie_mask'] = nparray(
                param_dict['specie_mask'], dtype=np.int)

        cutoffs = param_dict['cutoffs']

        hyps_length = 0
        for kernel in kernels+['cut3b']:

            n = param_dict.get(f'n{kernel}', 0)
            assert isinstance(n, int)

            if kernel != 'cut3b':
                hyps_length += 2*n
                assert kernel in cutoffs
                assert n > 0

            if n > 1:
                assert f'{kernel}_mask' in param_dict, f"{kernel}_mask key " \
                    "missing " \
                    "in param_dict dictionary"
                mask = param_dict[f'{kernel}_mask']
                param_dict[f'{kernel}_mask'] = nparray(mask, dtype=np.int)
                assert (npmax(mask) < n)

                dim = Parameters.ndim[kernel]
                assert len(mask) == nspecie ** dim, \
                    f"wrong dimension of bond_mask: " \
                    f" {len(mask)} != nspec ^ {dim} {nspecie**dim}"

                all_comb=list(combinations_with_replacement(np.arange(nspecie), dim))
                for comb in all_comb:
                    mask_value = None
                    perm = list(permutations(comb))
                    for ele_list in perm:
                        mask_id = 0
                        for ele in ele_list:
                            mask_id += ele
                            mask_id *= nspecie
                        mask_id = mask_id // nspecie
                        if mask_value == None:
                            mask_value = mask[mask_id]
                        else:
                            assert mask[mask_id] == mask_value, \
                                   'bond_mask has to be symmetrical'

                if kernel != 'cut3b':
                    if kernel+'_cutoff_list' in param_dict:
                        cutoff_list = param_dict[kernel+'_cutoff_list']
                        assert len(cutoff_list) == n, \
                            f'number of cutoffs should be the same as n {n}'
                        assert npmax(cutoff_list) <= cutoffs[kernel]
            else:
                assert f'{kernel}_mask' not in param_dict
                assert f'{kernel}_cutof_list' not in param_dict

        hyps = param_dict['hyps']
        if 'map' in param_dict:
            assert ('original_hyps' in param_dict), \
                "original hyper parameters have to be defined"
            # Ensure typed correctly as numpy array
            param_dict['original_hyps'] = nparray(
                param_dict['original_hyps'], dtype=np.float)
            if (len(param_dict['original_hyps']) - 1) not in param_dict['map']:
                assert param_dict['train_noise'] is False, \
                    "train_noise should be False when noise is not in hyps"
            assert len(param_dict['map']) == len(hyps), \
                "the hyperparmeter length is inconsistent with the mask"
            assert npmax(param_dict['map']) < len(param_dict['original_hyps'])
        else:
            assert param_dict['train_noise'] is True, \
                "train_noise should be True when map is not used"
        hyps = Parameters.get_hyps(param_dict)

        hyps_length += 1
        assert hyps_length == len(hyps), \
                    "the hyperparmeter length is inconsistent with the mask"
        for var in hyps:
            assert var >= 0

        return param_dict


    @staticmethod
    def get_component_hyps(param_dict, kernel_name, constraint=False, noise=False):

        if kernel_name not in param_dict['kernels']:
            return None

        hyps, opt = Parameters.get_hyps(param_dict, constraint=True)
        s = param_dict[kernel_name+'_start']
        e = s + 2*param_dict[f'n{kernel_name}']

        newhyps = hyps[s:e]

        if noise:
            newhyps = np.hstack(newhyps, hyps[-1])

        if constraint:
            return newhyps, opt[s:e]
        else:
            return newhyps

    @staticmethod
    def get_component_mask(param_dict, kernel_name):

        if kernel_name in param_dict['kernels']:
            new_dict = {}
            new_dict['hyps'] = get_component_hyps(param_dict, kernel_name, noise=True)
            new_dict['kernels'] = [kernel_name]
            new_dict['cutoffs'] = {kernel_name: param_dict['cutoffs'][kernel_name]}
            new_dict[kernel_name+'_start'] = 0

            name_list = ['nspecie', 'specie_mask',
                         n+kernel_name, kernel_name+'_mask',
                         kernel_name+'_cutoff_list']
            if kernel_name == 'triplet':
                name_list += ['ncut3b', 'cut3b_mask']

            for name in name_list:
                if name in param_dict:
                    new_dict[name] = param_dict[name]

            return new_dict
        else:
            return {}

    @staticmethod
    def get_noise(param_dict, constraint=False):
        hyps = Parameters.get_hyps(param_dict)
        if constraint:
            return hyps[-1], param_dict['train_noise']
        else:
            return hyps[-1]

    @staticmethod
    def get_cutoff(kernel_name, coded_species, param_dict):

        cutoffs = param_dict['cutoffs']
        universal_cutoff = cutoffs[kernel_name]

        if f'{kernel_name}_cutoff_list' in param_dict:

            specie_mask = param_dict['species_mask']
            cutoff_list = param_dict[f'{kernel_name}_cutoff_list']

            if kernel_name != 'triplet':
                mask_id = 0
                for ele in coded_specie:
                    mask_id += specie_mask[ele]
                    mask_id *= nspecie
                mask_id = mask_id // nspecie
                mask_id = param_dict[kernel_name+'_mask'][mask_id]
                return cutoff_list[mask_id]
            else:
                cut3b_mask = param_dict['cut3b_mask']
                ele1 = species_mask[coded_species[0]]
                ele2 = species_mask[coded_species[1]]
                ele3 = species_mask[coded_species[2]]
                bond1 = cut3b_mask[param_dict['nspecie']*ele1 + ele2]
                bond2 = cut3b_mask[param_dict['nspecie']*ele1 + ele3]
                bond12 = cut3b_mask[param_dict['nspecie']*ele2 + ele3]
                return np.array([cutoff_list[bond1],
                                 cutoff_list[bond2],
                                 cutoff_list[bond12]])
        else:
            return universal_cutoff


    @staticmethod
    def get_hyps(param_dict, constraint=False):

        hyps = param_dict['hyps']
        if 'map' in param_dict:
            newhyps = np.copy(param_dict['original_hyps'])
            opt = np.zeros_like(newhyps, dtype=bool)
            for i, ori in enumerate(param_dict['map']):
                newhyps[ori] = hyps[i]
                opt[ori] = True
        else:
            newhyps = hyps
            opt = np.zeros_like(hyps, dtype=bool)

        if constraint:
            return newhyps, opt
        else:
            return newhyps

    @staticmethod
    def compare_dict(dict1, dict2):

        if type(dict1) != type(dict2):
            return False

        if dict1 is None:
            return True

        list_of_names = ['nspecie', 'specie_mask', 'map', 'original_hyps']
        for k in Parameters.all_kernel_types:
            list_of_names += ['n'+k]
            list_of_names += [k+'_mask']
            list_of_names += ['cutoff_'+k]
            list_of_names += [k+'_cutoff_list']
        list_of_names += ['ncut3b']
        list_of_names += ['cut3b_mask']

        for k in list_of_names:
            if (k in dict1) != (k in dict2):
                return False
            elif k in dict1:
                if not (np.isclose(dict1[k], dict2[k]).all()):
                    return False

        for k in ['hyp_labels', 'original_labels']:
            if (k in dict1) != (k in dict2):
                return False
            elif k in dict1:
                if not (dict1[k]==dict2[k]).all():
                    return False

        for k in ['train_noise']:
            if (k in dict1) != (k in dict2):
                return False
            elif k in dict1:
                if dict1[k] !=dict2[k]:
                    return False

        return True
