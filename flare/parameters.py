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

from flare.output import set_logger
from flare.utils.element_coder import element_to_Z, Z_to_element


class Parameters():
    '''
    '''

    all_kernel_types = ['twobody', 'threebody', 'manybody']
    cutoff_types = {'cut3b': 'threebody'}
    ndim = {'twobody': 2, 'threebody': 3, 'manybody': 2, 'cut3b': 2}
    n_kernel_parameters = {'twobody': 2,
                           'threebody': 2, 'manybody': 2, 'cut3b': 0}

    cutoff_types_keys = list(cutoff_types.keys())
    cutoff_types_values = list(cutoff_types.values())

    logger = set_logger("Parameters", stream=True,
                        fileout_name=None, verbose="info")

    def __init__(self):
        '''
        Enumerate all keys and  their default values that hyps_mask should store
        '''

        self.param_dict = {'nspecie': 1,
                           'ntwobody': 0,
                           'nthreebody': 0,
                           'ncut3b': 0,
                           'nmanybody': 0,
                           'specie_mask': None,
                           'twobody_mask': None,
                           'threebody_mask': None,
                           'cut3b_mask': None,
                           'manybody_mask': None,
                           'twobody_cutoff_list': None,
                           'threebody_cutoff_list': None,
                           'manybody_cutoff_list': None,
                           'train_noise': True,
                           'energy_noise': 0,
                           'map': None,
                           'original_hyps': [],
                           'original_labels': []
                           }
        self.hyps = None
        self.hyp_labels = None
        self.cutoffs = {}
        self.kernels = []

    @staticmethod
    def cutoff_array_to_dict(cutoffs):
        '''
        Convert old cutoffs array to the new dictionary format
        '''

        if isinstance(cutoffs, dict):
            return cutoffs

        if (cutoffs is not None) and not isinstance(cutoffs, dict):
            DeprecationWarning("cutoffs is replace by dictionary")
            newcutoffs = {'twobody': cutoffs[0]}
            if len(cutoffs) > 1:
                newcutoffs['threebody'] = cutoffs[1]
            if len(cutoffs) > 2:
                newcutoffs['manybody'] = cutoffs[2]
            Parameters.logger.debug("Convert cutoffs array to cutoffs dict")
            Parameters.logger.debug("Original", cutoffs)
            Parameters.logger.debug("Now", newcutoffs)
            return newcutoffs
        else:
            raise TypeError("cannot handle cutoffs with {type(cutoffs)} type")

    @staticmethod
    def backward(kernels, param_dict):

        if param_dict is None:
            param_dict = {}

        # update old keys to new keys. for example nspec to nspecies
        replace_list = {'spec': 'specie', 'bond': 'twobody',
                        'triplet': 'threebody', 'mb': 'manybody'}
        keys = list(param_dict.keys())
        for key in keys:
            for original in replace_list:
                if original in key and replace_list[original] not in key:
                    newkey = key.replace(original, replace_list[original])
                    param_dict[newkey] = param_dict[key]
                    DeprecationWarning(
                        "{key} is being replaced with {newkey}")

        # add a couple new keys that was not there previously
        if 'train_noise' not in param_dict:
            param_dict['train_noise'] = True
            DeprecationWarning(
                "train_noise has to be in hyps_mask, set to True")
        if 'nspecie' not in param_dict:
            param_dict['nspecie'] = 1

        # sort the kernels dictionary again. but this can result in
        # wrong results...
        if set(kernels) != set(param_dict.get("kernels", [])):

            start = 0
            for k in Parameters.all_kernel_types:
                if k in kernels:
                    if k+'_start' not in param_dict:
                        param_dict[k+'_start'] = start
                    if 'n'+k not in param_dict:
                        Parameters.logger.debug("add in hyper parameter separators"
                                                "for", k)
                        param_dict['n'+k] = 1
                        start += Parameters.n_kernel_parameters[k]
                    else:
                        start += param_dict['n'+k] * \
                            Parameters.n_kernel_parameters[k]
                else:
                    Warning("inconsistency between input kernel and kernel list"
                            "stored in hyps_mask")

            Parameters.logger.debug("Replace kernel array in param_dict")
            param_dict['kernels'] = deepcopy(kernels)

        return param_dict

    @staticmethod
    def check_instantiation(hyps, cutoffs, kernels, param_dict):
        """
        Runs a series of checks to ensure that the user has not supplied
        contradictory arguments which will result in undefined behavior
        with multiple hyperparameters.

        :return:
        """

        assert isinstance(param_dict, dict)
        assert isinstance(cutoffs, dict)
        assert isinstance(kernels, list)

        param_dict['cutoffs'] = cutoffs

        # double check nspecie is there
        nspecie = param_dict['nspecie']
        if nspecie > 1:
            assert 'specie_mask' in param_dict, "specie_mask key " \
                "missing " \
                "in param_dict dictionary"
            param_dict['specie_mask'] = nparray(
                param_dict['specie_mask'], dtype=np.int)

        # for each kernel, check whether it is defined
        # and the length of corresponding hyper-parameters
        hyps_length = 0
        used_parameters = np.zeros_like(hyps, dtype=bool)
        for kernel in kernels+list(Parameters.cutoff_types.keys()):

            n = param_dict.get(f'n{kernel}', 0)
            assert isinstance(n, int)

            if kernel not in list(Parameters.cutoff_types.keys()):
                hyps_length += Parameters.n_kernel_parameters[kernel]*n
                assert n > 0, f"{kernel} has 0 hyperparameters defined"

                # check all corresponding keys exist
                assert kernel in cutoffs.keys()
                assert kernel+"_start" in param_dict

                # check the partition of hyperparameters are not used
                start = param_dict[kernel+"_start"]
                length = Parameters.n_kernel_parameters[kernel]*n
                assert not used_parameters[start:start+length].any()
                used_parameters[start:start+length] = True

            if n > 1:

                assert f'{kernel}_mask' in param_dict, f"{kernel}_mask key " \
                    "missing " \
                    "in param_dict dictionary"

                # check mask has the right dimension and values
                mask = param_dict[f'{kernel}_mask']
                param_dict[f'{kernel}_mask'] = nparray(mask, dtype=np.int)

                assert (npmax(mask) < n)
                dim = Parameters.ndim[kernel]
                assert len(mask) == nspecie ** dim, \
                    f"wrong dimension of {kernel}_mask: " \
                    f" {len(mask)} != nspec ^ {dim} {nspecie**dim}"

                # check whether the mask array is symmetrical
                # enumerate all possible combinations
                all_comb = list(combinations_with_replacement(
                    np.arange(nspecie), dim))
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
                                f'{kernel}_mask has to be symmetrical'

                if kernel not in list(Parameters.cutoff_types.keys()):
                    if kernel+'_cutoff_list' in param_dict:
                        cutoff_list = param_dict[kernel+'_cutoff_list']
                        assert len(cutoff_list) == n, \
                            f'number of cutoffs should be the same as n {n}'
                        assert npmax(cutoff_list) <= cutoffs[kernel]
            else:
                assert f'{kernel}_mask' not in param_dict,\
                        f'{kernel}_mask should not be in param_dict'
                assert f'{kernel}_cutoff_list' not in param_dict, \
                        f'{kernel}_cutoff_list should not be in param_dict'

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

        hyps = Parameters.get_hyps(param_dict, hyps)

        hyps_length += 1
        assert hyps_length == len(hyps), \
            "the hyperparmeter length is inconsistent with the mask"

        return param_dict

    @staticmethod
    def get_component_hyps(param_dict, kernel_name, hyps=None, constraint=False, noise=False):
        '''
        return the hyper-parameters correspond to the kernel specified by kernel_name

        Args:

        param_dict (dict): the hyps_mask dictionary used/stored in GaussianProcess
        kernel_name (str): the name of the kernel.
        hyps (np.array): if hyps is None, use the one stored in param_dict
        constraint (bool): if True, return one additional list that shows whether the
                           hyper-parmaeters can be trained
        noise (bool): if True, the last element of returned hyper-parameters is
                      the noise variance.

        return: hyper-parameters, and whether they can be optimized
        '''

        if kernel_name not in param_dict['kernels']:
            if constraint:
                if noise:
                    return [None, None, None], [None, None]
                else:
                    return [None, None], [None, None]
            else:
                if noise:
                    return [None, None, None]
                else:
                    return [None, None]

        hyps, opt = Parameters.get_hyps(param_dict, hyps=hyps, constraint=True)
        s = param_dict[kernel_name+'_start']
        n = param_dict[f'n{kernel_name}']

        newhyps = [hyps[s:s+n], hyps[s+n:s+2*n]]
        newopt = [opt[s:s+n], opt[s+n:s+2*n]]

        if noise:
            newhyps += [hyps[-1]]

        if constraint:
            return newhyps, newopt
        else:
            return newhyps

    @staticmethod
    def get_component_mask(param_dict, kernel_name, hyps=None):
        '''
        return the hyper-parameter masking correspond to the kernel specified by kernel_name

        Args:

        param_dict (dict): the hyps_mask dictionary used/stored in GaussianProcess
        kernel_name (str): the name of the kernel.
        hyps (np.array): if hyps is None, use the one stored in param_dict

        return: hyper-parameters, cutoffs, and new hyps_mask
        '''

        if kernel_name in param_dict['kernels']:
            new_dict = {}
            new_dict['kernels'] = [kernel_name]

            new_dict[kernel_name+'_start'] = 0

            name_list = ['nspecie', 'specie_mask',
                         'n'+kernel_name, kernel_name+'_mask',
                         kernel_name+'_cutoff_list']

            if kernel_name in Parameters.cutoff_types_values:

                key_ind = Parameters.cutoff_types_values.index(kernel_name)
                cutoff_key = Parameters.cutoff_types_keys[key_ind]
                name_list += ['n'+cutoff_key, cutoff_key+'_mask']

            for name in name_list:
                if name in param_dict:
                    new_dict[name] = deepcopy(param_dict[name])

            hyps = np.hstack(Parameters.get_component_hyps(
                param_dict, kernel_name, hyps=hyps, noise=True))

            cutoffs = {}
            if 'twobody' in param_dict['cutoffs']:
                cutoffs['twobody'] = param_dict['cutoffs']['twobody']
            cutoffs[kernel_name] = param_dict['cutoffs'][kernel_name]

            return hyps, cutoffs, new_dict
        else:
            return [], {}, {}

    @staticmethod
    def get_noise(param_dict, hyps=None, constraint=False):
        '''
        get the noise parameters

        Args:

        constraint (bool): if True, return one additional list that shows whether the
                           hyper-parmaeters can be trained
        noise (bool): if True, the last element of returned hyper-parameters is
                      the noise variance.
        '''
        hyps = Parameters.get_hyps(param_dict, hyps=hyps)
        if constraint:
            return hyps[-1], param_dict['train_noise']
        else:
            return hyps[-1]

    @staticmethod
    def get_cutoff(kernel_name, coded_species, param_dict):
        '''
        get the cutoff

        Args:

        kernel_name (str): name of the kernel
        coded_species (list): list of element names
        param_dict (dict): hyps_mask

        '''

        cutoffs = param_dict['cutoffs']
        universal_cutoff = cutoffs[kernel_name]

        if f'{kernel_name}_cutoff_list' in param_dict:

            specie_mask = param_dict['species_mask']
            cutoff_list = param_dict[f'{kernel_name}_cutoff_list']

            if kernel_name not in Parameters.cutoff_types_values:
                mask_id = 0
                for ele in coded_specie:
                    mask_id += specie_mask[ele]
                    mask_id *= nspecie
                mask_id = mask_id // nspecie
                mask_id = param_dict[kernel_name+'_mask'][mask_id]
                return cutoff_list[mask_id]
            else:

                key_ind = Parameters.cutoff_types_values.index(kernel_name)
                cutoff_key = Parameters.cutoff_types_keys[cutoff_key]

                cut_mask = param_dict[cutoff_key+'_mask']
                ele1 = species_mask[coded_species[0]]
                ele2 = species_mask[coded_species[1]]
                ele3 = species_mask[coded_species[2]]
                twobody1 = cut_mask[param_dict['nspecie']*ele1 + ele2]
                twobody2 = cut_mask[param_dict['nspecie']*ele1 + ele3]
                twobody12 = cut_mask[param_dict['nspecie']*ele2 + ele3]
                return np.array([cutoff_list[twobody1],
                                 cutoff_list[twobody2],
                                 cutoff_list[twobody12]])
        else:
            if kernel_name != 'threebody':
                return [universal_cutoff]
            else:
                return [universal_cutoff]*3

    @staticmethod
    def get_hyps(param_dict, hyps=None, constraint=False, label=False):
        '''
        get the cutoff

        Args:

        kernel_name (str): name of the kernel
        coded_species (list): list of element names
        param_dict (dict): hyps_mask

        '''

        if hyps is None:
            hyps = param_dict['hyps']

        if 'map' in param_dict:
            label_key = 'original_labels'
            newhyps = np.copy(param_dict['original_hyps'])
            opt = np.zeros_like(newhyps, dtype=bool)
            for i, ori in enumerate(param_dict['map']):
                newhyps[ori] = hyps[i]
                opt[ori] = True
        else:
            label_key = 'hyp_labels'
            newhyps = np.copy(hyps)
            opt = np.zeros_like(hyps, dtype=bool)

        if constraint:
            if label:
                return newhyps, opt, param_dict.get(label_key, None)
            else:
                return newhyps, opt
        else:
            if label:
                return newhyps, param_dict.get(label_key, None)
            else:
                return newhyps

    @staticmethod
    def compare_dict(dict1, dict2):
        '''
        compare whether two hyps_masks are the same
        '''

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
        for k in Parameters.cutoff_types:
            list_of_names += ['n'+k]
            list_of_names += [k+'_mask']

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
                if not (dict1[k] == dict2[k]):
                    return False

        for k in ['train_noise']:
            if (k in dict1) != (k in dict2):
                return False
            elif k in dict1:
                if dict1[k] != dict2[k]:
                    return False

        return True
