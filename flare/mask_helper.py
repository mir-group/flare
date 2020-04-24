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

from flare.util import  element_to_Z


class ParameterMasking():

    def __init__(self, hyps_mask=None):
        self.n = {}
        self.element_group = {}
        self.all_elements = {}
        self.all_ele_group = {}

        for group_type in ['spec', 'bond', 'triplet', 'cut3b', 'mb']:
            self.n[group_type] = 0
            self.element_group[group_type] = []
            self.all_elements[group_type] = []
            self.all_ele_group[group_type] = []
        self.sigma = {'bond':{}, 'triplet':{}, 'mb':{}}
        self.ls = {'bond':{}, 'triplet':{}, 'mb': {}}
        self.cutoff = {'bond':{}, 'cut3b':{}, 'mb': {}}
        self.hyps_sig = {}
        self.hyps_ls = {}
        self.mask = {}
        self.cutoff_list = {}

    def define_group(self, group_type, name, element_list, atomic_str=False):
        """
        group_type (str): spec, bond, triplet, cut3b, mb
        name (str): the name use for indexing
        element_list (list):
        """

        if (name in self.all_ele_group[group_type]):
            groupid = self.all_ele_group[group_type].index(name)
        else:
            groupid = self.n[group_type]
            self.all_ele_group[group_type].append(name)
            self.element_group[group_type].append([])
            self.n[group_type] += 1

        if (group_type is 'spec'):
            for ele in element_list:
                assert ele not in self.all_elements['spec'], \
                        "the element has already been defined"
                self.element_group['spec'][groupid].append(ele)
                self.all_elements['spec'].append(ele)
        else:
            gid = []
            for ele_name in element_list:
                if (atomic_str):
                    for idx in range(self.n['spec']):
                        if (ele_name in self.element_group['spec'][idx]):
                            gid += [idx]
                            print(f"Define {group_type}: Element {ele_name}"\
                                  f"is in group {self.all_ele_group[idx]}")
                else:
                    print(self.element_group['spec'])
                    print(self.all_ele_group['spec'])
                    gid += [self.all_ele_group['spec'].index(ele_name)]

            for ele in self.all_elements[group_type]:
                assert set(gid) != set(ele), \
                    f"the {group_type} {ele} has already been defined"

            self.element_group[group_type][groupid].append(gid)
            self.all_elements[group_type].append(gid)

    def define_parameters(self, group_type, name, sig, ls, cutoff=None):

        if (group_type != 'cut3b'):
            self.sigma[group_type][name] = sig
            self.ls[group_type][name] = ls
        if (cutoff is not None):
            self.cutoff[group_type][name] = cutoff


    def print_group(self, group_type):
        """
        group_type (str): spec, bond, triplet, cut3b, mb
        name (str): the name use for indexing
        element_list (list):
        """
        aeg = self.all_ele_group[group_type]
        if (group_type == "spec"):
            self.nspec = self.n['spec']
            self.spec_mask = np.ones(118, dtype=np.int)*(self.n['spec']-1)
            for idt in range(self.n['spec']):
                if (aeg[idt] == "*"):
                    for i in range(118):
                        if self.spec_mask[i] > idt:
                            self.spec_mask[i] = idt
                            print(f"element [i] is defined as type {idt} with name"\
                                f"aeg[idt]")
                else:
                    for ele in self.element_group['spec'][idt]:
                        atom_n = element_to_Z(ele)
                        self.spec_mask[atom_n] = idt
                        print(f"elemtn {ele} is defined as type {idt} with name"\
                                f"aeg[idt]")
            print("all the remaining elements are left as type 0")
        elif (group_type in ['bond', 'cut3b', 'mb']):
            nspec = self.n['spec']
            self.mask[group_type] = np.ones(nspec**2, dtype=np.int)*(self.n[group_type]-1)
            self.hyps_sig[group_type] = []
            self.hyps_ls[group_type] = []
            for idt in range(self.n[group_type]):
                name = aeg[idt]
                if (aeg[idt] == "*"):
                    for i in range(nspec**2):
                        if (self.mask[group_type][i]>idt):
                            self.mask[group_type][i] = idt
                else:
                    for bond in self.element_group[group_type][idt]:
                        g1 = bond[0]
                        g2 = bond[1]
                        self.mask[group_type][g1+g2*nspec] = idt
                        self.mask[group_type][g2+g1*nspec] = idt
                        print(f"{group_type} {bond} is defined as type {idt} with name"\
                                f"{name}")
                if (group_type != 'cut3b'):
                    self.hyps_sig[group_type] += [self.sigma[group_type][name]]
                    self.hyps_ls[group_type] += [self.ls[group_type][name]]
                    print(f"   using hyper-parameters of {self.hyps_sig[group_type][-1]}"\
                            "{ self.hyps_ls[group_type][-1]}")
            if len(self.cutoff[group_type]) >0:
                self.cutoff_list[group_type] = []
                for idt in range(self.n[group_type]):
                    self.cutoff_list[group_type] += [self.cutoff[group_type][aeg[idt]]]
        elif (group_type == "triplet"):
            nspec = self.n['spec']
            self.ntriplet = self.n['triplet']
            self.mask[group_type] = np.ones(nspec**3, dtype=np.int)*(self.ntriplet-1)
            self.hyps_sig[group_type] = []
            self.hyps_ls[group_type] = []
            for idt in range(self.n['triplet']):
                if (aeg[idt] == "*"):
                    for i in range(nspec**3):
                        if (self.mask[group_type][i]>idt):
                            self.mask[group_type][i] = idt
                else:
                    for triplet in self.element_group['triplet'][idt]:
                        g1 = triplet[0]
                        g2 = triplet[1]
                        g3 = triplet[2]
                        self.mask[group_type][g1+g2*nspec+g3*nspec**2] = idt
                        self.mask[group_type][g1+g3*nspec+g2*nspec**2] = idt
                        self.mask[group_type][g2+g1*nspec+g3*nspec**2] = idt
                        self.mask[group_type][g2+g3*nspec+g1*nspec**2] = idt
                        self.mask[group_type][g3+g1*nspec+g2*nspec**2] = idt
                        self.mask[group_type][g3+g2*nspec+g1*nspec**2] = idt
                        print(f"triplet {triplet} is defined as type {idt} with name"\
                                "self.all_ele_group[group_type][idt]")
                self.hyps_sig[group_type] += [self.sigma['triplet'][self.all_ele_group[group_type][idt]]]
                self.hyps_ls[group_type] += [self.ls['triplet'][self.all_ele_group[group_type][idt]]]
                print(f"   using hyper-parameters of {self.hyps_sig[group_type][-1]}"\
                        "{ self.hyps_ls[group_type][-1]}")
        else:
            pass

    def generate_dict(self):
        """Dictionary representation of the GP model."""
        if self.n['spec'] < 2:
            print("only one type of elements was defined. return None")
            hyps_mask = None
        else:
            self.print_group('spec')
            self.print_group('bond')
            self.print_group('triplet')
            self.print_group('cut3b')
            self.print_group('mb')
            hyps_mask = {}
            hyps_mask['nspec'] = self.n['spec']
            hyps_mask['spec_mask'] = self.spec_mask
            hyps = []
            for group in ['bond', 'triplet', 'mb']:
                if (self.n[group]>1):
                    hyps_mask['n'+group] = self.n[group]
                    hyps_mask[group+'_mask'] = self.mask[group]
                    hyps += [self.hyps_sig[group]]
                    hyps += [self.hyps_ls[group]]
            hyps_mask['original'] = np.hstack(hyps)
            if len(self.cutoff_list.get('bond', []))>0:
                hyps_mask['cutoff_2b'] = self.cutoff_list['bond']
            if len(self.cutoff_list.get('cut3b', []))>0:
                hyps_mask['cutoff_3b'] = self.cutoff_list['cut3b']
                hyps_mask['ncut3b'] = self.n['cut3b']
                hyps_mask['cut3b_mask'] = self.mask['cut3b']
            if len(self.cutoff_list.get('mb', []))>0:
                hyps_mask['cutoff_mb'] = self.cutoff_list['mb']

        self.hyps_mask = hyps_mask
        return hyps_mask

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

