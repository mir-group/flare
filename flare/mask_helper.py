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
        self.groups = {}
        self.all_members = {}
        self.all_group_names = {}

        for group_type in ['specie', 'bond', 'triplet', 'cut3b', 'mb']:
            self.n[group_type] = 0
            self.groups[group_type] = []
            self.all_members[group_type] = []
            self.all_group_names[group_type] = []
        self.sigma = {'bond':{}, 'triplet':{}, 'mb':{}}
        self.ls = {'bond':{}, 'triplet':{}, 'mb': {}}
        self.cutoff = {'bond':{}, 'cut3b':{}, 'mb': {}}
        self.hyps_sig = {}
        self.hyps_ls = {}
        self.mask = {}
        self.cutoff_list = {}

    def define_group(self, group_type, name, element_list, atomic_str=False):
        """
        group_type (str): species, bond, triplet, cut3b, mb
        name (str): the name use for indexing
        element_list (list):
        """

        if (name in self.all_group_names[group_type]):
            groupid = self.all_group_names[group_type].index(name)
        else:
            groupid = self.n[group_type]
            self.all_group_names[group_type].append(name)
            self.groups[group_type].append([])
            self.n[group_type] += 1

        if (group_type is 'specie'):
            for ele in element_list:
                assert ele not in self.all_members['specie'], \
                        "the element has already been defined"
                self.groups['specie'][groupid].append(ele)
                self.all_members['specie'].append(ele)
        else:
            gid = []
            for ele_name in element_list:
                if (atomic_str):
                    for idx in range(self.n['specie']):
                        if (ele_name in self.groups['specie'][idx]):
                            gid += [idx]
                            print(f"Define {group_type}: Element {ele_name}"\
                                  f"is in group {self.all_group_names[idx]}")
                else:
                    print(self.groups['specie'])
                    print(self.all_group_names['specie'])
                    gid += [self.all_group_names['specie'].index(ele_name)]

            for ele in self.all_members[group_type]:
                assert set(gid) != set(ele), \
                    f"the {group_type} {ele} has already been defined"

            self.groups[group_type][groupid].append(gid)
            self.all_members[group_type].append(gid)

    def define_parameters(self, group_type, name, sig, ls, cutoff=None):

        if (group_type != 'cut3b'):
            self.sigma[group_type][name] = sig
            self.ls[group_type][name] = ls
        if (cutoff is not None):
            self.cutoff[group_type][name] = cutoff


    def print_group(self, group_type):
        """
        group_type (str): species, bond, triplet, cut3b, mb
        name (str): the name use for indexing
        element_list (list):
        """
        aeg = self.all_group_names[group_type]
        if (group_type == "specie"):
            self.nspecie = self.n['specie']
            self.specie_mask = np.ones(118, dtype=np.int)*(self.n['specie']-1)
            for idt in range(self.n['specie']):
                if (aeg[idt] == "*"):
                    for i in range(118):
                        if self.specie_mask[i] > idt:
                            self.specie_mask[i] = idt
                            print(f"element [i] is defined as type {idt} with name"\
                                f"aeg[idt]")
                else:
                    for ele in self.groups['specie'][idt]:
                        atom_n = element_to_Z(ele)
                        self.specie_mask[atom_n] = idt
                        print(f"elemtn {ele} is defined as type {idt} with name"\
                                f"aeg[idt]")
            print("all the remaining elements are left as type 0")
        elif (group_type in ['bond', 'cut3b', 'mb']):
            nspecie = self.n['specie']
            self.mask[group_type] = np.ones(nspecie**2, dtype=np.int)*(self.n[group_type]-1)
            self.hyps_sig[group_type] = []
            self.hyps_ls[group_type] = []
            for idt in range(self.n[group_type]):
                name = aeg[idt]
                if (aeg[idt] == "*"):
                    for i in range(nspecie**2):
                        if (self.mask[group_type][i]>idt):
                            self.mask[group_type][i] = idt
                else:
                    for bond in self.groups[group_type][idt]:
                        g1 = bond[0]
                        g2 = bond[1]
                        self.mask[group_type][g1+g2*nspecie] = idt
                        self.mask[group_type][g2+g1*nspecie] = idt
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
            nspecie = self.n['specie']
            self.ntriplet = self.n['triplet']
            self.mask[group_type] = np.ones(nspecie**3, dtype=np.int)*(self.ntriplet-1)
            self.hyps_sig[group_type] = []
            self.hyps_ls[group_type] = []
            for idt in range(self.n['triplet']):
                if (aeg[idt] == "*"):
                    for i in range(nspecie**3):
                        if (self.mask[group_type][i]>idt):
                            self.mask[group_type][i] = idt
                else:
                    for triplet in self.groups['triplet'][idt]:
                        g1 = triplet[0]
                        g2 = triplet[1]
                        g3 = triplet[2]
                        self.mask[group_type][g1+g2*nspecie+g3*nspecie**2] = idt
                        self.mask[group_type][g1+g3*nspecie+g2*nspecie**2] = idt
                        self.mask[group_type][g2+g1*nspecie+g3*nspecie**2] = idt
                        self.mask[group_type][g2+g3*nspecie+g1*nspecie**2] = idt
                        self.mask[group_type][g3+g1*nspecie+g2*nspecie**2] = idt
                        self.mask[group_type][g3+g2*nspecie+g1*nspecie**2] = idt
                        print(f"triplet {triplet} is defined as type {idt} with name"\
                                "self.all_group_names[group_type][idt]")
                self.hyps_sig[group_type] += [self.sigma['triplet'][self.all_group_names[group_type][idt]]]
                self.hyps_ls[group_type] += [self.ls['triplet'][self.all_group_names[group_type][idt]]]
                print(f"   using hyper-parameters of {self.hyps_sig[group_type][-1]}"\
                        "{ self.hyps_ls[group_type][-1]}")
        else:
            pass

    def generate_dict(self):
        """Dictionary representation of the GP model."""
        if self.n['specie'] < 2:
            print("only one type of elements was defined. return None")
            hyps_mask = None
        else:
            self.print_group('specie')
            self.print_group('bond')
            self.print_group('triplet')
            self.print_group('cut3b')
            self.print_group('mb')
            hyps_mask = {}
            hyps_mask['nspecie'] = self.n['specie']
            hyps_mask['specie_mask'] = self.specie_mask
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

        assert 'nspecie' in hyps_mask, "nspecie key missing in " \
                                                 "hyps_mask dictionary"
        assert 'specie_mask' in hyps_mask, "specie_mask key " \
                                                     "missing " \
                                                     "in hyps_mask dicticnary"

        nspecie = hyps_mask['nspecie']
        hyps_mask['specie_mask'] = nparray(hyps_mask['specie_mask'], dtype=int)

        if 'nbond' in hyps_mask:
            n2b = hyps_mask['nbond']
            assert n2b>0
            assert isinstance(n2b, int)
            hyps_mask['bond_mask'] = nparray(hyps_mask['bond_mask'], dtype=int)
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
            assert n3b>0
            assert isinstance(n3b, int)
            hyps_mask['triplet_mask'] = nparray(hyps_mask['triplet_mask'], dtype=int)
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
            assert nmb>0
            assert isinstance(nmb, int)
            hyps_mask['mb_mask'] = nparray(hyps_mask['mb_mask'], dtype=int)
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
        else:
            nmb = 1
            hyps_mask['mb_mask'] = np.zeros(nspecie**2)

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

