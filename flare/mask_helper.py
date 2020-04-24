import time
import math
import pickle
import inspect
import json

import numpy as np
from copy import deepcopy
from numpy.random import random
from numpy import array as nparray
from numpy import max as npmax
from typing import List, Callable, Union
from warnings import warn

from flare.util import  element_to_Z


class ParameterMasking():
    """
    A helper class to construct the hyps_mask dictionary for AtomicEnvironment
    and GaussianProcess
    """
    def __init__(self, hyps_mask=None, specie=None, bond=None,
                 triplet=None, cut3b=None, mb=None, para=None,
                 constraint={}):
        self.n = {}
        self.groups = {}
        self.all_members = {}
        self.all_group_names = {}
        self.all_names = []
        self.all_types = ['specie', 'bond', 'triplet', 'mb', 'cut3b']

        for group_type in self.all_types:
            self.n[group_type] = 0
            self.groups[group_type] = []
            self.all_members[group_type] = []
            self.all_group_names[group_type] = []
        self.sigma = {}
        self.ls = {}
        self.cutoff = {}
        self.hyps_sig = {}
        self.hyps_ls = {}
        self.hyps_opt = {}
        self.opt = {'noise':True}
        self.mask = {}
        self.cutoff_list = {}
        self.noise = 0.05

        self.cutoffs = {}
        self.hyps = None

        if (specie is not None):
            self.list_sweeping('specie', specie)
            if (bond is not None):
                self.list_sweeping('bond', bond)
            if (triplet is not None):
                self.list_sweeping('triplet', triplet)
            if (cut3b is not None):
                self.list_sweeping('cut3b', cut3b)
            if (mb is not None):
                self.list_sweeping('mb', mb)
            if (para is not None):
                self.list_parameters(para, constraint)
            self.hyps_mask = self.generate_dict()

    def list_parameters(self, para_list, constraint={}):
        for name in para_list:
            self.set_parameters(name, para_list[name], constraint.get(name, True))

    def list_sweeping(self, group_type, element_list):
        if (group_type == 'specie'):
            if (len(self.all_group_names['specie'])>0):
                raise RuntimeError("this function has to be run "\
                        "before any define_group")
            if (isinstance(element_list, list)):
                for ele in element_list:
                    if isinstance(ele, list):
                        self.define_group('specie', ele, ele)
                    else:
                        self.define_group('specie', ele, [ele])
            elif (isinstance(elemnt_list, dict)):
                for ele in element_list:
                    self.define_group('specie', ele, element_list[ele])
            else:
                raise RuntimeError("type unknown")
        else:
            if (len(self.all_group_names['specie'])==0):
                raise RuntimeError("this function has to be run "\
                        "before any define_group")
            if (isinstance(element_list, list)):
                ngroup = len(element_list)
                for idg in range(ngroup):
                    self.define_group(group_type, f"{group_type}{idg}",
                            element_list[idg])
            elif (isinstance(element_list, dict)):
                for name in element_list:
                    if (isinstance(element_list[name][0], list)):
                        for ele in element_list[name]:
                            self.define_group(group_type, name, ele)
                    else:
                        self.define_group(group_type, name, element_list[name])

    def define_group(self, group_type, name, element_list, parameters=None, atomic_str=False):
        """
        group_type (str): species, bond, triplet, cut3b, mb
        name (str): the name use for indexing
        element_list (list):
        """

        if (name == '*'):
            raise ValueError("* is reserved for substitution, cannot be used "\
                    "as a group name")

        if (group_type != 'specie'):
            fullname = group_type + name
            exclude_list = deepcopy(self.all_types)
            ide = exclude_list.index(group_type)
            exclude_list.pop(ide)
            for gt in exclude_list:
                if (name in self.all_group_names[gt]):
                    raise ValueError("group name has to be unique across all types. "\
                                     f"{name} is found in type {gt}")
        # else:
        #     fullname = name

        if (name in self.all_group_names[group_type]):
            groupid = self.all_group_names[group_type].index(name)
        else:
            groupid = self.n[group_type]
            self.all_group_names[group_type].append(name)
            self.groups[group_type].append([])
            self.n[group_type] += 1

        if (group_type == 'specie'):
            for ele in element_list:
                assert ele not in self.all_members['specie'], \
                        "The element has already been defined"
                self.groups['specie'][groupid].append(ele)
                self.all_members['specie'].append(ele)
                print(f"Element {ele} will be defined as group {name}")
        else:
            if (len(self.all_group_names['specie'])==0):
                raise RuntimeError("The atomic species have to be"
                        "defined in advance")
            if ("*" not in element_list):
                gid = []
                for ele_name in element_list:
                    if (atomic_str):
                        for idx in range(self.n['specie']):
                            if (ele_name in self.groups['specie'][idx]):
                                gid += [idx]
                                print(f"Warning: Element {ele_name} is used for "\
                                      f"definition, but the whole group "\
                                      f"{self.all_group_names[idx]} is affected")
                    else:
                        gid += [self.all_group_names['specie'].index(ele_name)]

                for ele in self.all_members[group_type]:
                    if set(gid) == set(ele):
                        print(f"Warning: the definition of {group_type} {ele} will be overriden")
                self.groups[group_type][groupid].append(gid)
                self.all_members[group_type].append(gid)
                print(f"{group_type} {gid} will be defined as group {name}")
                if (parameters is not None):
                    self.set_parameters(name, parameters)
            else:
                one_star_less = deepcopy(element_list)
                idstar = element_list.index('*')
                one_star_less.pop(idstar)
                for sub in self.all_group_names['specie']:
                    # print("head of replacement", group_type, name,
                    #       non_star_element +[sub])
                    self.define_group(group_type, name,
                            one_star_less +[sub], parameters=parameters, atomic_str=atomic_str)

    def set_parameters(self, name, parameters, opt=True):

        if (name == 'noise'):
            self.noise = parameters
            self.opt['noise'] = opt
            return

        if (isinstance(opt, bool)):
            opt = [opt, opt, opt]
        if ('cut3b' not in name):
            if (name in self.sigma):
                print(f"Warning, the sig, ls of group {name} is overriden")
            self.sigma[name] = parameters[0]
            self.ls[name] = parameters[1]
            self.opt[name+'sig'] = opt[0]
            self.opt[name+'ls'] = opt[1]
            print(f"Parameters for group {name} will be set as "\
                  f"sig={parameters[0]} ({opt[0]}) "\
                  f"ls={parameters[1]} ({opt[1]})")
        if (len(parameters)>2):
            if (name in self.cutoff):
                print(f"Warning, the cutoff of group {name} is overriden")
            self.cutoff[name] = parameters[2]


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
                for ele in self.groups['specie'][idt]:
                    atom_n = element_to_Z(ele)
                    self.specie_mask[atom_n] = idt
                    print(f"elemtn {ele} is defined as type {idt} with name "\
                            f"{aeg[idt]}")
            print(f"All the remaining elements are left as type {idt}")
        elif (group_type in ['bond', 'cut3b', 'mb']):
            nspecie = self.n['specie']
            if (self.n[group_type] == 0):
                return
            self.mask[group_type] = np.ones(nspecie**2, dtype=np.int)*(self.n[group_type]-1)
            self.hyps_sig[group_type] = []
            self.hyps_ls[group_type] = []
            self.hyps_opt[group_type] = []
            for idt in range(self.n[group_type]):
                name = aeg[idt]
                for bond in self.groups[group_type][idt]:
                    g1 = bond[0]
                    g2 = bond[1]
                    self.mask[group_type][g1+g2*nspecie] = idt
                    self.mask[group_type][g2+g1*nspecie] = idt
                    s1 = self.groups['specie'][g1]
                    s2 = self.groups['specie'][g2]
                    print(f"{group_type} {s1} - {s2} is defined as type {idt} "\
                          f"with name {name}")
                if (group_type != 'cut3b'):
                    sig = self.sigma[name]
                    ls = self.ls[name]
                    self.hyps_sig[group_type] += [sig]
                    self.hyps_ls[group_type] += [ls]
                    self.hyps_opt[group_type] += [self.opt[name+'sig']]
                    self.hyps_opt[group_type] += [self.opt[name+'ls']]
                    print(f"   using hyper-parameters of {sig} {ls}")
            print(f"All the remaining elements are left as type {idt}")
            self.cutoff_list[group_type] = []
            cut_define = np.zeros(self.n[group_type], dtype=bool)
            max_cutoff = 0
            for idt in range(self.n[group_type]):
                name = aeg[idt]
                if (name in self.cutoff):
                    cut_define[idt] = True
                    if (self.cutoff[name] > max_cutoff):
                        max_cutoff = self.cutoff[name]
            self.cutoffs[group_type] = max_cutoff
            if cut_define.all():
                self.cutoff_list[group_type] = []
                for idt in range(self.n[group_type]):
                    self.cutoff_list[group_type] += [self.cutoff[aeg[idt]]]
                print("Different cutoffs were also defined", self.cutoff_list[group_type])
            elif cut_define.any():
                print("There were some cutoff defined, but not all of them")
        elif (group_type == "triplet"):
            nspecie = self.n['specie']
            self.ntriplet = self.n['triplet']
            if (self.ntriplet == 0):
                return
            self.mask[group_type] = np.ones(nspecie**3, dtype=np.int)*(self.ntriplet-1)
            self.hyps_sig[group_type] = []
            self.hyps_ls[group_type] = []
            self.hyps_opt[group_type] = []
            for idt in range(self.n['triplet']):
                name = aeg[idt]
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
                    s1 = self.groups['specie'][g1]
                    s2 = self.groups['specie'][g2]
                    s3 = self.groups['specie'][g3]
                    print(f"triplet {s1} - {s2} - {s3} is defined as type {idt} with name "\
                            f"{name}")
                sig = self.sigma[name]
                ls = self.ls[name]
                self.hyps_sig[group_type] += [sig]
                self.hyps_ls[group_type] += [ls]
                self.hyps_opt[group_type] += [self.opt[name+'sig']]
                self.hyps_opt[group_type] += [self.opt[name+'ls']]
                print(f"   using hyper-parameters of {sig} {ls}")
            print(f"all the remaining elements are left as type {idt}")
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
            opt = []
            for group in ['bond', 'triplet', 'mb']:
                if (self.n[group]>=1):
                    hyps_mask['n'+group] = self.n[group]
                    hyps_mask[group+'_mask'] = self.mask[group]
                    hyps += [self.hyps_sig[group]]
                    hyps += [self.hyps_ls[group]]
                    opt += [self.hyps_opt[group]]
            opt += [self.opt['noise']]
            hyps_mask['original'] = np.hstack(hyps)
            hyps_mask['original'] = np.hstack([hyps_mask['original'], self.noise])
            opt = np.hstack(opt)
            hyps_mask['train_noise'] = self.opt['noise']
            if (not opt.all()):
                nhyps = len(hyps_mask['original'])
                mapping = []
                for i in range(nhyps):
                    if (opt[i]):
                        mapping += [i]
                newhyps = hyps_mask['original'][mapping]
                hyps_mask['map'] = np.hstack(mapping)
            else:
                newhyps = hyps_mask['original']
            hyps_mask['hyps'] = newhyps

            if len(self.cutoff_list.get('bond', []))>0:
                hyps_mask['cutoff_2b'] = self.cutoff_list['bond']
            if len(self.cutoff_list.get('cut3b', []))>0:
                hyps_mask['cutoff_3b'] = self.cutoff_list['cut3b']
                hyps_mask['ncut3b'] = self.n['cut3b']
                hyps_mask['cut3b_mask'] = self.mask['cut3b']
            if len(self.cutoff_list.get('mb', []))>0:
                hyps_mask['cutoff_mb'] = self.cutoff_list['mb']

        self.hyps_mask = hyps_mask
        hyps_mask['cutoffs'] = self.cutoffs
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

    @staticmethod
    def mask2cutoff(cutoffs, cutoffs_mask):

        if isinstance(cutoffs, dict):
            scalar_cutoff_2 = cutoffs.get('bond', 0)
            scalar_cutoff_3 = cutoffs.get('cut3b', 0)
            scalar_cutoff_mb = cutoffs.get('mb', 0)
        else:
            ncutoffs = len(cutoffs)
            scalar_cutoff_2 = cutoffs[0]
            scalar_cutoff_3 = 0
            scalar_cutoff_mb = 0
            if (ncutoffs > 1):
                scalar_cutoff_3 = cutoffs[1]
            if (ncutoffs > 2):
                scalar_cutoff_3 = cutoffs[2]

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

        assert n2b >=1
        assert n3b >=1
        assert nmb >=1

        assert isinstance(n2b, int)
        assert isinstance(n3b, int)
        assert isinstance(nmb, int)

        if cutoff_2b is not None:
            scalar_cutoff_2 = np.max(cutoff_2b)
            if (n2b > 1):
                assert bond_mask is not None
        else:
            n2b == 1

        if cutoff_3b is not None:
            scalar_cutoff_3 = np.max(cutoff_3b)
            assert scalar_cutoff_3 <= scalar_cutoff_2, \
                "2b cutoff has to be larger than 3b cutoff"
            if (n3b > 1):
                assert cut3b_mask is not None
        else:
            n3b == 1

        if cutoff_mb is not None:
            scalar_cutoff_mb = np.max(cutoff_mb)
            if (nmb > 1):
                assert mb_mask is not None
        # # TO DO, once the mb function is updated to use the bond_array_2
        # # this block should be activated.
        # assert scalar_cutoff_mb <= scalar_cutoff_2, \
        #         "mb cutoff has to be larger than mb cutoff"
        return scalar_cutoff_2, scalar_cutoff_3, scalar_cutoff_mb, \
            cutoff_2b, cutoff_3b, cutoff_mb, \
            nspecie, n2b, n3b, nmb, specie_mask, bond_mask, cut3b_mask, mb_mask
