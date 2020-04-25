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
from sys import stdout
from os import devnull

from flare.util import  element_to_Z


class ParameterMasking():
    """
    A helper class to construct the hyps_mask dictionary for AtomicEnvironment
    and GaussianProcess

    examples:
        pm = ParameterMasking(species=['Cu', 'C', 'H', 'O'],
                              bonds=[['*', '*'], ['Cu','Cu']],
                              triplets=[['*', '*', '*'], ['Cu','Cu', 'Cu']],
                              parameters={'bond0':[1, 0.5, 1], 'bond1':[2, 0.2, 2],
                                    'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                                    'cutoff3b':1},
                              constraints={'bond0':[False, True]})
        hm = pm.hyps_mask
        hyps = hm['hyps']
        cutoffs = hm['cutoffs']

    In this example, four atomic species are involved. There are many kinds
    of bonds and triplets. But we only want to use eight different sigmas
    and lengthscales.

    In order to do so, we first define all the bonds to be group "bond0", by
    listing "*-*" as the first element in the bond argument. The second
    element Cu-Cu is then defined to be group "bond1". Note that the order
    matters here. The later element overrides the ealier one. If
    bonds=[['Cu', 'Cu'], ['*', '*']], then all bonds belong to group "bond1".

    Similarly, Cu-Cu-Cu is defined as triplet1, while all remaining ones
    are left as triplet0.

    The hyperpameters for each group is listed in the order of
    [sig, ls, cutoff] in the parameters argument.  So in this example,
    Cu-Cu interaction will use [2, 0.2, 2] as its sigma, length scale, and
    cutoff.

    For triplet, the parameter arrays only come with two elements. So there
    is no cutoff associated with triplet0 or triplet1; instead, a universal
    cutoff is used, which is defined as 'cutoff3b'.

    The constraints argument define which hyper-parameters will be optimized.
    True for optimized and false for being fixed.

    There are more examples see tests/test_mask_helper.py

    """
    def __init__(self, hyps_mask=None, species=None, bonds=None,
                 triplets=None, cut3b=None, mb=None, parameters=None,
                 constraints={}, verbose=False):
        """ Initialization function

        :param hyps_mask: Not implemented yet
        :type hyps_mask: dict
        :param species: list or dictionary that define specie groups
        :type species: [dict, list]
        :param bonds: list or dictionary that define bond groups
        :type bonds: [dict, list]
        :param triplets: list or dictionary that define triplet groups
        :type triplets: [dict, list]
        :param cut3b: list or dictionary that define 3b-cutoff groups
        :type cut3b: [dict, list]
        :param mb: list or dictionary that define many-body groups
        :type mb: [dict, list]
        :param parameters: dictionary of parameters
        :type parameters: dict
        :param constraints: whether the hyperparmeters are optimized (True) or not (False)
        :constraints: dict
        :param verbose: print the process to screen
        :type verbose: bool

        See format of species, bonds, triplets, cut3b, mb in list_groups() function.

        See format of parameters and constraints in list_parameters() function.

        """

        if (verbose):
            self.fout = stdout
        else:
            self.fout = open(devnull, 'w')

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
        self.all_cutoff = {}
        self.hyps_sig = {}
        self.hyps_ls = {}
        self.hyps_opt = {}
        self.opt = {'noise':True}
        self.mask = {}
        self.cutoff_list = {}
        self.noise = 0.05

        self.cutoffs_array = [0, 0, 0]
        self.hyps = None

        if (species is not None):
            self.list_groups('specie', species)
            if (bonds is not None):
                self.list_groups('bond', bonds)
            if (triplets is not None):
                self.list_groups('triplet', triplets)
            if (cut3b is not None):
                self.list_groups('cut3b', cut3b)
            if (mb is not None):
                self.list_groups('mb', mb)
            if (parameters is not None):
                self.list_parameters(parameters, constraints)
            try:
                self.hyps_mask = self.generate_dict()
            except:
                print("more parameters needed to generate the hypsmask", file=self.fout)

    def list_parameters(self, parameter_dict, constraints={}):
        """Define many groups of parameters

        :param parameter_dict: dictionary of all parameters
        :type parameter_dict: dict
        :param constraints: dictionary of all constraints
        :type constraints: dict

        example: parameter_dict={"name":[sig, ls, cutoffs], ...}
                 constraints={"name":[True, False, False], ...}

        The name of parameters can be the group name previously defined in
        define_group or list_groups function. Aside from the group name,
        "noise", "cutoff2b", "cutoff3b", and "cutoffmb" are reserved for
        noise parmater and universal cutoffs.

        For non-reserved keys, the value should be a list of 2-3 elements,
        correspond to the sigma, lengthscale (and cutoff if the third one
        is defined). For reserved keys, the value should be a scalar.

        The parameter_dict and constraints should uses the same set of keys.
        The keys in constraints but not in parameter_dict will be ignored.

        The value in the constraints can be either a single bool, which apply
        to all parameters, or list of bools that apply to each parameter.
        """

        for name in parameter_dict:
            self.set_parameters(name, parameter_dict[name], constraints.get(name, True))

    def list_groups(self, group_type, definition_list):
        """define groups in batches.

        Args:

        group_type (str): "specie", "bond", "triplet", "cut3b", "mb"
        definition_list (list, dict): list of elements

        This function runs define_group in batch. Please first read
        the manual of define_group.

        If the definition_list is a list, it is equivalent to
        executing define_group through the definition_list.

        | for all terms in the list:
        |     define_group(group_type, group_type+'n', the nth term in the list)

        So the first bond defined will be group bond0, second one will be
        group bond1. For specie, it will define all the listed elements as
        groups with only one element with their original name.

        If the definition_list is a dictionary, it is equivalent to

        | for k, v in the dict:
        |     define_group(group_type, k, v)

        It is not recommended to use the dictionary mode, especially when
        the group definitions are conflicting with each other. There is no
        guarantee that the looping order is the same as you want.

        Unlike define_group, it can only be called once for each
        group_type, and not after any define_group calls.

        """
        if (group_type == 'specie'):
            if (len(self.all_group_names['specie'])>0):
                raise RuntimeError("this function has to be run "\
                        "before any define_group")
            if (isinstance(definition_list, list)):
                for ele in definition_list:
                    if isinstance(ele, list):
                        self.define_group('specie', ele, ele)
                    else:
                        self.define_group('specie', ele, [ele])
            elif (isinstance(elemnt_list, dict)):
                for ele in definition_list:
                    self.define_group('specie', ele, definition_list[ele])
            else:
                raise RuntimeError("type unknown")
        else:
            if (len(self.all_group_names['specie'])==0):
                raise RuntimeError("this function has to be run "\
                        "before any define_group")
            if (isinstance(definition_list, list)):
                ngroup = len(definition_list)
                for idg in range(ngroup):
                    self.define_group(group_type, f"{group_type}{idg}",
                            definition_list[idg])
            elif (isinstance(definition_list, dict)):
                for name in definition_list:
                    if (isinstance(definition_list[name][0], list)):
                        for ele in definition_list[name]:
                            self.define_group(group_type, name, ele)
                    else:
                        self.define_group(group_type, name, definition_list[name])

    def define_group(self, group_type, name, element_list, parameters=None, atomic_str=False):
        """Define specie/bond/triplet/3b cutoff/manybody group

        Args:
        group_type (str): "specie", "bond", "triplet", "cut3b", "mb"
        name (str): the name use for indexing. can be anything but "*"
        element_list (list): list of elements
        parameters (list): corresponding parameters for this group
        atomic_str (bool): whether the element in element_list is
                           group name or periodic table element name.

        The function is helped to define different groups for specie/bond/triplet
        /3b cutoff/manybody terms. This function can be used for many times.
        The later one always overrides the former one.

        The name of the group has to be unique string (but not "*"), that
        define a group of species or bonds, etc. If the same name is used,
        in two function calls, the definitions of the group will be merged.
        Both calls will be effective.

        element_list has to be a list of atomic elements, or a list of
        specie group names (which should be defined in previous calls), or "*".
        "*" will loop the function over all previously defined species.
        It has to be two elements for bond/3b cutoff/manybody term, or
        three elements for triplet. For specie group definition, it can be
        as many elements as you want.

        If multiple define_group calls have conflict with element, the later one
        has higher priority. For example, bond 1-2 are defined as group1 in
        the first call, and as group2 in the second call. In the end, the bond
        will be left as group2.

        Example 1:

            define_group('specie', 'water', ['H', 'O'])
            define_group('specie', 'salt', ['Cl', 'Na'])

        They define H and O to be group water, and Na and Cl to be group salt.

        Example 2.1:

            define_group('bond', 'in-water', ['H', 'H'], atomic_str=True)
            define_group('bond', 'in-water', ['H', 'O'], atomic_str=True)
            define_group('bond', 'in-water', ['O', 'O'], atomic_str=True)

        Example 2.2:
            define_group('bond', 'in-water', ['water', 'water'])

        The 2.1 is equivalent to 2.2.

        Example 3.1:

            define_group('specie', '1', ['H'])
            define_group('specie', '2', ['O'])
            define_group('bond', 'Hgroup', ['H', 'H'], atomic_str=True)
            define_group('bond', 'Hgroup', ['H', 'O'], atomic_str=True)
            define_group('bond', 'OO', ['O', 'O'], atomic_str=True)

        Example 3.2:

            define_group('specie', '1', ['H'])
            define_group('specie', '2', ['O'])
            define_group('bond', 'Hgroup', ['H', '*'], atomic_str=True)
            define_group('bond', 'OO', ['O', 'O'], atomic_str=True)

        Example 3.3:

            list_groups('specie', ['H', 'O'])
            define_group('bond', 'Hgroup', ['H', '*'])
            define_group('bond', 'OO', ['O', 'O'])

        Example 3.4:

            list_groups('specie', ['H', 'O'])
            define_group('bond', 'OO', ['*', '*'])
            define_group('bond', 'Hgroup', ['H', '*'])

        3.1 to 3.4 are all equivalent.
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
                print(f"Element {ele} will be defined as group {name}", file=self.fout)
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
                                      f"{self.all_group_names[idx]} is affected", file=self.fout)
                    else:
                        gid += [self.all_group_names['specie'].index(ele_name)]

                for ele in self.all_members[group_type]:
                    if set(gid) == set(ele):
                        print(f"Warning: the definition of {group_type} {ele} will be overriden", file=self.fout)
                self.groups[group_type][groupid].append(gid)
                self.all_members[group_type].append(gid)
                print(f"{group_type} {gid} will be defined as group {name}", file=self.fout)
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
        """Set the parameters for certain group

        :param name: name of the patermeters
        :type name: str
        :param parameters: the sigma, lengthscale, and cutoff of each group.
        :type parameters: list
        :param opt: whether to optimize the parameter or not
        :type opt: bool, list

        The name of parameters can be the group name previously defined in
        define_group or list_groups function. Aside from the group name,
        "noise", "cutoff2b", "cutoff3b", and "cutoffmb" are reserved for
        noise parmater and universal cutoffs.

        The parameter should be a list of 2-3 elements, for sigma,
        lengthscale (and cutoff if the third one is defined).

        The optimization flag can be a single bool, which apply to all
        parameters, or list of bools that apply to each parameter.
        """

        if (name == 'noise'):
            self.noise = parameters
            self.opt['noise'] = opt
            return

        if (name in ['cutoff2b', 'cutoff3b', 'cutoffmb']):
            name_map = {'cutoff2b':0, 'cutoff3b':1, 'cutoffmb':2}
            self.cutoffs_array[name_map[name]] = parameters
            return

        if (isinstance(opt, bool)):
            opt = [opt, opt, opt]
        if ('cut3b' not in name):
            if (name in self.sigma):
                print(f"Warning, the sig, ls of group {name} is overriden", file=self.fout)
            self.sigma[name] = parameters[0]
            self.ls[name] = parameters[1]
            self.opt[name+'sig'] = opt[0]
            self.opt[name+'ls'] = opt[1]
            print(f"Parameters for group {name} will be set as "\
                  f"sig={parameters[0]} ({opt[0]}) "\
                  f"ls={parameters[1]} ({opt[1]})", file=self.fout)
        if (len(parameters)>2):
            if (name in self.all_cutoff):
                print(f"Warning, the cutoff of group {name} is overriden", file=self.fout)
            self.all_cutoff[name] = parameters[2]


    def summarize_group(self, group_type):
        """Sort and combine all the previous definition to internal varialbes

        Args:

        group_type (str): species, bond, triplet, cut3b, mb
        """
        aeg = self.all_group_names[group_type]
        nspecie = self.n['specie']
        if (group_type == "specie"):
            self.nspecie = nspecie
            if (nspecie==1):
                return
            self.specie_mask = np.ones(118, dtype=np.int)*(self.n['specie']-1)
            for idt in range(self.n['specie']):
                for ele in self.groups['specie'][idt]:
                    atom_n = element_to_Z(ele)
                    self.specie_mask[atom_n] = idt
                    print(f"elemtn {ele} is defined as type {idt} with name "\
                            f"{aeg[idt]}", file=self.fout)
            print(f"All the remaining elements are left as type {idt}", file=self.fout)
        elif (group_type in ['bond', 'cut3b', 'mb']):
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
                          f"with name {name}", file=self.fout)
                if (group_type != 'cut3b'):
                    sig = self.sigma[name]
                    ls = self.ls[name]
                    self.hyps_sig[group_type] += [sig]
                    self.hyps_ls[group_type] += [ls]
                    self.hyps_opt[group_type] += [self.opt[name+'sig']]
                    self.hyps_opt[group_type] += [self.opt[name+'ls']]
                    print(f"   using hyper-parameters of {sig} {ls}", file=self.fout)
            print(f"All the remaining elements are left as type {idt}", file=self.fout)

            name_map = {'bond':0, 'cut3b':1, 'mb':2}

            self.cutoff_list[group_type] = []
            cut_define = np.zeros(self.n[group_type], dtype=bool)
            allcut = [self.cutoffs_array[name_map[group_type]]]
            for idt in range(self.n[group_type]):
                if (aeg[idt] in self.all_cutoff):
                    cut_define[idt] = True
                    allcut += [self.all_cutoff[aeg[idt]]]

            if cut_define.all():
                self.cutoff_list[group_type] = []
                for idt in range(self.n[group_type]):
                    self.cutoff_list[group_type] += [self.all_cutoff[aeg[idt]]]
                print("Different cutoffs were also defined", self.cutoff_list[group_type], file=self.fout)
                self.cutoffs_array[name_map[group_type]] = np.max(self.cutoff_list[group_type])
            else:
                if cut_define.any():
                    print("There were some cutoff defined, but not all of them", file=self.fout)
                    self.cutoffs_array[name_map[group_type]] = np.max(allcut)
                if (self.cutoffs_array[name_map[group_type]] <=0):
                    raise RuntimeError(f"cutoffs for {group_type} is undefined")
        elif (group_type == "triplet"):
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
                            f"{name}", file=self.fout)
                sig = self.sigma[name]
                ls = self.ls[name]
                self.hyps_sig[group_type] += [sig]
                self.hyps_ls[group_type] += [ls]
                self.hyps_opt[group_type] += [self.opt[name+'sig']]
                self.hyps_opt[group_type] += [self.opt[name+'ls']]
                print(f"   using hyper-parameters of {sig} {ls}", file=self.fout)
            print(f"all the remaining elements are left as type {idt}", file=self.fout)
            if (self.cutoffs_array[1] == 0):
                cut_define = False
                allcut = []
                for idt in range(self.n[group_type]):
                    if (aeg[idt] in self.all_cutoff):
                        cut_define = True
                        allcut += [self.all_cutoff[aeg[idt]]]
                if cut_define:
                    self.cutoffs_array[1] = np.max(allcut)
                else:
                    raise RuntimeError(f"cutoffs for {group_type} is undefined")
        else:
            pass

    def generate_dict(self):
        """Dictionary representation of the GP model."""
        if self.n['specie'] < 2:
            print("only one type of elements was defined. return None", file=self.fout)
            hyps_mask = None
        else:
            self.summarize_group('specie')
            self.summarize_group('bond')
            self.summarize_group('cut3b')
            self.summarize_group('triplet')
            self.summarize_group('mb')
            hyps_mask = {}
            hyps_mask['nspecie'] = self.n['specie']
            hyps_mask['specie_mask'] = self.specie_mask
            hyps = []
            hyps_label = []
            opt = []
            for group in ['bond', 'triplet', 'mb']:
                if (self.n[group]>=1):
                    hyps_mask['n'+group] = self.n[group]
                    hyps_mask[group+'_mask'] = self.mask[group]
                    hyps += [self.hyps_sig[group]]
                    hyps += [self.hyps_ls[group]]
                    opt += [self.hyps_opt[group]]
                    aeg = self.all_group_names[group]
                    for idt in range(self.n[group]):
                        hyps_label += ['sig_'+aeg[idt]]
                    for idt in range(self.n[group]):
                        hyps_label += ['ls_'+group]
            opt += [self.opt['noise']]
            hyps_mask['original'] = np.hstack(hyps)
            hyps_mask['original'] = np.hstack([hyps_mask['original'], self.noise])
            hyps_label += ['Noise']
            hyps_mask['original'] = np.array(hyps_mask['original'], dtype=np.float)
            hyps_mask['hyps_label']=hyps_label
            opt = np.hstack(opt)
            hyps_mask['train_noise'] = self.opt['noise']
            if (not opt.all()):
                nhyps = len(hyps_mask['original'])
                mapping = []
                hyps_mask['hyps_label']=[]
                for i in range(nhyps):
                    if (opt[i]):
                        mapping += [i]
                        hyps_mask['hyps_label'] += [hyps_label[i]]
                newhyps = hyps_mask['original'][mapping]
                hyps_mask['map'] = np.array(mapping, dtype=np.int)
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
        if (self.cutoffs_array[2]>0):
            hyps_mask['cutoffs'] = self.cutoffs_array
        else:
            hyps_mask['cutoffs'] = self.cutoffs_array[:2]
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
        hyps_mask['specie_mask'] = nparray(hyps_mask['specie_mask'], dtype=np.int)

        if 'nbond' in hyps_mask:
            n2b = hyps_mask['nbond']
            assert n2b>0
            assert isinstance(n2b, int)
            hyps_mask['bond_mask'] = nparray(hyps_mask['bond_mask'], dtype=np.int)
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
            hyps_mask['triplet_mask'] = nparray(hyps_mask['triplet_mask'], dtype=np.int)
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
        else:
            nmb = 1
            hyps_mask['mb_mask'] = np.zeros(nspecie**2, dtype=np.int)

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
