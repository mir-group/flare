import time
import math
import pickle
import inspect
import json
import logging

import numpy as np
from copy import deepcopy
from numpy import array as nparray
from numpy import max as npmax
from typing import List, Callable, Union
from warnings import warn
from sys import stdout
from os import devnull

from flare.parameters import Parameters
from flare.utils.element_coder import element_to_Z, Z_to_element


class ParameterHelper():
    """
    A helper class to construct the hyps_mask dictionary for AtomicEnvironment
    , GaussianProcess and MappedGaussianProcess

    Examples:

        pm = ParameterHelper(species=['C', 'H', 'O'],
                                   bonds=[['*', '*'], ['O','O']],
                                   triplets=[['*', '*', '*'],
                                       ['O','O', 'O']],
                                   parameters={'bond0':[1, 0.5, 1], 'bond1':[2, 0.2, 2],
                                         'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                                         'cutoff3b':1},
                                   constraints={'bond0':[False, True]})
        hm = pm.hyps_mask
        hyps = hm['hyps']
        cutoffs = hm['cutoffs']
        kernel_name = hm['kernel_name']

    In this example, four atomic species are involved. There are many kinds
    of bonds and triplets. But we only want to use eight different sigmas
    and lengthscales.

    In order to do so, we first define all the bonds to be group "bond0", by
    listing "*-*" as the first element in the bond argument. The second
    element O-O is then defined to be group "bond1". Note that the order
    matters here. The later element overrides the ealier one. If
    bonds=[['O', 'O'], ['*', '*']], then all bonds belong to group "bond1".

    Similarly, O-O-O is defined as triplet1, while all remaining ones
    are left as triplet0.

    The hyperpameters for each group is listed in the order of
    [sig, ls, cutoff] in the parameters argument.  So in this example,
    O-O interaction will use [2, 0.2, 2] as its sigma, length scale, and
    cutoff.

    For triplet, the parameter arrays only come with two elements. So there
    is no cutoff associated with triplet0 or triplet1; instead, a universal
    cutoff is used, which is defined as 'cutoff3b'.

    The constraints argument define which hyper-parameters will be optimized.
    True for optimized and false for being fixed.

    See more examples in tests/test_parameters.py

    """

    def __init__(self, hyps_mask=None, species=None, kernels={},
                 cutoff_group={}, parameters=None,
                 constraints={}, allseparate=False, random=False, ones=False,
                 verbose="INFO"):
        """ Initialization function

        :param hyps_mask: Not implemented yet
        :type hyps_mask: dict
        :param species: list or dictionary that define specie groups
        :type species: [dict, list]
        :param kernels: list or dictionary that define kernels and groups for the kernels
        :type kernels: [dict, list]
        :param parameters: dictionary of parameters
        :type parameters: dict
        :param constraints: whether the hyperparmeters are optimized (True) or not (False)
        :constraints: dict
        :param random: if True, define each single bond type into a separate group and randomized initial parameters
        :type random: bool
        :param verbose: level to print with "INFO", "DEBUG"
        :type verbose: str

        See format of species, bonds, triplets, cut3b, mb in list_groups() function.

        See format of parameters and constraints in list_parameters() function.

        """

        self.set_logger(verbose)

        # TO DO, sync it to kernel class
        #  need to be synced with kernel class
        self.all_kernel_types = ['bond', 'triplet', 'mb']
        self.ndim = {'bond': 2, 'triplet': 3, 'mb': 2, 'cut3b': 2}
        self.additional_groups = ['cut3b']
        self.all_types = ['specie'] + \
            self.all_kernel_types + self.additional_groups
        self.all_group_types = self.all_kernel_types + self.additional_groups

        # number of groups {'bond': 1, 'triplet': 2}
        self.n = {}
        # definition of groups {'specie': [['C', 'H'], ['O']], 'bond': [[['*', '*']], [[ele1, ele2]]]}
        self.groups = {}
        # joint values of the groups {'specie': ['C', 'H', 'O'], 'bond': [['*', '*'], [ele1, ele2]]}
        self.all_members = {}
        # names of each group {'specie': ['group1', 'group2'], 'bond': ['bond0', 'bond1']}
        self.all_group_names = {}
        # joint list of all the keys in self.all_group_names
        self.all_names = []

        # set up empty container
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
        self.opt = {'noise': True}
        self.mask = {}
        self.cutoff_list = {}
        self.noise = 0.05
        self.universal = {}

        self.cutoffs_array = np.array([0, 0, 0], dtype=np.float)
        self.hyps = None

        if isinstance(kernels, dict):
            self.kernel_dict = kernels
            self.kernel_array = list(kernels.keys())
            assert (not allseparate)
        elif isinstance(kernels, list):
            self.kernel_array = kernels
            # by default, there is only one group of hyperparameters
            # for each type of the kernel
            # unless allseparate is defined
            self.kernel_dict = {}
            for ktype in kernels:
                self.kernel_dict[ktype] = [['*']*self.ndim[ktype]]

        if species is not None:
            self.list_groups('specie', species)

            # define groups
            if allseparate:
                for ktype in self.kernel_array:
                    self.all_separate_groups(ktype)
            else:
                for ktype in self.kernel_array:
                    self.list_groups(ktype, self.kernel_dict[ktype])

            # define parameters
            if parameters is not None:
                self.list_parameters(parameters, constraints)

            if ('lengthscale' in self.universal and 'sigma' in self.universal):
                universal = True
            else:
                universal = False

            if (random+ones+universal) > 1:
                raise RuntimeError(
                    "random and ones cannot be simultaneously True")
            elif random or ones or universal:
                for ktype in self.kernel_array:
                    self.fill_in_parameters(
                        ktype, random=random, ones=ones, universal=universal)

            try:
                self.hyps_mask = self.as_dict()
            except:
                self.logger.info("more parameters needed to generate the hypsmask")

    def set_logger(self, verbose):

        verbose = getattr(logging, verbose.upper())
        logger = logging.getLogger('parameter_helper')
        logger.setLevel(verbose)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(verbose)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)
        self.logger = logger


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
            self.set_parameters(
                name, parameter_dict[name], constraints.get(name, True))

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
            if (len(self.all_group_names['specie']) > 0):
                raise RuntimeError("this function has to be run "
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
            if (len(self.all_group_names['specie']) == 0):
                raise RuntimeError("this function has to be run "
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
                        self.define_group(group_type, name,
                                          definition_list[name])

    def all_separate_groups(self, group_type):
        """Separate all possible types of bonds, triplets, mb.
        One type per group.

        Args:

        group_type (str): "specie", "bond", "triplet", "cut3b", "mb"

        """
        nspec = len(self.all_group_names['specie'])
        if (nspec < 1):
            raise RuntimeError("the specie group has to be defined in advance")
        if (group_type in self.all_group_types):
            # TO DO: the two blocks below can be replace by some upper triangle operation

            # generate all possible combination of group
            ele_grid = self.all_group_names['specie']
            grid = np.meshgrid(*[ele_grid]*self.ndim[group_type])
            grid = np.array(grid).T.reshape(-1, self.ndim[group_type])

            # remove the redundant groups
            allgroup = []
            for group in grid:
                exist = False
                set_list_group = set(list(group))
                for prev_group in allgroup:
                    if set(prev_group) == set_list_group:
                        exist = True
                if (not exist):
                    allgroup += [list(group)]

            # define the group
            tid = 0
            for group in allgroup:
                self.define_group(group_type, f'{group_type}{tid}',
                                  group)
                tid += 1
        else:
            logger.warning(f"{group_type} will be ignored")

    def fill_in_parameters(self, group_type, random=False, ones=False, universal=False):
        """Separate all possible types of bonds, triplets, mb.
        One type per group. And fill in either universal ls and sigma from
        pre-defined parameters from set_parameters("sigma", ..) and set_parameters("ls", ..)
        or random parameters if random is True.

        Args:

        group_type (str): "specie", "bond", "triplet", "cut3b", "mb"
        definition_list (list, dict): list of elements

        """
        nspec = len(self.all_group_names['specie'])
        if (nspec < 1):
            raise RuntimeError("the specie group has to be defined in advance")
        if random:
            for group_name in self.all_group_names[group_type]:
                self.set_parameters(group_name, parameters=np.random.random(2))
        elif ones:
            for group_name in self.all_group_names[group_type]:
                self.set_parameters(group_name, parameters=np.ones(2))
        elif universal:
            for group_name in self.all_group_names[group_type]:
                self.set_parameters(group_name,
                                    parameters=[self.universal['sigma'],
                                                self.universal['lengthscale']])

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
            raise ValueError("* is reserved for substitution, cannot be used "
                             "as a group name")

        if (group_type != 'specie'):

            # Check all the other group_type to
            exclude_list = deepcopy(self.all_types)
            ide = exclude_list.index(group_type)
            exclude_list.pop(ide)

            for gt in exclude_list:
                if (name in self.all_group_names[gt]):
                    raise ValueError("group name has to be unique across all types. "
                                     f"{name} is found in type {gt}")

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
                self.logger.debug(
                    f"Element {ele} will be defined as group {name}")
        else:
            if (len(self.all_group_names['specie']) == 0):
                raise RuntimeError("The atomic species have to be"
                                   "defined in advance")
            # first translate element/group name to group name
            group_name_list = []
            if (atomic_str):
                for ele_name in element_list:
                    if (ele_name == "*"):
                        gid += ["*"]
                    else:
                        for idx in range(self.n['specie']):
                            group_name = self.all_group_names['species'][idx]
                            if (ele_name in self.groups['specie'][idx]):
                                group_name_list += [group_name]
                                self.logger.warning(f"Element {ele_name} is used for "
                                      f"definition, but the whole group "
                                      f"{group_name} is affected")
            else:
                group_name_list = element_list

            if ("*" not in group_name_list):

                gid = []
                for ele_name in group_name_list:
                    gid += [self.all_group_names['specie'].index(ele_name)]

                for ele in self.all_members[group_type]:
                    if set(gid) == set(ele):
                        self.logger.warning(
                            f"the definition of {group_type} {ele} will be overriden")
                self.groups[group_type][groupid].append(gid)
                self.all_members[group_type].append(gid)
                self.logger.debug(
                    f"{group_type} {gid} will be defined as group {name}")
                if (parameters is not None):
                    self.set_parameters(name, parameters)
            else:
                one_star_less = deepcopy(group_name_list)
                idstar = group_name_list.index('*')
                one_star_less.pop(idstar)
                for sub in self.all_group_names['specie']:
                    self.logger.debug(f"{sub}, {one_star_less}")
                    self.define_group(group_type, name,
                                      one_star_less + [sub], parameters=parameters,
                                      atomic_str=False)

    def find_group(self, group_type, element_list, atomic_str=False):

        # remember the later command override the earlier ones
        if (group_type == 'specie'):
            if (not isinstance(element_list, str)):
                self.logger.debug("for element, it has to be a string")
                return None
            name = None
            for igroup in range(self.n['specie']):
                gname = self.all_group_names[group_type][igroup]
                allspec = self.groups[group_type][igroup]
                if (element_list in allspec):
                    name = gname
            return name
            self.logger.debug("cannot find the group")
            return None
        else:
            if ("*" in element_list):
                self.logger.debug("* cannot be used for find")
                return None
            gid = []
            for ele_name in element_list:
                gid += [self.all_group_names['specie'].index(ele_name)]
            setlist = set(gid)
            name = None
            for igroup in range(self.n[group_type]):
                gname = self.all_group_names[group_type][igroup]
                for ele in self.groups[group_type][igroup]:
                    if set(gid) == set(ele):
                        name = gname
            return name
        #         self.groups[group_type][groupid].append(gid)
        #         self.all_members[group_type].append(gid)
        #         self.logger.debug(
        #             f"{group_type} {gid} will be defined as group {name}")
        #         if (parameters is not None):
        #             self.set_parameters(name, parameters)
        #     else:
        #         one_star_less = deepcopy(element_list)
        #         idstar = element_list.index('*')
        #         one_star_less.pop(idstar)
        #         for sub in self.all_group_names['specie']:
        #             self.define_group(group_type, name,
        #                               one_star_less + [sub], parameters=parameters, atomic_str=atomic_str)

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
            cutstr2index = {'cutoff2b': 0, 'cutoff3b': 1, 'cutoffmb': 2}
            self.cutoffs_array[cutstr2index[name]] = parameters
            return

        if (name in ['sigma', 'lengthscale']):
            self.universal[name] = parameters
            return

        if (isinstance(opt, bool)):
            opt = [opt, opt, opt]

        if ('cut3b' not in name):
            if (name in self.sigma):
                self.logger.warning(
                    f"the sig, ls of group {name} is overriden")
            self.sigma[name] = parameters[0]
            self.ls[name] = parameters[1]
            self.opt[name+'sig'] = opt[0]
            self.opt[name+'ls'] = opt[1]
            self.logger.debug(f"ParameterHelper for group {name} will be set as "
                  f"sig={parameters[0]} ({opt[0]}) "
                  f"ls={parameters[1]} ({opt[1]})")
            if (len(parameters) > 2):
                if (name in self.all_cutoff):
                    self.logger.warning(
                        f"the cutoff of group {name} is overriden")
                self.all_cutoff[name] = parameters[2]
                self.logger.debug(f"Cutoff for group {name} will be set as "
                      f"{parameters[2]}")
        else:
            self.all_cutoff[name] = parameters

    def set_constraints(self, name, opt):
        """Set the parameters for certain group

        :param name: name of the patermeters
        :type name: str
        :param opt: whether to optimize the parameter or not
        :type opt: bool, list

        The name of parameters can be the group name previously defined in
        define_group or list_groups function. Aside from the group name,
        "noise", "cutoff2b", "cutoff3b", and "cutoffmb" are reserved for
        noise parmater and universal cutoffs.

        The optimization flag can be a single bool, which apply to all
        parameters under that name, or list of bools that apply to each
        parameter.
        """

        if (name == 'noise'):
            self.opt['noise'] = opt
            return

        if (name in ['cutoff2b', 'cutoff3b', 'cutoffmb']):
            cutstr2index = {'cutoff2b': 0, 'cutoff3b': 1, 'cutoffmb': 2}
            return

        if (isinstance(opt, bool)):
            opt = [opt, opt, opt]

        if ('cut3b' not in name):
            if (name in self.sigma):
                self.logger.warning(
                    f"the sig, ls of group {name} is overriden")
            self.opt[name+'sig'] = opt[0]
            self.opt[name+'ls'] = opt[1]
            self.logger.debug(f"ParameterHelper for group {name} will be set as "
                  f"sig {opt[0]} "
                  f"ls {opt[1]}")

    def summarize_group(self, group_type):
        """Sort and combine all the previous definition to internal varialbes

        Args:

        group_type (str): species, bond, triplet, cut3b, mb
        """
        aeg = self.all_group_names[group_type]
        nspecie = self.n['specie']
        if (group_type == "specie"):
            self.nspecie = nspecie
            self.specie_mask = np.ones(118, dtype=np.int)*(nspecie-1)
            for idt in range(self.nspecie):
                for ele in self.groups['specie'][idt]:
                    atom_n = element_to_Z(ele)
                    self.specie_mask[atom_n] = idt
                    self.logger.debug(f"elemtn {ele} is defined as type {idt} with name "
                          f"{aeg[idt]}")
            self.logger.debug(
                f"All the remaining elements are left as type {idt}")
        elif (group_type in ['bond', 'cut3b', 'mb']):
            if (self.n[group_type] == 0):
                self.logger.debug(f"{group_type} is not defined. Skipped")
                return
            self.mask[group_type] = np.ones(
                nspecie**2, dtype=np.int)*(self.n[group_type]-1)
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
                    self.logger.debug(f"{group_type} {s1} - {s2} is defined as type {idt} "
                          f"with name {name}")
                if (group_type != 'cut3b'):
                    sig = self.sigma[name]
                    ls = self.ls[name]
                    self.hyps_sig[group_type] += [sig]
                    self.hyps_ls[group_type] += [ls]
                    self.hyps_opt[group_type] += [self.opt[name+'sig']]
                    self.hyps_opt[group_type] += [self.opt[name+'ls']]
                    self.logger.debug(f"   using hyper-parameters of {sig:6.2g} "
                          f"{ls:6.2g}")
            self.logger.debug(
                f"All the remaining elements are left as type {idt}")

            cutstr2index = {'bond': 0, 'cut3b': 1, 'mb': 2}

            # sort out the cutoffs
            allcut = []
            alldefine = True
            for idt in range(self.n[group_type]):
                if (aeg[idt] in self.all_cutoff):
                    allcut += [self.all_cutoff[aeg[idt]]]
                else:
                    alldefine = False
                    self.logger.warning(f"{aeg[idt]} cutoff is not define. "
                          "it's going to use the universal cutoff.")

            if len(allcut) > 0:
                universal_cutoff = self.cutoffs_array[cutstr2index[group_type]]
                if (universal_cutoff <= 0):
                    universal_cutoff = np.max(allcut)
                    self.logger.warning(f"universal cutoffs {cutstr2index[group_type]}for "
                          f"{group_type} is defined as zero! reset it to {universal_cutoff}")

                self.cutoff_list[group_type] = []
                for idt in range(self.n[group_type]):
                    self.cutoff_list[group_type] += [
                        self.all_cutoff.get(aeg[idt], universal_cutoff)]

                max_cutoff = np.max(self.cutoff_list[group_type])
                # update the universal cutoff to make it higher than
                if (alldefine):
                    self.cutoffs_array[cutstr2index[group_type]] = \
                        max_cutoff
                elif (not np.any(self.cutoff_list[group_type]-max_cutoff)):
                    # if not all the cutoffs are defined separately
                    # and they are all the same value. so
                    del self.cutoff_list[group_type]
                    if (group_type == 'cut3b'):
                        self.cutoffs_array[cutstr2index[group_type]
                                           ] = max_cutoff
                        self.n['cut3b'] = 0

            if (self.cutoffs_array[cutstr2index[group_type]] <= 0):
                raise RuntimeError(
                    f"cutoffs for {group_type} is undefined")

        elif (group_type == "triplet"):
            self.ntriplet = self.n['triplet']
            if (self.ntriplet == 0):
                self.logger.debug(group_type, "is not defined. Skipped")
                return
            self.mask[group_type] = np.ones(
                nspecie**3, dtype=np.int)*(self.ntriplet-1)
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
                    self.logger.debug(f"triplet {s1} - {s2} - {s3} is defined as type {idt} with name "
                          f"{name}")
                sig = self.sigma[name]
                ls = self.ls[name]
                self.hyps_sig[group_type] += [sig]
                self.hyps_ls[group_type] += [ls]
                self.hyps_opt[group_type] += [self.opt[name+'sig']]
                self.hyps_opt[group_type] += [self.opt[name+'ls']]
                self.logger.debug(
                    f"   using hyper-parameters of {sig} {ls}")
            self.logger.debug(
                f"all the remaining elements are left as type {idt}")
            if (self.cutoffs_array[1] == 0):
                allcut = []
                for idt in range(self.n[group_type]):
                    allcut += [self.all_cutoff.get(aeg[idt], 0)]
                aeg_ = self.all_group_names['cut3b']
                for idt in range(self.n['cut3b']):
                    allcut += [self.all_cutoff.get(aeg_[idt], 0)]
                if (len(allcut) > 0):
                    self.cutoffs_array[1] = np.max(allcut)
                else:
                    raise RuntimeError(
                        f"cutoffs for {group_type} is undefined")
        else:
            pass

    def as_dict(self):
        """Dictionary representation of the mask. The output can be used for AtomicEnvironment
        or the GaussianProcess
        """

        # sort out all the definitions and resolve conflicts
        # cut3b has to be summarize before triplet
        # because the universal triplet cutoff is checked
        # at the end of triplet search
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
            self.logger.debug("only one type of elements was defined. Please use multihyps=False")

        return hyps_mask

    @staticmethod
    def from_dict(hyps_mask, verbose=False, init_spec=[]):
        """ convert dictionary mask to HM instance
        This function is not tested yet
        """

        Parameters.check_instantiation(hyps_mask)

        pm = ParameterHelper(verbose=verbose)

        hyps = hyps_mask['hyps']

        if 'map' in hyps_mask:
            ihyps = hyps
            hyps = hyps_mask['original']
            constraints = np.zeros(len(hyps), dtype=bool)
            for ele in hyps_mask['map']:
                constraints[ele] = True
            for i, ori in enumerate(hyps_mask['map']):
                hyps[ori] = ihyps[i]
        else:
            constraints = np.ones(len(hyps), dtype=bool)

        pm.nspecie = hyps_mask['nspecie']
        nele = len(hyps_mask['specie_mask'])
        max_species = np.max(hyps_mask['specie_mask'])
        specie_mask = hyps_mask['specie_mask']
        for i in range(max_species+1):
            elelist = np.where(specie_mask == i)[0]
            if len(elelist) > 0:
                for ele in elelist:
                    if (ele != 0):
                        elename = Z_to_element(ele)
                        if (len(init_spec) > 0):
                            if elename in init_spec:
                                pm.define_group(
                                    "specie", i, [elename])
                        else:
                            pm.define_group("specie", i, [elename])

        nbond = hyps_mask.get('nbond', 0)
        ntriplet = hyps_mask.get('ntriplet', 0)
        nmb = hyps_mask.get('nmb', 0)
        for t in ['bond', 'mb']:
            if (f'n{t}' in hyps_mask):
                if (t == 'bond'):
                    cutoffname = 'cutoff_2b'
                    sig = hyps[:nbond]
                    ls = hyps[nbond:2*nbond]
                    csig = constraints[:nbond]
                    cls = constraints[nbond:2*nbond]
                else:
                    cutoffname = 'cutoff_mb'
                    sig = hyps[nbond*2+ntriplet*2:nbond*2+ntriplet*2+nmb]
                    ls = hyps[nbond*2+ntriplet*2+nmb:nbond*2+ntriplet*2+nmb*2]
                    csig = constraints[nbond*2+ntriplet*2:]
                    cls = constraints[nbond*2+ntriplet*2+nmb:]
                for i in range(pm.nspecie):
                    for j in range(i, pm.nspecie):
                        ttype = hyps_mask[f'{t}_mask'][i+j*pm.nspecie]
                        pm.define_group(f"{t}", f"{t}{ttype}", [i, j])
                for i in range(hyps_mask[f'n{t}']):
                    if (cutoffname in hyps_mask):
                        pm.set_parameters(f"{t}{i}", [sig[i], ls[i], hyps_mask[cutoffname][i]],
                                          opt=[csig[i], cls[i]])
                    else:
                        pm.set_parameters(f"{t}{i}", [sig[i], ls[i]],
                                          opt=[csig[i], cls[i]])
        if ('ntriplet' in hyps_mask):
            sig = hyps[nbond*2:nbond*2+ntriplet]
            ls = hyps[nbond*2+ntriplet:nbond*2+ntriplet*2]
            csig = constraints[nbond*2:nbond*2+ntriplet]
            cls = constraints[nbond*2+ntriplet:nbond*2+ntriplet*2]
            for i in range(pm.nspecie):
                for j in range(i, pm.nspecie):
                    for k in range(j, pm.nspecie):
                        triplettype = hyps_mask[f'triplet_mask'][i +
                                                                 j*pm.nspecie+k*pm.nspecie*pm.nspecie]
                        pm.define_group(
                            f"triplet", f"triplet{triplettype}", [i, j, k])
            for i in range(hyps_mask[f'ntriplet']):
                pm.set_parameters(f"triplet{i}", [sig[i], ls[i]],
                                  opt=[csig[i], cls[i]])
        if (f'ncut3b' in hyps_mask):
            for i in range(pm.nspecie):
                for j in range(i, pm.nspecie):
                    ttype = hyps_mask[f'cut3b_mask'][i+j*pm.nspecie]
                    pm.define_group("cut3b", f"cut3b{ttype}", [i, j])
            for i in range(hyps_mask['ncut3b']):
                pm.set_parameters(
                    f"cut3b{i}", [0, 0, hyps_mask['cutoff_3b'][i]])

        pm.set_parameters('noise', hyps[-1])
        if 'cutoffs' in hyps_mask:
            cut = hyps_mask['cutoffs']
            pm.set_parameters(f"cutoff2b", cut[0])
            try:
                pm.set_parameters(f"cutoff3b", cut[1])
            except:
                pass
            try:
                pm.set_parameters(f"cutoffmb", cut[2])
            except:
                pass

        return pm
