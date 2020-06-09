import numpy as np
from math import exp
from numba import njit
import flare.kernels.cutoffs as cf

from flare.kernels.cutoffs import quadratic_cutoff

from flare.kernels.sc import two_body_jit, two_body_grad_jit,\
                        two_body_force_en_jit, two_body_en_jit, \
                            three_body_jit, three_body_grad_jit, \
                        three_body_force_en_jit, three_body_en_jit, \
                        many_body_grad_jit, \
                        many_body_force_en_jit, many_body_en_jit

from flare.kernels.mc_simple import two_body_mc_jit, two_body_mc_grad_jit

from flare.env import AtomicEnvironment
from typing import Union
from flare.parameters import Parameters
import inspect
from typing import List, Set
from flare.utils.parameter_helper import ParameterHelper

# -----------------------------------------------------------------------------
#                            master kernel class
# -----------------------------------------------------------------------------



class KernelBase(object):

    def __init__(self):
        pass

    def energy_energy(self, env1:AtomicEnvironment, env2:AtomicEnvironment):
        pass

    def force_energy(self, env1: AtomicEnvironment,env2:AtomicEnvironment):
        pass

    def stress_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    def force_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    def stress_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    def stress_stress(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    def force_force_gradient(self, env1: AtomicEnvironment,
                             env2: AtomicEnvironment):
        pass

    def energy_all(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    def force_all(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass





class CoreKernel(object):

    # Aarguments to Numba functions which mirror the attribute
    # names in AtomicEnvironments
    env_attr_keys = {'bond_array',
                'ctype',
                'etypes',
                'cross_bond_inds',
                'cross_bond_dists',
                'triplets'}

    # Keys for hyperparameters which are routed through a hyp. mask
    hyp_attr_keys = {'ls', 'sig'}

    # Keys for cutoff values routed through ???
    cutoff_keys = {'r_cut', 'cutoff_func'}

    def __init__(self,
                 init_string: str = '',
                 two_body: bool = False,
                 three_body: bool = False,
                 many_body: bool = False,
                 separate_cutoffs: bool =False,
                 separate_hyps: bool = False,
                 multicomponent: bool = True,
                 use_stress: bool = False
                 ):



        self.call_arg_sets = []
        self.force_callables = []
        self.force_energy_callables = []
        self.energy_callables = []
        self.grad_callables = []


        # Set what attribtutes will need to be extracted from envs, hyps,
        # cutoffs

        # attr_set_from_envs is a list of lists; each list contains a tuple
        # with the first element corresponding to what is loaded from the
        # env / hyp and the second element corresponded to what is passed
        # into the callable.
        self.attr_set_from_envs = []
        self.attr_set_from_hyps = []
        self.attr_set_from_cutoffs = []

        # Set up internal kernel
        init_two_body, init_three_body, init_many_body = \
            self.parse_init_string(init_string)
        self.two_body = init_two_body or two_body
        self.three_body = init_three_body or three_body
        self.many_body = init_many_body or many_body

        kernel_string_list = []
        if self.two_body: kernel_string_list.append('two_body')
        if self.three_body: kernel_string_list.append('three_body')
        if self.many_body: kernel_string_list.append('many_body')



        #TODO line
        # TODO SEP HYP SETUP HERE
        self.hyp_helper = ParameterHelper(kernels=kernel_string_list)

        if multicomponent and not separate_hyps:
            self.elements_to_hyp_index =
        else:
            None


        # Add a callable and argument set for the two-body term
        if two_body:

            if separate_hyps:
                self.two_body_hyps = hyp_helper.n_terms
            else:
                self.two_body_hyps = 2

            self.two_body_kernel = TwoBodyKernel()

        if three_body:
            if not multicomponent:
                self.force_callables.append(three_body_jit)
                self.force_energy_callables.append(three_body_force_en_jit)
                self.energy_callables.append(three_body_en_jit)
                self.grad_callables.append(three_body_grad_jit)

            elif multicomponent:

                if two_body:
                    self.force_callables.append(two_body_mc_jit)
                    self.grad_callables.append(two_body_mc_grad_jit)

        if many_body:
            raise NotImplementedError

    self.two_body_kernel()



    def evaluate(self,env1: AtomicEnvironment,
                        env2: AtomicEnvironment,
                        d1: int, d2: int,
                        hyps, cutoffs, parameters, kernel,
                        **kwargs):
        """
        Main loop where arguments are extracted from the kernel, extracted
        from the input, and the kernel is evaluated.

        :param env1:
        :param env2:
        :param d1:
        :param d2:
        :param hyps:
        :param cutoffs:
        :param parameters:
        :param kernel:
        :param kwargs:
        :return:
        """

        kernel_name = kernel.__name__
        kernel_args = list(inspect.signature(kernel).parameters.keys())
        arg_set = set(strip_trailing_number(key) for key in kernel_args)

        # Set up dictionary which will supply keyword
        # arguments to njit function. Begin with d1 and d2.
        call_dict = {'d1': d1, 'd2': d2}

        env_args = arg_set.intersection(self.env_attr_keys)

        call_dict = self.populate_call_dict_with_env_attr(env_args,
                                                          kernel_name,
                                                          call_dict,
                                                          env1,
                                                          env2)

        # TODO BLOCK HERE WHERE HYPERPARAMETERS ARE ACCESSED
        # CURRENTLY ARE JUST USING THE ARCHAIC CONVENTION
        # CAN USE C1, C2 here with hyp masks

        hyp_args = arg_set.intersection(self.hyp_keys)


        call_dict = self.populate_call_dict_with_hyp_values(hyp_args,
                                                          kernel_name,
                                                          call_dict,
                                                          env1,
                                                          env2)

        cutoff_args = arg_set.intersection(self.cutoff_keys)

        call_dict = self.populate_call_dict_with_cutoff_values(c)

        for arg in self.attr_set_from_cutoffs[i]:
            call_dict[arg[1]] = cutoffs[arg[0]]

        for key in kernel_args:
            if key in kwargs.keys():
                call_dict[key] = kwargs[key]

        # Get around Numba's lack of support for kwargs by assembling
        # input tuple in real-time
        arg_input = tuple(call_dict[arg] for arg in kernel_args)

        return kernel(*arg_input)


    def populate_call_dict_with_env_attr(self, env_args: Set[str],
                         kernel_name: str,
                         call_dict: dict,
                         env1: AtomicEnvironment,
                         env2: AtomicEnvironment):

        # TODO This  could be cleaned up with conventions decided
        # for the input arguments for the jit function.s.
        for arg in env_args:
            access_arg = str(arg)
            write_arg = str(arg)

            # Modify access arg for bond array
            if arg == 'bond_array':
                if 'two_body' in kernel_name:
                    access_arg += '_2'
                elif 'three_body' in kernel_name:
                    access_arg += '_3'
                elif 'many_body' in kernel_name:
                    raise NotImplementedError

            call_dict[write_arg + '_1'] = getattr(env1, access_arg)
            call_dict[write_arg + '_2'] = getattr(env2, access_arg)

        return call_dict


    def populate_call_dict_with_hyp_values(self, hyp_args: Set[str],
                         kernel_name: str,
                         call_dict: dict,
                         env1: AtomicEnvironment,
                         env2: AtomicEnvironment,
                         hyps,
                         hyp_mask = None
                        ):

        if hyp_mask is None:
            hyp_mask = {}

        for arg in hyp_args:
            access_arg = str(arg)
            write_arg = str(arg)


            #TODO LOGIC HERE MAPS THROUGH HYPS MASK?
            access_index = None # SOME FUNCTION OF the ARG (ls, sig)

            call_dict[write_arg] = hyps[access_index]

        return call_dict


    def populate_call_dict_with_cutoff_values(self, hyp_args: Set[str],
                         kernel_name: str,
                         call_dict: dict,
                         env1: AtomicEnvironment,
                         env2: AtomicEnvironment):

        raise NotImplementedError





    def force_eval(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0

        for i, kernel in enumerate(self.force_callables):
            value += self.evaluate_kernel(kernel=kernel, env1=env1,
                                     env2=env2,d1=d1,d2=d2,hyps=hyps,
                                     cutoffs=cutoffs, parameters=parameters,
                                     **kwargs)
        return value



    def energy_eval(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0
        for i, kernel in enumerate(self.energy_callables):
            value += self.evaluate_kernel(kernel=kernel, env1=env1,
                                     env2=env2, d1=d1, d2=d2, hyps=hyps,
                                     cutoffs=cutoffs, parameters=parameters,
                                     **kwargs)
        return value



    def force_energy_eval(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0
        for i, kernel in enumerate(self.force_energy_callables):
            value += self.evaluate_kernel(kernel=kernel, env1=env1,
                                     env2=env2, d1=d1, d2=d2, hyps=hyps,
                                     cutoffs=cutoffs, parameters=parameters,
                                     **kwargs)
        return value



    def grad_eval(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0
        for i, kernel in enumerate(self.grad_callables):
            value += self.evaluate_kernel(kernel=kernel, env1=env1,
                                     env2=env2, d1=d1, d2=d2, hyps=hyps,
                                     cutoffs=cutoffs, parameters=parameters,
                                     **kwargs)
        return value

    @staticmethod
    def parse_init_string(init_string: str = None):

        if 'two' in init_string.lower() or '2' in init_string:
            two_body = True
        if 'three' in init_string.lower() or '3' in init_string:
            three_body = True
        if 'mc' in init_string.lower():
            multicomponent = True
        if 'sc' in init_string.lower():
            multicomponent = False

        return two_body, three_body, multicomponent




class Kernel_first_draft:

    def __init__(self,
                 init_string: str = '',
                 two_body: bool = False,
                 three_body: bool = False,
                 many_body: bool = False,
                 separate_cutoffs: bool =False,
                 separate_hyps: bool = False,
                 multicomponent: bool = False,
                 use_force: bool = False,
                 use_energy: bool = False,
                 use_stress: bool = False
                 ):



        self.call_arg_sets  = []
        self.kernel_callables = []
        self.grad_callables = []

        # Set what attribtutes will need to be extracted from envs, hyps,
        # cutoffs

        # attr_set_from_envs is a list of lists; each list contains a tuple
        # with the first element corresponding to what is loaded from the
        # env / hyp and the second element corresponded to what is passed
        # into the callable.
        self.attr_set_from_envs = []
        self.attr_set_from_hyps = []
        self.attr_set_from_cutoffs = []

        # Add a callable and argument set for the two-body term
        if two_body:
            self.call_arg_sets.append(['bond_arrays_1', 'bond_arrays_2',
                                'd1', 'd2', 'sig', 'ls', 'r_cut',
                                   'cutoff_func'])
            self.attr_set_from_envs.append([
                                        ('bond_array_2', 'bond_array')
                                        ])
            self.attr_set_from_hyps.append(
                                    [(0, 'sig'),
                                     (1, 'ls')])

            self.attr_set_from_hyps.append(
                                    [(0, 'sig'),
                                     (1, 'ls')])

            self.attr_set_from_cutoffs.append([
                                    (0, 'r_cut')
                                    ])

            if not multicomponent:
                self.callables.append(two_body_jit)

        # Add a callable and argument set for the three-body term
        if three_body:
            self.call_arg_sets.append(['bond_arrays_1', 'bond_arrays_2',
                                   'cross_bond_inds_1', 'cross_bond_inds_2',
                                   'cross_bond_dists_1', 'cross_bond_dists_2',
                                   'triplets_1', 'triplets_2',
                                'd1', 'd2', 'sig', 'ls', 'r_cut',
                                   'cutoff_func'])

            self.attr_set_from_envs.append([
                                            ('bond_array_3', 'bond_array'),
                                                ('ctype','ctype'),
                ('etypes','etypes'),
                ('cross_bond_inds','cross_bond_inds'),
                 ('cross_bond_dists','cross_bond_dists'),
                ('triplet_counts','triplet_counts')
                                                 ])
            raise NotImplementedError


        if multicomponent:
            for arg_set in self.call_arg_sets:
                arg_set += ['c1', 'c2', 'etypes1', 'etypes2']
            raise NotImplementedError


        self.elt_pair_to_hyp = {}
        self.elt_to_cutoff = {}

        # Convert to python set format for later comparison
        for i, arg_set in enumerate(self.call_arg_sets):
            self.call_arg_sets[i] = set(arg_set)


    def __call__(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0

        c1 = env1.ctype
        c2 = env2.ctype

        for i, call_arg_set in enumerate(self.call_arg_sets):

            # Set up dictionary which will supply keyword
            # arguments to njit function
            call_dict = {}

            # Obtain keys for argument from current callable
            call_keys = set(kwargs.keys()).intersection(call_arg_set)

            # Fetch requisite data from environments, hyps, cutoffs objects
            for arg in self.attr_set_from_envs[i]:
                call_dict[arg[1]+'_1'] = getattr(env1, arg[0])
                call_dict[arg[1]+'_2'] = getattr(env2, arg[0])


            # TODO BLOCK HERE WHERE HYPERPARAMETERS ARE ACCESSED
            # CURRENTLY ARE JUST USING THE ARCHAIC CONVENTION
            # CAN USE C1, C2 here with hyp masks

            for arg in self.attr_set_from_hyps[i]:
                call_dict[arg[1]] = hyps[arg[0]]

            for arg in self.attr_set_from_cutoffs[i]:
                call_dict[arg[1]] = cutoffs[arg[0]]

            for key in call_keys:
                if key in kwargs.keys():
                    call_dict[key] = kwargs[key]

            call_dict['d1'] = d1
            call_dict['d2'] = d2

            # Get around Numba's lack of support for kwargs by assembling
            # input tuple in real-time
            callable_arguments = inspect.signature(self.callables[i])
            arg_input = tuple(call_dict[arg] for arg in
                              callable_arguments.parameters.keys())
            print(arg_input)
            value += self.callables[i](*arg_input)

        return value





class Kernel_second_draft:

    # Aarguments to Numba functions which mirror the attribute
    # names in AtomicEnvironments
    env_attr_keys = {'bond_array',
                'ctype',
                'etypes',
                'cross_bond_inds',
                'cross_bond_dists',
                'triplets'}

    # Keys for hyperparameters which are routed through a hyp. mask
    hyp_attr_keys = {'ls', 'sig'}

    # Keys for cutoff values routed through ???
    cutoff_keys = {'r_cut', 'cutoff_func'}

    def __init__(self,
                 init_string: str = '',
                 two_body: bool = False,
                 three_body: bool = False,
                 many_body: bool = False,
                 separate_cutoffs: bool =False,
                 separate_hyps: bool = False,
                 multicomponent: bool = True,
                 use_stress: bool = False
                 ):

        self.call_arg_sets = []
        self.force_callables = []
        self.force_energy_callables = []
        self.energy_callables = []
        self.grad_callables = []

        # Set what attribtutes will need to be extracted from envs, hyps,
        # cutoffs

        # attr_set_from_envs is a list of lists; each list contains a tuple
        # with the first element corresponding to what is loaded from the
        # env / hyp and the second element corresponded to what is passed
        # into the callable.
        self.attr_set_from_envs = []
        self.attr_set_from_hyps = []
        self.attr_set_from_cutoffs = []

        if init_string:
            if 'two' in init_string.lower() or '2' in init_string:
                two_body = True
            if 'three' in init_string.lower() or '3' in init_string:
                three_body = True
            if 'mc' in init_string.lower():
                multicomponent = True
            if 'sc' in init_string.lower():
                multicomponent = False
            #TODO ADD MORE HERE

        # Add a callable and argument set for the two-body term
        if two_body:

            if not multicomponent:

                if two_body:
                    self.force_callables.append(two_body_jit)
                    self.force_energy_callables.append(two_body_force_en_jit)
                    self.energy_callables.append(two_body_en_jit)
                    self.grad_callables.append(two_body_grad_jit)

            elif multicomponent:

                pass

        if three_body:
            if not multicomponent:
                self.force_callables.append(three_body_jit)
                self.force_energy_callables.append(three_body_force_en_jit)
                self.energy_callables.append(three_body_en_jit)
                self.grad_callables.append(three_body_grad_jit)

            elif multicomponent:

                if two_body:
                    self.force_callables.append(two_body_mc_jit)
                    self.grad_callables.append(two_body_mc_grad_jit)

        if many_body:
            raise NotImplementedError



    def evaluate_kernel(self,
                        env1: AtomicEnvironment,
                        env2: AtomicEnvironment,
                        d1: int, d2: int,
                        hyps, cutoffs, parameters, kernel,
                        **kwargs):
        """
        Main loop where arguments are extracted from the kernel, extracted
        from the input, and the kernel is evaluated.

        :param env1:
        :param env2:
        :param d1:
        :param d2:
        :param hyps:
        :param cutoffs:
        :param parameters:
        :param kernel:
        :param kwargs:
        :return:
        """

        kernel_name = kernel.__name__
        kernel_args = list(inspect.signature(kernel).parameters.keys())
        arg_set = set(strip_trailing_number(key) for key in kernel_args)

        # Set up dictionary which will supply keyword
        # arguments to njit function. Begin with d1 and d2.
        call_dict = {'d1': d1, 'd2': d2}

        env_args = arg_set.intersection(self.env_attr_keys)

        call_dict = self.populate_call_dict_with_env_attr(env_args,
                                                          kernel_name,
                                                          call_dict,
                                                          env1,
                                                          env2)

        # TODO BLOCK HERE WHERE HYPERPARAMETERS ARE ACCESSED
        # CURRENTLY ARE JUST USING THE ARCHAIC CONVENTION
        # CAN USE C1, C2 here with hyp masks

        hyp_args = arg_set.intersection(self.hyp_keys)


        call_dict = self.populate_call_dict_with_hyp_values(hyp_args,
                                                          kernel_name,
                                                          call_dict,
                                                          env1,
                                                          env2)

        cutoff_args = arg_set.intersection(self.cutoff_keys)

        call_dict = self.populate_call_dict_with_cutoff_values(c)

        for arg in self.attr_set_from_cutoffs[i]:
            call_dict[arg[1]] = cutoffs[arg[0]]

        for key in kernel_args:
            if key in kwargs.keys():
                call_dict[key] = kwargs[key]

        # Get around Numba's lack of support for kwargs by assembling
        # input tuple in real-time
        arg_input = tuple(call_dict[arg] for arg in kernel_args)

        return kernel(*arg_input)


    def populate_call_dict_with_env_attr(self, env_args: Set[str],
                         kernel_name: str,
                         call_dict: dict,
                         env1: AtomicEnvironment,
                         env2: AtomicEnvironment):

        # TODO This  could be cleaned up with conventions decided
        # for the input arguments for the jit function.s.
        for arg in env_args:
            access_arg = str(arg)
            write_arg = str(arg)

            # Modify access arg for bond array
            if arg == 'bond_array':
                if 'two_body' in kernel_name:
                    access_arg += '_2'
                elif 'three_body' in kernel_name:
                    access_arg += '_3'
                elif 'many_body' in kernel_name:
                    raise NotImplementedError

            call_dict[write_arg + '_1'] = getattr(env1, access_arg)
            call_dict[write_arg + '_2'] = getattr(env2, access_arg)

        return call_dict


    def populate_call_dict_with_hyp_values(self, hyp_args: Set[str],
                         kernel_name: str,
                         call_dict: dict,
                         env1: AtomicEnvironment,
                         env2: AtomicEnvironment,
                         hyps,
                         hyp_mask = None
                        ):

        if hyp_mask is None:
            hyp_mask = {}

        for arg in hyp_args:
            access_arg = str(arg)
            write_arg = str(arg)


            #TODO LOGIC HERE MAPS THROUGH HYPS MASK?
            access_index = None # SOME FUNCTION OF the ARG (ls, sig)

            call_dict[write_arg] = hyps[access_index]

        return call_dict


    def populate_call_dict_with_cutoff_values(self, hyp_args: Set[str],
                         kernel_name: str,
                         call_dict: dict,
                         env1: AtomicEnvironment,
                         env2: AtomicEnvironment):

        raise NotImplementedError





    def force_eval(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0

        for i, kernel in enumerate(self.force_callables):
            value += self.evaluate_kernel(kernel=kernel, env1=env1,
                                     env2=env2,d1=d1,d2=d2,hyps=hyps,
                                     cutoffs=cutoffs, parameters=parameters,
                                     **kwargs)
        return value



    def energy_eval(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0
        for i, kernel in enumerate(self.energy_callables):
            value += self.evaluate_kernel(kernel=kernel, env1=env1,
                                     env2=env2, d1=d1, d2=d2, hyps=hyps,
                                     cutoffs=cutoffs, parameters=parameters,
                                     **kwargs)
        return value



    def force_energy_eval(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0
        for i, kernel in enumerate(self.force_energy_callables):
            value += self.evaluate_kernel(kernel=kernel, env1=env1,
                                     env2=env2, d1=d1, d2=d2, hyps=hyps,
                                     cutoffs=cutoffs, parameters=parameters,
                                     **kwargs)
        return value



    def grad_eval(self,
                 env1: AtomicEnvironment,
                 env2: AtomicEnvironment,
                 d1: int,
                 d2: int,
                 hyps: np.array,
                 cutoffs: np.array,
                 parameters: Parameters = None,
                 **kwargs):

        value = 0
        for i, kernel in enumerate(self.grad_callables):
            value += self.evaluate_kernel(kernel=kernel, env1=env1,
                                     env2=env2, d1=d1, d2=d2, hyps=hyps,
                                     cutoffs=cutoffs, parameters=parameters,
                                     **kwargs)
        return value


def strip_trailing_number(string:str)->str:
    """
    Removes trailing numbers and underscores from strings.
    :return:
    """
    copy = str(string)
    if copy[-1].isnumeric():
        copy = copy[:-1]

    return copy.strip('_')

