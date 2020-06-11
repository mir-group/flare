import numpy as np
from math import exp
from numba import njit
import flare.kernels.cutoffs as cf

from abc import abstractmethod

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
from flare.kernels.two_body_mc_simple import TwoBodyKernel
from flare.kernels.three_body_mc_simple import ThreeBodyKernel

# -----------------------------------------------------------------------------
#                            master kernel class
# -----------------------------------------------------------------------------



class KernelBase(object):

    def __init__(self, separate_cutoffs = None, separate_hyperparameters =
    None, cutoff=None, cutoff_func: callable = quadratic_cutoff):

        self.separate_cutoffs = separate_cutoffs
        self.separate_hyperparameters = separate_hyperparameters
        self.cutoff = cutoff
        self.cutoff_func = cutoff_func



    @abstractmethod
    def energy_energy(self, env1:AtomicEnvironment, env2:AtomicEnvironment):
        pass

    @abstractmethod
    def force_energy(self, env1: AtomicEnvironment,env2:AtomicEnvironment):
        pass

    @abstractmethod
    def stress_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    @abstractmethod
    def force_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    @abstractmethod
    def stress_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    @abstractmethod
    def stress_stress(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    @abstractmethod
    def force_force_gradient(self, env1: AtomicEnvironment,
                             env2: AtomicEnvironment):
        pass

    @abstractmethod
    def energy_all(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass

    @abstractmethod
    def force_all(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        pass





class CoreKernel(object):

    def __init__(self,
                 init_string: str = '',
                 two_body: bool = False,
                 three_body: bool = False,
                 many_body: bool = False,
                 separate_cutoffs: bool = False,
                 separate_hyps: bool = False,
                 use_force: bool = True,
                 use_energy: bool = False,
                 use_stress: bool = False,
                 hyperparameters: List[float] = None
                 ):


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
            parse_init_string(init_string)
        self.two_body = init_two_body or two_body
        self.three_body = init_three_body or three_body
        self.many_body = init_many_body or many_body

        if not self.two_body: self.two_body_kernel = None
        if not self.three_body: self.three_body_kernel = None
        if not self.many_body: self.many_body_kernel = None

        kernel_string_list = []
        if self.two_body: kernel_string_list.append('two_body')
        if self.three_body: kernel_string_list.append('three_body')
        if self.many_body: kernel_string_list.append('many_body')


        #TODO line
        # TODO SEP HYP SETUP HERE
        self.hyp_helper = ParameterHelper(kernels=kernel_string_list)

        if separate_cutoffs or separate_hyps:
            raise NotImplementedError

        self.hyperparameters = hyperparameters

        # Add a callable and argument set for the two-body term
        if two_body:
            self.two_body_kernel = TwoBodyKernel(signal_variance =
                                                 self.hyperparameters[0],
                                                 length_scale =
                                                 self.hyperparameters[1])




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

        if self.two_body:
            if self.se
            value += self.two_body_kernel.force_all()
            if self.us
            value += self.two_body_kernel()

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



def strip_trailing_number(string:str)->str:
    """
    Removes trailing numbers and underscores from strings.
    :return:
    """
    copy = str(string)
    if copy[-1].isnumeric():
        copy = copy[:-1]

    return copy.strip('_')

