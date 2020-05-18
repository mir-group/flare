import time
import math
import pickle
import inspect
import json

import numpy as np
import multiprocessing as mp

from collections import Counter
from copy import deepcopy
from numpy.random import random
from typing import List, Callable, Union
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.gp_algebra import get_like_from_mats, get_neg_like_grad, \
    force_force_vector, energy_force_vector, get_force_block, \
    get_ky_mat_update, _global_training_data, _global_training_labels, \
    _global_training_structures, _global_energy_labels, get_Ky_mat, \
    get_kernel_vector, en_kern_vec
from flare.utils.mask_helper import HyperParameterMasking

from flare.kernels.utils import str_to_kernel_set, from_mask_to_args
from flare.utils.element_coder import NumpyEncoder, Z_to_element
from flare.output import Output


class GaussianProcess:
    """Gaussian process force field. Implementation is based on Algorithm 2.1
    (pg. 19) of "Gaussian Processes for Machine Learning" by Rasmussen and
    Williams.

    Args:
        kernel (Callable, optional): Name of the kernel to use, or the kernel
            itself.
        kernel_grad (Callable, optional): Function that returns the gradient
            of the GP kernel with respect to the hyperparameters.
        hyps (np.ndarray, optional): Hyperparameters of the GP.
        cutoffs (np.ndarray, optional): Cutoffs of the GP kernel.
        hyp_labels (List, optional): List of hyperparameter labels. Defaults
            to None.
        energy_force_kernel (Callable, optional): Energy/force kernel of the
            GP used to make energy predictions. Defaults to None.
        energy_kernel (Callable, optional): Energy/energy kernel of the GP.
            Defaults to None.
        opt_algorithm (str, optional): Hyperparameter optimization algorithm.
            Defaults to 'L-BFGS-B'.
        maxiter (int, optional): Maximum number of iterations of the
            hyperparameter optimization algorithm. Defaults to 10.
        parallel (bool, optional): If True, the covariance matrix K of the GP
            is computed in parallel. Defaults to False.
        n_cpus (int, optional): Number of cpus used for parallel calculations.
            Defaults to 1 (serial)
        n_samples (int, optional): Size of submatrix to use when parallelizing
            predictions.
        output (Output, optional): Output object used to dump hyperparameters
            during optimization. Defaults to None.
        multihyps (bool, optional): If True, turn on multiple-group of hyper-
            parameters.
        hyps_mask (dict, optional): If multihyps is True, hyps_mask can set up
            which hyper parameter is used for what interaction. Details see
            kernels/mc_sephyps.py
        kernel_name (str, optional): Determine the type of kernels. Example:
            2+3_mc, 2+3+mb_mc, 2_mc, 2_sc, 3_sc, ...
        name (str, optional): Name for the GP instance.
    """

    def __init__(self, kernel: Callable = None, kernel_grad: Callable = None,
                 hyps: 'ndarray' = None, cutoffs: 'ndarray' = None,
                 hyp_labels: List = None, opt_algorithm: str = 'L-BFGS-B',
                 maxiter: int = 10, parallel: bool = False,
                 per_atom_par: bool = True, n_cpus: int = 1,
                 n_sample: int = 100, output: Output = None,
                 multihyps: bool = False, hyps_mask: dict = None,
                 kernel_name="2+3_mc", name="default_gp",
                 energy_noise: float = 0.01, **kwargs,):
        """Initialize GP parameters and training data."""

        # load arguments into attributes
        self.name = name

        self.cutoffs = np.array(cutoffs, dtype=np.float64)
        self.opt_algorithm = opt_algorithm

        self.output = output
        self.per_atom_par = per_atom_par
        self.maxiter = maxiter

        if hyps is None:
            # If no hyperparameters are passed in, assume 2 hyps for each
            # cutoff, plus one noise hyperparameter, and use a guess value
            hyps = np.array([0.1]*(1+2*len(cutoffs)))

        self.update_hyps(hyps, hyp_labels, multihyps, hyps_mask)

        # set up parallelization
        self.n_cpus = n_cpus
        self.n_sample = n_sample
        self.parallel = parallel
        if 'nsample' in kwargs:
            DeprecationWarning("nsample is being replaced with n_sample")
            self.n_sample = kwargs.get('nsample')
        if 'par' in kwargs:
            DeprecationWarning("par is being replaced with parallel")
            self.parallel = kwargs.get('par')
        if 'no_cpus' in kwargs:
            DeprecationWarning("no_cpus is being replaced with n_cpu")
            self.n_cpus = kwargs.get('no_cpus')

        # TO DO, clean up all the other kernel arguments
        if kernel is None:
            kernel, grad, ek, efk = str_to_kernel_set(kernel_name, multihyps)
            self.kernel = kernel
            self.kernel_grad = grad
            self.energy_force_kernel = efk
            self.energy_kernel = ek
            self.kernel_name = kernel.__name__
        else:
            DeprecationWarning("kernel, kernel_grad, energy_force_kernel "
                               "and energy_kernel will be replaced by "
                               "kernel_name")
            self.kernel_name = kernel.__name__
            self.kernel = kernel
            self.kernel_grad = kernel_grad
            self.energy_force_kernel = kwargs.get('energy_force_kernel')
            self.energy_kernel = kwargs.get('energy_kernel')

        self.name = name

        # parallelization
        if self.parallel:
            if n_cpus is None:
                self.n_cpus = mp.cpu_count()
            else:
                self.n_cpus = n_cpus
        else:
            self.n_cpus = 1

        self.training_data = []   # Atomic environments
        self.training_labels = []  # Forces acting on central atoms

        self.training_labels_np = np.empty(0, )
        self.n_envs_prev = len(self.training_data)

        # Attributes to accomodate energy labels:
        self.training_structures = []  # Environments of each structure
        self.energy_labels = []  # Energies of training structures
        self.energy_labels_np = np.empty(0, )
        self.energy_noise = energy_noise
        self.all_labels = np.empty(0, )

        # Parameters set during training
        self.ky_mat = None
        self.force_block = None
        self.energy_block = None
        self.force_energy_block = None
        self.l_mat = None
        self.alpha = None
        self.ky_mat_inv = None
        self.likelihood = None
        self.likelihood_gradient = None
        self.bounds = None

        self.check_instantiation()

    def check_instantiation(self):
        """
        Runs a series of checks to ensure that the user has not supplied
        contradictory arguments which will result in undefined behavior
        with multiple hyperparameters.
        :return:
        """

        if (self.name in _global_training_labels):
            base = f'{self.name}'
            count = 2
            while (self.name in _global_training_labels and count < 100):
                time.sleep(random())
                self.name = f'{base}_{count}'
                print("Specified GP name is present in global memory; "
                      "Attempting to rename the "
                      f"GP instance to {self.name}")
                count += 1
            if (self.name in _global_training_labels):
                milliseconds = int(round(time.time() * 1000) % 10000000)
                self.name = f"{base}_{milliseconds}"
                print("Specified GP name still present in global memory: "
                      f"renaming the gp instance to {self.name}")
            print(f"Final name of the gp instance is {self.name}")

        assert (self.name not in _global_training_labels), \
            f"the gp instance name, {self.name} is used"
        assert (self.name not in _global_training_data),  \
            f"the gp instance name, {self.name} is used"

        _global_training_data[self.name] = self.training_data
        _global_training_structures[self.name] = self.training_structures
        _global_training_labels[self.name] = self.training_labels_np
        _global_energy_labels[self.name] = self.energy_labels_np

        assert (len(self.cutoffs) <= 3)

        if (len(self.cutoffs) > 1):
            assert self.cutoffs[0] >= self.cutoffs[1], \
                    "2b cutoff has to be larger than 3b cutoffs"

        if ('three' in self.kernel_name):
            assert len(self.cutoffs) >= 2, \
                    "3b kernel needs two cutoffs, one for building"\
                    " neighbor list and one for the 3b"
        if ('many' in self.kernel_name):
            assert len(self.cutoffs) >= 3, \
                    "many-body kernel needs three cutoffs, one for building"\
                    " neighbor list and one for the 3b"

        if self.multihyps is True and self.hyps_mask is None:
            raise ValueError("Warning! Multihyperparameter mode enabled,"
                             "but no configuration hyperparameter mask was "
                             "passed. Did you mean to set multihyps to False?")
        elif self.multihyps is False and self.hyps_mask is not None:
            raise ValueError("Warning! Multihyperparameter mode disabled,"
                             "but a configuration hyperparameter mask was "
                             "passed. Did you mean to set multihyps to True?")

        if self.multihyps is True:

            self.hyps_mask = HyperParameterMasking.check_instantiation(
                self.hyps_mask)
            HyperParameterMasking.check_matching(
                self.hyps_mask, self.hyps, self.cutoffs)
            self.bounds = deepcopy(self.hyps_mask.get('bounds', None))

        else:
            self.multihyps = False
            self.hyps_mask = None

    def update_kernel(self, kernel_name, multihyps=False):
        kernel, grad, ek, efk = str_to_kernel_set(kernel_name, multihyps)
        self.kernel = kernel
        self.kernel_grad = grad
        self.energy_force_kernel = efk
        self.energy_kernel = ek
        self.kernel_name = kernel.__name__

    def update_hyps(self, hyps=None, hyp_labels=None, multihyps=False, hyps_mask=None):
        self.hyps = np.array(hyps, dtype=np.float64)
        self.hyp_labels = hyp_labels
        self.hyps_mask = hyps_mask
        self.multihyps = multihyps

    def update_db(self, struc: Structure, forces: List,
                  custom_range: List[int] = (), energy: float = None,
                  sweep: int = 1):
        """Given a structure and forces, add local environments from the
        structure to the training set of the GP. If energy is given, add the
        entire structure to the training set.

        Args:
            struc (Structure): Input structure. Local environments of atoms
                in this structure will be added to the training set of the GP.

            forces (np.ndarray): Forces on atoms in the structure.

            custom_range (List[int]): Indices of atoms whose local
                environments will be added to the training set of the GP.

            energy (float): Energy of the structure.
        """

        # By default, use all atoms in the structure
        noa = len(struc.positions)
        update_indices = custom_range or list(range(noa))

        # If forces are given, update the environment list.
        if forces is not None:
            for atom in update_indices:
                env_curr = \
                    AtomicEnvironment(struc, atom, self.cutoffs, sweep=sweep,
                            cutoffs_mask=self.hyps_mask)
                forces_curr = np.array(forces[atom])

                self.training_data.append(env_curr)
                self.training_labels.append(forces_curr)

            # create numpy array of training labels
            self.training_labels_np = np.hstack(self.training_labels)
            _global_training_data[self.name] = self.training_data
            _global_training_labels[self.name] = self.training_labels_np

        # If an energy is given, update the structure list.
        if energy is not None:
            structure_list = []  # Populate with all environments of the struc
            for atom in range(noa):
                env_curr = \
                    AtomicEnvironment(struc, atom, self.cutoffs, sweep=sweep)
                structure_list.append(env_curr)

            self.energy_labels.append(energy)
            self.training_structures.append(structure_list)
            self.energy_labels_np = np.array(self.energy_labels)
            _global_training_structures[self.name] = self.training_structures
            _global_energy_labels[self.name] = self.energy_labels_np

        # update list of all labels
        self.all_labels = np.concatenate((self.training_labels_np,
                                          self.energy_labels_np))

    def add_one_env(self, env: AtomicEnvironment,
                    force, train: bool = False, **kwargs):
        """Add a single local environment to the training set of the GP.

        Args:
            env (AtomicEnvironment): Local environment to be added to the
                training set of the GP.
            force (np.ndarray): Force on the central atom of the local
                environment in the form of a 3-component Numpy array
                containing the x, y, and z components.
            train (bool): If True, the GP is trained after the local
                environment is added.
        """
        self.training_data.append(env)
        self.training_labels.append(force)
        self.training_labels_np = np.hstack(self.training_labels)
        _global_training_data[self.name] = self.training_data
        _global_training_labels[self.name] = self.training_labels_np

        # update list of all labels
        self.all_labels = np.concatenate((self.training_labels_np,
                                          self.energy_labels_np))

        if train:
            self.train(**kwargs)

    def train(self, output=None, custom_bounds=None,
              grad_tol: float = 1e-4,
              x_tol: float = 1e-5,
              line_steps: int = 20,
              print_progress: bool = False):
        """Train Gaussian Process model on training data. Tunes the
        hyperparameters to maximize the likelihood, then computes L and alpha
        (related to the covariance matrix of the training set).

        Args:
            output (Output): Output object specifying where to write the
                progress of the optimization.
            custom_bounds (np.ndarray): Custom bounds on the hyperparameters.
            grad_tol (float): Tolerance of the hyperparameter gradient that
                determines when hyperparameter optimization is terminated.
            x_tol (float): Tolerance on the x values used to decide when
                Nelder-Mead hyperparameter optimization is terminated.
            line_steps (int): Maximum number of line steps for L-BFGS
                hyperparameter optimization.
        """

        if len(self.training_data) == 0 or len(self.training_labels) == 0:
            raise Warning("You are attempting to train a GP with no "
                          "training data. Add environments and forces "
                          "to the GP and try again.")

        x_0 = self.hyps

        args = (self.name, self.kernel_grad, output,
                self.cutoffs, self.hyps_mask,
                self.n_cpus, self.n_sample, print_progress)

        res = None

        if self.opt_algorithm == 'L-BFGS-B':

            # bound signal noise below to avoid overfitting
            if self.bounds is None:
                bounds = np.array([(1e-6, np.inf)] * len(x_0))
                bounds[-1, 0] = 1e-3
            else:
                bounds = self.bounds

            # Catch linear algebra errors and switch to BFGS if necessary
            try:
                res = minimize(get_neg_like_grad, x_0, args,
                               method='L-BFGS-B', jac=True, bounds=bounds,
                               options={'disp': False, 'gtol': grad_tol,
                                        'maxls': line_steps,
                                        'maxiter': self.maxiter})
            except np.linalg.LinAlgError:
                print("Warning! Algorithm for L-BFGS-B failed. Changing to "
                      "BFGS for remainder of run.")
                self.opt_algorithm = 'BFGS'

        if custom_bounds is not None:
            res = minimize(get_neg_like_grad, x_0, args,
                           method='L-BFGS-B', jac=True, bounds=custom_bounds,
                           options={'disp': False, 'gtol': grad_tol,
                                    'maxls': line_steps,
                                    'maxiter': self.maxiter})

        elif self.opt_algorithm == 'BFGS':
            res = minimize(get_neg_like_grad, x_0, args,
                           method='BFGS', jac=True,
                           options={'disp': False, 'gtol': grad_tol,
                                    'maxiter': self.maxiter})

        if res is None:
            raise RuntimeError("Optimization failed for some reason.")
        self.hyps = res.x
        self.set_L_alpha()
        self.likelihood = -res.fun
        self.likelihood_gradient = -res.jac

        return res

    def check_L_alpha(self):
        """
        Check that the alpha vector is up to date with the training set. If
        not, update_L_alpha is called.
        """

        # Check that alpha is up to date with training set
        size3 = len(self.training_data)*3

        # If model is empty, then just return
        if size3 == 0:
            return

        if (self.alpha is None):
            self.update_L_alpha()
        elif (size3 > self.alpha.shape[0]):
            self.update_L_alpha()
        elif (size3 != self.alpha.shape[0]):
            self.set_L_alpha()

    def predict(self, x_t: AtomicEnvironment, d: int) -> [float, float]:
        """
        Predict a force component of the central atom of a local environment.

        Args:
            x_t (AtomicEnvironment): Input local environment.
            d (int): Force component to be predicted (1 is x, 2 is y, and
                3 is z).

        Return:
            (float, float): Mean and epistemic variance of the prediction.
        """

        assert (d in [1, 2, 3]), "d should be 1, 2, or 3"

        # Kernel vector allows for evaluation of atomic environments.
        if self.parallel and not self.per_atom_par:
            n_cpus = self.n_cpus
        else:
            n_cpus = 1

        _global_training_data[self.name] = self.training_data
        _global_training_labels[self.name] = self.training_labels_np

        k_v = \
            get_kernel_vector(self.name, self.kernel, self.energy_force_kernel,
                              x_t, d, self.hyps, cutoffs=self.cutoffs,
                              hyps_mask=self.hyps_mask, n_cpus=n_cpus,
                              n_sample=self.n_sample)

        # Guarantee that alpha is up to date with training set
        self.check_L_alpha()

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance without cholesky (possibly faster)
        # pass args to kernel based on if mult. hyperparameters in use
        args = from_mask_to_args(self.hyps, self.hyps_mask, self.cutoffs)
        self_kern = self.kernel(x_t, x_t, d, d, *args)
        pred_var = self_kern - np.matmul(np.matmul(k_v, self.ky_mat_inv), k_v)

        return pred_mean, pred_var

    def predict_local_energy(self, x_t: AtomicEnvironment) -> float:
        """Predict the local energy of a local environment.

        Args:
            x_t (AtomicEnvironment): Input local environment.

        Return:
            float: Local energy predicted by the GP.
        """

        if self.parallel and not self.per_atom_par:
            n_cpus = self.n_cpus
        else:
            n_cpus = 1

        _global_training_data[self.name] = self.training_data
        _global_training_labels[self.name] = self.training_labels_np

        k_v = en_kern_vec(self.name, self.energy_force_kernel,
                          self.energy_kernel,
                          x_t, self.hyps, cutoffs=self.cutoffs,
                          hyps_mask=self.hyps_mask, n_cpus=n_cpus,
                          n_sample=self.n_sample)

        pred_mean = np.matmul(k_v, self.alpha)

        return pred_mean

    def predict_local_energy_and_var(self, x_t: AtomicEnvironment):
        """Predict the local energy of a local environment and its
        uncertainty.

        Args:
            x_t (AtomicEnvironment): Input local environment.

        Return:
            (float, float): Mean and predictive variance predicted by the GP.
        """

        if self.parallel and not self.per_atom_par:
            n_cpus = self.n_cpus
        else:
            n_cpus = 1

        _global_training_data[self.name] = self.training_data
        _global_training_labels[self.name] = self.training_labels_np

        # get kernel vector
        k_v = en_kern_vec(self.name, self.energy_force_kernel,
                          self.energy_kernel,
                          x_t, self.hyps, cutoffs=self.cutoffs,
                          hyps_mask=self.hyps_mask, n_cpus=n_cpus,
                          n_sample=self.n_sample)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        args = from_mask_to_args(self.hyps, self.hyps_mask, self.cutoffs)

        self_kern = self.energy_kernel(x_t, x_t, *args)

        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def set_L_alpha(self):
        """
        Invert the covariance matrix, setting L (a lower triangular
        matrix s.t. L L^T = (K + sig_n^2 I)) and alpha, the inverse
        covariance matrix multiplied by the vector of training labels.
        The forces and variances are later obtained using alpha.
        """

        _global_training_data[self.name] = self.training_data
        _global_training_structures[self.name] = self.training_structures
        _global_training_labels[self.name] = self.training_labels_np
        _global_energy_labels[self.name] = self.energy_labels_np

        ky_mat = \
            get_Ky_mat(self.hyps, self.name, self.kernel,
                       self.energy_kernel, self.energy_force_kernel,
                       self.energy_noise,
                       cutoffs=self.cutoffs, hyps_mask=self.hyps_mask,
                       n_cpus=self.n_cpus, n_sample=self.n_sample)

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.all_labels)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv

        self.likelihood = get_like_from_mats(ky_mat, l_mat, alpha, self.name)
        self.n_envs_prev = len(self.training_data)

    def update_L_alpha(self):
        """
        Update the GP's L matrix and alpha vector without recalculating
        the entire covariance matrix K.
        """

        # Set L matrix and alpha if set_L_alpha has not been called yet
        if self.l_mat is None or np.array(self.ky_mat) is np.array(None):
            self.set_L_alpha()
            return

        # Reset global variables.
        _global_training_data[self.name] = self.training_data
        _global_training_structures[self.name] = self.training_structures
        _global_training_labels[self.name] = self.training_labels_np
        _global_energy_labels[self.name] = self.energy_labels_np

        ky_mat = get_ky_mat_update(self.ky_mat, self.n_envs_prev, self.hyps,
                                   self.name, self.kernel, self.energy_kernel,
                                   self.energy_force_kernel, self.energy_noise,
                                   cutoffs=self.cutoffs,
                                   hyps_mask=self.hyps_mask,
                                   n_cpus=self.n_cpus,
                                   n_sample=self.n_sample)

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.all_labels)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv
        self.n_envs_prev = len(self.training_data)

    def __str__(self):
        """String representation of the GP model."""

        thestr = "GaussianProcess Object\n"
        thestr += f'Kernel: {self.kernel_name}\n'
        thestr += f"Training points: {len(self.training_data)}\n"
        thestr += f'Cutoffs: {self.cutoffs}\n'
        thestr += f'Model Likelihood: {self.likelihood}\n'

        thestr += f'MultiHyps: {self.multihyps}\n'
        thestr += 'Hyperparameters: \n'
        if self.hyp_labels is None:
            # Put unlabeled hyperparameters on one line
            thestr = thestr[:-1]
            thestr += str(self.hyps) + '\n'
        else:
            for hyp, label in zip(self.hyps, self.hyp_labels):
                thestr += f"{label}: {hyp}\n"

        if self.multihyps:
            nspecie = self.hyps_mask['nspecie']
            thestr += f'nspecie: {nspecie}\n'
            thestr += f'specie_mask: \n'
            thestr += str(self.hyps_mask['specie_mask']) + '\n'

            nbond = self.hyps_mask['nbond']
            thestr += f'nbond: {nbond}\n'

            if nbond > 0:
                thestr += f'bond_mask: \n'
                thestr += str(self.hyps_mask['bond_mask']) + '\n'

            ntriplet = self.hyps_mask['ntriplet']
            thestr += f'ntriplet: {ntriplet}\n'
            if ntriplet > 0:
                thestr += f'triplet_mask: \n'
                thestr += str(self.hyps_mask['triplet_mask']) + '\n'

        return thestr

    def as_dict(self):
        """Dictionary representation of the GP model."""

        self.check_L_alpha()

        out_dict = deepcopy(dict(vars(self)))

        out_dict['training_data'] = [env.as_dict() for env in
                                     self.training_data]

        # Write training structures (which are just list of environments)
        out_dict['training_structures'] = []
        for n, env_list in enumerate(self.training_structures):
            out_dict['training_structures'].append([])
            for env_curr in env_list:
                out_dict['training_structures'][n].append(env_curr.as_dict())

        # Remove the callables
        for key in ['kernel', 'kernel_grad', 'energy_kernel',
                    'energy_force_kernel']:
            if out_dict.get(key) is not None:
                del out_dict[key]

        return out_dict

    @staticmethod
    def from_dict(dictionary):
        """Create GP object from dictionary representation."""

        multihyps = dictionary.get('multihyps', False)

        new_gp = GaussianProcess(kernel_name=dictionary['kernel_name'],
                                 cutoffs=np.array(dictionary['cutoffs']),
                                 hyps=np.array(dictionary['hyps']),
                                 hyp_labels=dictionary['hyp_labels'],
                                 parallel=dictionary.get('parallel', False) or
                                 dictionary.get('par', False),
                                 per_atom_par=dictionary.get('per_atom_par',
                                                             True),
                                 n_cpus=dictionary.get('n_cpus') or
                                 dictionary.get('no_cpus'),
                                 maxiter=dictionary['maxiter'],
                                 opt_algorithm=dictionary.get(
                                     'opt_algorithm', 'L-BFGS-B'),
                                 multihyps=multihyps,
                                 hyps_mask=dictionary.get('hyps_mask', None),
                                 name=dictionary.get('name', 'default_gp')
                                 )

        # Save time by attempting to load in computed attributes
        new_gp.training_data = [AtomicEnvironment.from_dict(env) for env in
                                dictionary['training_data']]

        # Reconstruct training structures.
        new_gp.training_structures = []
        for n, env_list in enumerate(dictionary['training_structures']):
            new_gp.training_structures.append([])
            for env_curr in env_list:
                new_gp.training_structures[n].append(
                    AtomicEnvironment.from_dict(env_curr))

        new_gp.training_labels = deepcopy(dictionary['training_labels'])
        new_gp.training_labels_np = deepcopy(dictionary['training_labels_np'])
        new_gp.energy_labels = deepcopy(dictionary['energy_labels'])
        new_gp.energy_labels_np = deepcopy(dictionary['energy_labels_np'])
        new_gp.all_labels = deepcopy(dictionary['all_labels'])

        new_gp.likelihood = dictionary['likelihood']
        new_gp.likelihood_gradient = dictionary['likelihood_gradient']
        new_gp.training_labels_np = np.hstack(new_gp.training_labels)
        new_gp.n_envs_prev = dictionary['n_envs_prev']

        _global_training_data[new_gp.name] = new_gp.training_data
        _global_training_structures[new_gp.name] = new_gp.training_structures
        _global_training_labels[new_gp.name] = new_gp.training_labels_np
        _global_energy_labels[new_gp.name] = new_gp.energy_labels_np

        # Save time by attempting to load in computed attributes
        if len(new_gp.training_data) > 5000:
            try:
                new_gp.ky_mat = np.load(dictionary['ky_mat_file'])
                new_gp.compute_matrices()
            except FileNotFoundError:
                new_gp.ky_mat = None
                new_gp.l_mat = None
                new_gp.alpha = None
                new_gp.ky_mat_inv = None
                filename = dictionary['ky_mat_file']
                Warning("the covariance matrices are not loaded"
                        f"because {filename} cannot be found")
        else:
            new_gp.ky_mat_inv = np.array(dictionary['ky_mat_inv']) \
                if dictionary.get('ky_mat_inv') is not None else None
            new_gp.ky_mat = np.array(dictionary['ky_mat']) \
                if dictionary.get('ky_mat') is not None else None
            new_gp.l_mat = np.array(dictionary['l_mat']) \
                if dictionary.get('l_mat') is not None else None
            new_gp.alpha = np.array(dictionary['alpha']) \
                if dictionary.get('alpha') is not None else None

        return new_gp

    def compute_matrices(self):

        ky_mat = self.ky_mat
        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv

    def adjust_cutoffs(self, new_cutoffs: Union[list, tuple, 'np.ndarray'],
                       reset_L_alpha=True, train=True, new_hyps_mask=None):
        """
        Loop through atomic environment objects stored in the training data,
        and re-compute cutoffs for each. Useful if you want to gauge the
        impact of cutoffs given a certain training set! Unless you know
        *exactly* what you are doing for some development or test purpose,
        it is **highly** suggested that you call set_L_alpha and
        re-optimize your hyperparameters afterwards as is default here.

        :param new_cutoffs:
        :return:
        """

        if (new_hyps_mask is not None):
            hm = new_hyps_mask
        else:
            hm = self.hyps_mask

        # update environment
        nenv = len(self.training_data)
        for i in range(nenv):
            self.training_data[i].cutoffs = np.array(
                new_cutoffs, dtype=np.float)
            self.training_data[i].cutoffs_mask = hm
            self.training_data[i].setup_mask()
            self.training_data[i].compute_env()

        # Ensure that training data and labels are still consistent
        _global_training_data[self.name] = self.training_data
        _global_training_labels[self.name] = self.training_labels_np

        self.cutoffs = np.array(new_cutoffs)

        if reset_L_alpha:
            del self.l_mat
            del self.ky_mat
            self.set_L_alpha()

        if train:
            self.train()

    def write_model(self, name: str, format: str = 'json'):
        """
        Write model in a variety of formats to a file for later re-use.
        Args:
            name (str): Output name.
            format (str): Output format.
        """

        if len(self.training_data) > 5000:
            np.save(f"{name}_ky_mat.npy", self.ky_mat)
            self.ky_mat_file = f"{name}_ky_mat.npy"
            del self.ky_mat
            del self.l_mat
            del self.alpha
            del self.ky_mat_inv

        supported_formats = ['json', 'pickle', 'binary']

        if format.lower() == 'json':
            with open(f'{name}.json', 'w') as f:
                json.dump(self.as_dict(), f, cls=NumpyEncoder)

        elif format.lower() == 'pickle' or format.lower() == 'binary':
            with open(f'{name}.pickle', 'wb') as f:
                pickle.dump(self, f)

        else:
            raise ValueError("Output format not supported: try from "
                             "{}".format(supported_formats))

        if len(self.training_data) > 5000:
            self.ky_mat = np.load(f"{name}_ky_mat.npy")
            self.compute_matrices()

    @staticmethod
    def from_file(filename: str, format: str = ''):
        """
        One-line convenience method to load a GP from a file stored using
        write_file

        Args:
            filename (str): path to GP model
            format (str): json or pickle if format is not in filename
        :return:
        """

        if '.json' in filename or 'json' in format:
            with open(filename, 'r') as f:
                gp_model = GaussianProcess.from_dict(json.loads(f.readline()))
                gp_model.check_instantiation()
                _global_training_data[gp_model.name] = gp_model.training_data
                _global_training_labels[gp_model.name] = \
                    gp_model.training_labels_np

        elif '.pickle' in filename or 'pickle' in format:
            with open(filename, 'rb') as f:
                gp_model = pickle.load(f)

                if ('name' not in gp_model.__dict__):
                    gp_model.name = 'default_gp'

                gp_model.check_instantiation()

                if len(gp_model.training_data) > 5000:
                    try:
                        gp_model.ky_mat = np.load(gp_model.ky_mat_file)
                        gp_model.compute_matrices()
                    except FileNotFoundError:
                        gp_model.ky_mat = None
                        gp_model.l_mat = None
                        gp_model.alpha = None
                        gp_model.ky_mat_inv = None
                        Warning("the covariance matrices are not loaded"
                                f"it can take extra long time to recompute")

        else:
            raise ValueError("Warning: Format unspecieified or file is not "
                             ".json or .pickle format.")

        _global_training_data[gp_model.name] = gp_model.training_data
        _global_training_labels[gp_model.name] = gp_model.training_labels_np

        return gp_model

    @property
    def training_statistics(self) -> dict:
        """
        Return a dictionary with statistics about the current training data.
        Useful for quickly summarizing info about the GP.
        :return:
        """

        data = {}

        data['N'] = len(self.training_data)

        # Count all of the present species in the atomic env. data
        present_species = []
        for env, _ in zip(self.training_data, self.training_labels):
            present_species.append(Z_to_element(env.structure.coded_species[
                env.atom]))

        # Summarize the relevant information
        data['species'] = set(present_species)
        data['envs_by_species'] = dict(Counter(present_species))

        return data

    @property
    def par(self):
        """
        Backwards compability attribute
        :return:
        """
        return self.parallel

    def __del__(self):
        if (self.name in _global_training_labels):
            _global_training_labels.pop(self.name, None)
            _global_training_data.pop(self.name, None)
