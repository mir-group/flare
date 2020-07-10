import inspect
import json
import logging
import math
import pickle
import time

import multiprocessing as mp
import numpy as np

from collections import Counter
from copy import deepcopy
from numpy.random import random
from scipy.linalg import solve_triangular
from scipy.optimize import minimize, basinhopping, \
    differential_evolution, dual_annealing
from scipy.sparse import triu
from scipy.cluster.hierarchy import dendrogram, \
    linkage, fcluster, optimal_leaf_ordering
from typing import List, Callable, Union

from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.gp_algebra import get_like_from_mats, get_neg_like_grad, \
    force_force_vector, energy_force_vector, get_force_block, \
    get_ky_mat_update, _global_training_data, _global_training_labels, \
    _global_training_structures, _global_energy_labels, get_Ky_mat, \
    get_kernel_vector, en_kern_vec, kernel_distance_mat
from flare.kernels.utils import str_to_kernel_set, from_mask_to_args, kernel_str_to_array
from flare.output import Output, set_logger
from flare.parameters import Parameters
from flare.struc import Structure
from flare.utils.element_coder import NumpyEncoder, Z_to_element


class RobustBayesianCommitteeMachine(GaussianProcess):
    """Gaussian process force field. Implementation is based on Algorithm 2.1
    (pg. 19) of "Gaussian Processes for Machine Learning" by Rasmussen and
    Williams.

    Args:
        kernels (list, optional): Determine the type of kernels. Example:
            ['2', '3'], ['2', '3', 'mb'], ['2']. Defaults to ['2', '3']
        component (str, optional): Determine single- ("sc") or multi-
            component ("mc") kernel to use. Defaults to "mc"
        hyps (np.ndarray, optional): Hyperparameters of the GP.
        cutoffs (Dict, optional): Cutoffs of the GP kernel.
        hyp_labels (List, optional): List of hyperparameter labels. Defaults
            to None.
        opt_algorithm (str, optional): Hyperparameter optimization algorithm.
            Defaults to 'L-BFGS-B'.
        maxiter (int, optional): Maximum number of iterations of the
            hyperparameter optimization algorithm. Defaults to 10.
        parallel (bool, optional): If True, the covariance matrix K of the GP is
            computed in parallel. Defaults to False.
        n_cpus (int, optional): Number of cpus used for parallel
            calculations. Defaults to 1 (serial)
        n_sample (int, optional): Size of submatrix to use when parallelizing
            predictions.
        output (Output, optional): Output object used to dump hyperparameters
            during optimization. Defaults to None.
        hyps_mask (dict, optional): hyps_mask can set up which hyper parameter
            is used for what interaction. Details see kernels/mc_sephyps.py
        name (str, optional): Name for the GP instance.
    """

    def __init__(self, n_experts, ndata_per_expert, prior_variance,
                 per_expert_parallel=False,
                 **kwargs):

        self.n_experts = n_experts
        self.prior_variance = prior_variance
        self.log_prior_var = np.log(prior_variance)
        self.ndata_per_expert = ndata_per_expert
        self.per_expert_parallel = per_expert_parallel

        GaussianProcess.__init__(self, **kwargs)
        self.reset_container()

    def reset_container(self):

        self.training_data = []
        self.training_labels = []  # Forces acting on central atoms
        self.training_labels_np = []  # np.empty(0, )
        self.n_envs_prev = []  # len(self.training_data)

        # Attributes to accomodate energy labels:
        # self.training_structures = []  # Environments of each structure
        self.energy_labels = []  # Energies of training structures
        self.energy_labels_np = []
        self.all_labels = []

        # Parameters set during training
        self.ky_mat = []
        self.force_block = []
        self.energy_block = []
        self.force_energy_block = []
        self.l_mat = []
        self.alpha = []
        self.ky_mat_inv = []
        self.likelihood = []

        assert self.n_experts > 0
        for i in range(self.n_experts):
            self.add_container()

        self.current_expert = 0

    def check_instantiation(self):
        """
        Runs a series of checks to ensure that the user has not supplied
        contradictory arguments which will result in undefined behavior
        with multiple hyperparameters.
        :return:
        """

        if self.logger_name is None:
            if self.output is None:
                self.logger_name = self.name+"GaussianProcess"
                set_logger(self.logger_name, stream=True,
                           fileout_name=None, verbose=self.verbose)
            else:
                self.logger_name = self.output.basename+'log'
        logger = logging.getLogger(self.logger_name)

        # check whether it's be loaded before
        loaded = False
        if self.name+"_0" in _global_training_labels:
            if _global_training_labels.get(self.name+"_0", None) \
                    is not self.training_labels_np:
                loaded = True
        if self.name+"_0" in _global_energy_labels:
            if _global_energy_labels.get(self.name+"_0", None) \
                    is not self.energy_labels_np:
                loaded = True

        if loaded:

            base = f'{self.name}'
            count = 2
            while (self.name+"_0" in _global_training_labels and count < 100):
                time.sleep(random())
                self.name = f'{base}_{count}'
                logger.debug("Specified GP name is present in global memory; "
                             "Attempting to rename the "
                             f"GP instance to {self.name}")
                count += 1
            if (self.name+"_0" in _global_training_labels):
                milliseconds = int(round(time.time() * 1000) % 10000000)
                self.name = f"{base}_{milliseconds}"
                logger.debug("Specified GP name still present in global memory: "
                             f"renaming the gp instance to {self.name}")
            logger.debug(f"Final name of the gp instance is {self.name}")

        self.sync_data()

        self.hyps_mask = Parameters.check_instantiation(self.hyps, self.cutoffs,
                                                        self.kernels, self.hyps_mask)

    def find_expert_to_add(self):

        expert_id = self.current_expert
        if len(self.training_data[expert_id]) > self.ndata_per_expert:
            self.current_expert += 1
            expert_id = self.current_expert

        return expert_id

    def add_container(self):

        self.training_data += [[]]
        self.training_labels += [[]]
        self.training_labels_np += [np.empty(0, )]
        self.n_envs_prev += [0]

        self.training_structures += [[]]  # Environments of each structure
        self.energy_labels += [[]]  # Energies of training structures
        self.energy_labels_np += [np.empty(0, )]
        self.all_labels += [np.empty(0, )]

        self.ky_mat += [None]
        self.force_block += [None]
        self.energy_block += [None]
        self.force_energy_block += [None]
        self.l_mat += [None]
        self.alpha += [None]
        self.ky_mat_inv += [None]

        self.likelihood += [None]

    def update_db(self, struc: Structure, forces: List,
                  custom_range: List[int] = (), energy: float = None,
                  expert_id: int = None):
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

        if expert_id is None:
            expert_id = self.find_expert_to_add()

        if expert_id >= self.n_experts:
            for i in range(expert_id-self.n_experts+1):
                self.add_container()
                self.n_experts += 1

        print(f"adding data to expert {expert_id}")

        # By default, use all atoms in the structure
        noa = len(struc.positions)
        update_indices = custom_range or list(range(noa))

        # If forces are given, update the environment list.
        if forces is not None:
            for atom in update_indices:
                env_curr = \
                    AtomicEnvironment(struc, atom, self.cutoffs,
                                      cutoffs_mask=self.hyps_mask)
                forces_curr = np.array(forces[atom])

                self.training_data[expert_id].append(env_curr)
                self.training_labels[expert_id].append(forces_curr)

            self.training_labels_np[expert_id] = np.hstack(
                self.training_labels[expert_id])

        # If an energy is given, update the structure list.
        if energy is not None:
            raise NotImplementedError
            # structure_list = []  # Populate with all environments of the struc
            # for atom in range(noa):
            #     env_curr = \
            #         AtomicEnvironment(struc, atom, self.cutoffs,
            #                           cutoffs_mask=self.hyps_mask)
            #     structure_list.append(env_curr)

            # # self.energy_labels[expert_id].append(energy)
            # self.training_structures[expert_id].append(structure_list)
            # self.energy_labels_np[expert_id] = np.array(
            # self.energy_labels[expert_id])

        # update list of all labels
        self.all_labels[expert_id] = np.concatenate((self.training_labels_np[expert_id],
                                                     self.energy_labels_np[expert_id]))

    def add_one_env(self, env: AtomicEnvironment,
                    force, train: bool = False, expert_id=None, **kwargs):
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

        if expert_id is None:
            expert_id = self.find_expert_to_add()

        if expert_id >= self.n_experts:
            for i in range(expert_id-self.n_experts+1):
                self.add_container()
                self.n_experts += 1

        logger = logging.getLogger(self.logger_name)
        logger.debug(f"add environment to Expert {expert_id}")

        self.training_data[expert_id].append(env)
        self.training_labels[expert_id].append(force)
        self.training_labels_np[expert_id] = np.hstack(
            self.training_labels[expert_id])

        # update list of all labels
        self.all_labels[expert_id] = np.concatenate((self.training_labels_np[expert_id],
                                                     self.energy_labels_np[expert_id]))

        if train:
            self.train(**kwargs)

    def train(self, logger_name=None, custom_bounds=None,
              grad_tol: float = 1e-4,
              x_tol: float = 1e-5,
              line_steps: int = 20,
              print_progress: bool = False,
              **kwargs):
        """Train Gaussian Process model on training data. Tunes the
        hyperparameters to maximize the likelihood, then computes L and alpha
        (related to the covariance matrix of the training set).

        Args:
            logger (logging.Logger): logger object specifying where to write the
                progress of the optimization.
            custom_bounds (np.ndarray): Custom bounds on the hyperparameters.
            grad_tol (float): Tolerance of the hyperparameter gradient that
                determines when hyperparameter optimization is terminated.
            x_tol (float): Tolerance on the x values used to decide when
                Nelder-Mead hyperparameter optimization is terminated.
            line_steps (int): Maximum number of line steps for L-BFGS
                hyperparameter optimization.
        """

        verbose = "info"
        if print_progress:
            verbose = "info"
        if self.verbose.lower() == "debug":
            verbose = "debug"
        if logger_name is None:
            set_logger("gp_algebra", stream=False,
                       fileout_name="log.gp_algebra",
                       verbose=verbose)
            logger_name = "gp_algebra"

        disp = False  # print_progress

        if len(self.training_data) == 0 or len(self.training_labels) == 0:
            raise Warning("You are attempting to train a GP with no "
                          "training data. Add environments and forces "
                          "to the GP and try again.")

        self.sync_data()

        x_0 = self.hyps

        opt_algorithm = f'{self.opt_algorithm}'

        args = (self.n_experts, self.name, self.kernel_grad,
                logger_name, self.cutoffs, self.hyps_mask,
                self.n_cpus, self.n_sample, self.per_expert_parallel)
        func = rbcm_get_neg_like_grad

        if opt_algorithm in ['differential evolution', 'dual annealing']:
            args0 = (self.n_experts, self.name, self.kernel,
                     logger_name, self.cutoffs, self.hyps_mask,
                     self.n_cpus, self.n_sample, self.per_expert_parallel)
            func0 = rbcm_get_neg_like

        res = None

        if self.bounds is None:
            bounds = np.array([(1e-6, 100)] * len(x_0))
            if self.hyps_mask.get('train_noise', True):
                bounds[-1, 0] = 1e-3
                bounds[-1, 1] = self.prior_variance
        elif custom_bounds is not None:
            bounds = custom_bounds
        else:
            bounds = self.bounds

        if opt_algorithm == 'basin hopping':
            minimizer_kwargs = {"method": "L-BFGS-B", "jac": True,
                                "args": args, "maxiter": 200}
            res = basinhopping(func, x_0,
                               minimizer_kwargs=minimizer_kwargs,
                               niter=self.maxiter)

        if opt_algorithm == 'differential evolution':
            res = differential_evolution(func0, bounds, args=args0,
                                         maxiter=self.maxiter, polish=False,
                                         **kwargs)
            opt_algorithm = 'L-BFGS-B'

        if opt_algorithm == 'dual annealining':

            res = dual_annealing(func0, bounds, args=args0,
                                 maxiter=self.maxiter, x0=x_0, **kwargs)

        if opt_algorithm == 'L-BFGS-B':

            # bound signal noise below to avoid overfitting
            # Catch linear algebra errors and switch to BFGS if necessary
            try:
                res = minimize(func, x_0, args,
                               method='L-BFGS-B', jac=True, bounds=bounds,
                               options={'disp': disp, 'gtol': grad_tol,
                                        'maxls': line_steps,
                                        'maxiter': self.maxiter})
            except np.linalg.LinAlgError:
                logger = logging.getLogger(self.logger_name)
                logger.warning("Algorithm for L-BFGS-B failed. Changing to "
                               "BFGS for remainder of run.")
                opt_algorithm = 'BFGS'

        if opt_algorithm == 'BFGS':
            res = minimize(func, x_0, args,
                           method='BFGS', jac=True,
                           options={'disp': disp, 'gtol': grad_tol,
                                    'maxiter': self.maxiter})

        if res is None:
            raise RuntimeError("Optimization failed for some reason.")
        self.hyps = res.x
        self.set_L_alpha()
        self.total_likelihood = -res.fun
        self.total_likelihood_gradient = -res.jac

        return res

    def check_L_alpha(self):
        """
        Check that the alpha vector is up to date with the training set. If
        not, update_L_alpha is called.
        """

        self.sync_data()

        # Check that alpha is up to date with training set
        for i in range(self.n_experts):
            size3 = len(self.training_data[i])*3

            # If model is empty, then just return
            if size3 == 0:
                return

            if self.alpha[i] is None:
                self.update_L_alpha(i)
            elif size3 > self.alpha[i].shape[0]:
                self.update_L_alpha(i)
            elif size3 != self.alpha[i].shape[0]:
                self.set_L_alpha_part(i)

    def predict(self, x_t: AtomicEnvironment) -> [float, float]:
        """
        Predict a force component of the central atom of a local environment.

        Args:
            x_t (AtomicEnvironment): Input local environment.

        Return:
            (float, float): Mean and epistemic variance of the prediction.
        """

        # Kernel vector allows for evaluation of atomic environments.
        if self.parallel and not self.per_atom_par:
            n_cpus = self.n_cpus
        else:
            n_cpus = 1

        self.sync_data()

        k_v = []
        for i in range(self.n_experts):
            k_v += \
                [get_kernel_vector(f"{self.name}_{i}", self.kernel,
                                   self.energy_force_kernel,
                                   x_t, self.hyps, cutoffs=self.cutoffs,
                                   hyps_mask=self.hyps_mask, n_cpus=n_cpus,
                                   n_sample=self.n_sample)]

        # Guarantee that alpha is up to date with training set
        self.check_L_alpha()

        # get predictive mean
        variance_rbcm = 0
        mean = 0.0
        var = 0.0
        beta = 0.0

        args = from_mask_to_args(self.hyps, self.cutoffs, self.hyps_mask)

        for i in range(self.n_experts):

            mean_i = np.matmul(self.alpha[i], k_v[i])

            # get predictive variance without cholesky (possibly faster)
            # pass args to kernel based on if mult. hyperparameters in use

            self_kern = self.kernel(x_t, x_t, *args)
            var_i = np.diagonal(
                self_kern - np.matmul(np.matmul(k_v[i].T, self.ky_mat_inv[i]), k_v[i]))

            beta_i = 0.5*(self.log_prior_var - np.log(var_i))
            mean += beta_i / var_i * mean_i
            var += beta_i / var_i
            beta += beta_i

        var += (1-beta)/self.prior_variance
        pred_var = 1.0/var
        pred_mean = pred_var * mean

        return pred_mean, pred_var

    # def predict_local_energy(self, x_t: AtomicEnvironment) -> float:
    #     """Predict the local energy of a local environment.

    #     Args:
    #         x_t (AtomicEnvironment): Input local environment.

    #     Return:
    #         float: Local energy predicted by the GP.
    #     """

    #     if self.parallel and not self.per_atom_par:
    #         n_cpus = self.n_cpus
    #     else:
    #         n_cpus = 1

    #     _global_training_data[self.name] = self.training_data
    #     _global_training_labels[self.name] = self.training_labels_np

    #     k_v = en_kern_vec(self.name, self.energy_force_kernel,
    #                       self.energy_kernel,
    #                       x_t, self.hyps, cutoffs=self.cutoffs,
    #                       hyps_mask=self.hyps_mask, n_cpus=n_cpus,
    #                       n_sample=self.n_sample)

    #     pred_mean = np.matmul(k_v, self.alpha)

    #     return pred_mean

    # def predict_local_energy_and_var(self, x_t: AtomicEnvironment):
    #     """Predict the local energy of a local environment and its
    #     uncertainty.

    #     Args:
    #         x_t (AtomicEnvironment): Input local environment.

    #     Return:
    #         (float, float): Mean and predictive variance predicted by the GP.
    #     """

    #     if self.parallel and not self.per_atom_par:
    #         n_cpus = self.n_cpus
    #     else:
    #         n_cpus = 1

    #     # get kernel vector
    #     k_v = en_kern_vec(self.name, self.energy_force_kernel,
    #                       self.energy_kernel,
    #                       x_t, self.hyps, cutoffs=self.cutoffs,
    #                       hyps_mask=self.hyps_mask, n_cpus=n_cpus,
    #                       n_sample=self.n_sample)

    #     # get predictive mean
    #     pred_mean = np.matmul(k_v, self.alpha)

    #     # get predictive variance
    #     v_vec = solve_triangular(self.l_mat, k_v, lower=True)
    #     args = from_mask_to_args(self.hyps, self.cutoffs, self.hyps_mask)

    #     self_kern = self.energy_kernel(x_t, x_t, *args)

    #     pred_var = self_kern - np.matmul(v_vec, v_vec)

    #     return pred_mean, pred_var

    def set_L_alpha(self):

        self.sync_data()

        logger = logging.getLogger(self.logger_name)
        logger.debug("set_L_alpha")

        if self.per_expert_parallel and self.n_cpus > 1:

            time0 = time.time()
            with mp.Pool(processes=self.n_cpus) as pool:
                results = []
                for expert_id in range(self.n_experts):
                    results.append(
                        pool.apply_async(get_Ky_mat,
                                         (self.hyps,
                                          f"{self.name}_{expert_id}",
                                          self.kernel,
                                          self.energy_kernel,
                                          self.energy_force_kernel,
                                          self.energy_noise,
                                          self.cutoffs,
                                          self.hyps_mask, 1,
                                          self.n_sample)))
                for i in range(self.n_experts):
                    ky_mat = results[i].get()
                    self.compute_one_matrices(ky_mat, i)
                pool.close()
                pool.join()
            logger.debug(
                f"set_L_alpha with per_expert_par {time.time()-time0}")
        else:

            for expert_id in range(self.n_experts):

                logger.debug(f"compute L_alpha for {expert_id}")
                time0 = time.time()

                ky_mat = get_Ky_mat(self.hyps, f"{self.name}_{expert_id}", self.kernel,
                                    self.energy_kernel, self.energy_force_kernel,
                                    self.energy_noise,
                                    self.cutoffs, self.hyps_mask,
                                    self.n_cpus, self.n_sample)

                self.compute_one_matrices(ky_mat, expert_id)
                logger.debug(
                    f"{expert_id} compute_L_alpha {time.time()-time0} {len(self.training_data[expert_id])}")

        for expert_id in range(self.n_experts):
            self.likelihood[expert_id] = get_like_from_mats(self.ky_mat[expert_id],
                                                            self.l_mat[expert_id],
                                                            self.alpha[expert_id],
                                                            f"{self.name}_{expert_id}")

        self.total_likelihood = np.sum(self.likelihood)

    def set_L_alpha_part(self, expert_id):
        """
        Invert the covariance matrix, setting L (a lower triangular
        matrix s.t. L L^T = (K + sig_n^2 I)) and alpha, the inverse
        covariance matrix multiplied by the vector of training labels.
        The forces and variances are later obtained using alpha.
        """

        self.sync_one_data(expert_id)

        if self.per_expert_parallel and self.n_cpus > 1:
            n_cpus = 1
        else:
            n_cpus = self.n_cpus

        ky_mat = \
            get_Ky_mat(self.hyps, f"{self.name}_{expert_id}", self.kernel,
                       self.energy_kernel, self.energy_force_kernel,
                       self.energy_noise,
                       cutoffs=self.cutoffs, hyps_mask=self.hyps_mask,
                       n_cpus=n_cpus, n_sample=self.n_sample)

        self.compute_one_matrices(ky_mat, expert_id)

        self.likelihood[expert_id] = get_like_from_mats(self.ky_mat[expert_id],
                                                        self.l_mat[expert_id],
                                                        self.alpha[expert_id],
                                                        f"{self.name}_{expert_id}")

    def sync_data(self):
        for i in range(self.n_experts):
            self.sync_one_data(i)

    def unsync_one_data(self, expert_id):
        """ Reset global variables. """
        if len(self.training_data) > expert_id:
            _global_training_data.pop(f"{self.name}_{expert_id}", None)
            _global_training_labels.pop(f"{self.name}_{expert_id}", None)
            # _global_training_structures.pop(f"{self.name}_{expert_id}",None)
            _global_energy_labels.pop(f"{self.name}_{expert_id}", None)

    def sync_one_data(self, expert_id):
        """ Reset global variables. """
        if len(self.training_data) > expert_id:
            _global_training_data[f"{self.name}_{expert_id}"] = \
                self.training_data[expert_id]
            _global_training_labels[f"{self.name}_{expert_id}"] = \
                self.training_labels_np[expert_id]
            _global_training_structures[f"{self.name}_{expert_id}"] = \
                self.training_structures[expert_id]
            _global_energy_labels[f"{self.name}_{expert_id}"] = \
                self.energy_labels_np[expert_id]

    def write_model(self, name: str, format: str = 'json'):

        if np.sum(self.n_envs_prev) > 5000:

            np.savez(f"{name}_ky_mat.npz", self.ky_mat)
            self.ky_mat_file = f"{name}_ky_mat.npz"

            temp_ky_mat = self.ky_mat
            temp_l_mat = self.l_mat
            temp_alpha = self.alpha
            temp_ky_mat_inv = self.ky_mat_inv

            self.ky_mat = None
            self.l_mat = None
            self.alpha = None
            self.ky_mat_inv = None

        GaussianProcess.write_model(self, name, format)

        self.ky_mat = temp_ky_mat
        self.l_mat = temp_l_mat
        self.alpha = temp_alpha
        self.ky_mat_inv = temp_ky_mat_inv

    def update_L_alpha(self, expert_id):
        """
        Update the GP's L matrix and alpha vector without recalculating
        the entire covariance matrix K.
        """

        # Set L matrix and alpha if set_L_alpha has not been called yet
        if self.l_mat[expert_id] is None or np.array(self.ky_mat[expert_id]) is np.array(None):
            self.set_L_alpha_part(expert_id)
            return

        self.sync_one_data(expert_id)

        ky_mat = get_ky_mat_update(self.ky_mat[expert_id],
                                   self.n_envs_prev[expert_id],
                                   self.hyps,
                                   f"{self.name}_{expert_id}", self.kernel,
                                   self.energy_kernel,
                                   self.energy_force_kernel,
                                   self.energy_noise,
                                   cutoffs=self.cutoffs,
                                   hyps_mask=self.hyps_mask,
                                   n_cpus=self.n_cpus,
                                   n_sample=self.n_sample)

        self.compute_one_matrices(ky_mat, expert_id)

    def compute_one_matrices(self, ky_mat, expert_id):
        """
        When covariance matrix is known, reconstruct other matrices.
        Used in re-loading large GPs.
        :return:
        """

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.all_labels[expert_id])

        self.ky_mat[expert_id] = ky_mat
        self.l_mat[expert_id] = l_mat
        self.alpha[expert_id] = alpha
        self.ky_mat_inv[expert_id] = ky_mat_inv
        self.n_envs_prev[expert_id] = len(self.training_data[expert_id])

    def redistribute_training_data(self):
        """ redistribute data """

        joint_data = []
        joint_structures = []
        joint_labels = []
        for i in range(self.n_experts):
            joint_data += self.training_data[i]
            # joint_structures += self.training_structures[i]
            joint_labels += self.training_labels[i]
            self.unsync_one_data(i)

        self.n_experts = 1
        self.reset_container()

        _global_training_data[f"{self.name}_join"] = joint_data
        _global_training_structures[f"{self.name}_join"] = joint_structures

        kmat = kernel_distance_mat(self.hyps, self.name+"_join",
                                   self.energy_kernel, self.cutoffs,
                                   self.hyps_mask, self.n_cpus, self.n_sample)
        for i in range(kmat.shape[0]):
            norm = np.sqrt(kmat[i, i])
            kmat[i, :] /= norm
            kmat[:, i] /= norm

        iu1 = np.triu_indices(kmat.shape[0], 1)
        upper_triang = kmat[iu1]
        Z = linkage(upper_triang, 'average')
        dn = dendrogram(Z)
        new_indices = list(map(int, dn['ivl']))

        for i in new_indices:
            self.add_one_env(
                joint_data[i], joint_labels[i], train=False, expert_id=None)

        del _global_training_data[f"{self.name}_join"]
        del _global_training_structures[f"{self.name}_join"]
        del joint_data
        del joint_structures
        del joint_labels

        self.set_L_alpha()

        return kmat

    def compute_join_dist_mat(self, list_expert_id):
        """ Reset global variables. """
        joint_data = []
        joint_structures = []
        for i in list_expert_id:
            joint_data += self.training_data[expert_id]
            # joint_structures += self.training_structures[expert_id]
        _global_training_data[f"{self.name}_join"] = joint_data
        _global_training_structures[f"{self.name}_join"] = joint_labels
        kmat = kernel_distance_mat(self.hyps, self.name+"_join",
                                   self.energy_kernel, self.cutoffs,
                                   self.hyps_mask, self.n_cpus, self.n_sample)
        del _global_training_data[f"{self.name}_join"]
        del _global_training_structures[f"{self.name}_join"]
        del joint_data
        del joint_structures
        return kmat

    @property
    def training_statistics(self) -> dict:
        """
        Return a dictionary with statistics about the current training data.
        Useful for quickly summarizing info about the GP.
        :return:
        """

        data = {}

        # Count all of the present species in the atomic env. data
        present_species = []
        data['N'] = 0
        for i in range(self.n_experts):
            data['N'] += self.n_envs_prev[i]
            data[f'N_{i}'] = self.n_envs_prev[i]
            for env, _ in zip(self.training_data[i], self.training_labels[i]):
                present_species.append(Z_to_element(env.structure.coded_species[
                    env.atom]))

        # Summarize the relevant information
        data['species'] = list(set(present_species))
        data['envs_by_species'] = dict(Counter(present_species))

        return data

    def write_model(self, name: str, format: str = 'pickle'):
        """
        Write model in a variety of formats to a file for later re-use.
        Args:
            name (str): Output name.
            format (str): Output format.
        """

        supported_formats = ['json', 'pickle', 'binary']

        for detect in ['json', 'pickle', 'binary']:
            if detect in name.lower():
                format = detect
                break

        if format is None:
            format = 'pickle'

        if format.lower() == 'json':
            raise ValueError("Output format not supported: try from "
                             "{}".format(supported_formats))
            # with open(f'{name}.json', 'w') as f:
            #     json.dump(self.as_dict(), f, cls=NumpyEncoder)

        elif format.lower() == 'pickle' or format.lower() == 'binary':
            if '.pickle' != name[-7:]:
                name += '.pickle'
            with open(f'{name}', 'wb') as f:
                pickle.dump(self, f)

        else:
            raise ValueError("Output format not supported: try from "
                             "{}".format(supported_formats))

    def get_full_gp(self):

        gp_model = GaussianProcess(**self.__dict__)
        gp_model.training_data = []
        gp_model.training_labels = []
        gp_model.name = "rbcm_derived_gp"
        for i in range(self.n_experts):
            gp_model.training_data += self.training_data[i]
            gp_model.training_labels += self.training_labels[i]
            gp_model.training_labels_np = np.hstack(gp_model.training_labels)
            gp_model.all_labels = np.hstack((gp_model.training_labels_np,
                                             gp_model.energy_labels_np))
        gp_model.sync_data()
        return gp_model

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
            raise ValueError("Output format not supported: try from "
                             "{}".format(supported_formats))
            # with open(filename, 'r') as f:
            #     gp_model = GaussianProcess.from_dict(json.loads(f.readline()))

        elif '.pickle' in filename or 'pickle' in format:
            with open(filename, 'rb') as f:

                gp_model = pickle.load(f)

                GaussianProcess.backward_arguments(
                    gp_model.__dict__, gp_model.__dict__)

                GaussianProcess.backward_attributes(gp_model.__dict__)

        else:
            raise ValueError("Warning: Format unspecieified or file is not "
                             ".json or .pickle format.")

        # # TO DO, be careful of this one
        # gp_model.check_instantiation()
        gp_model.sync_data()

        return gp_model

    def __str__(self):
        """String representation of the GP model."""

        thestr = "GaussianProcess Object\n"
        thestr += f'Number of cpu cores: {self.n_cpus}\n'
        thestr += f'Kernel: {self.kernels}\n'
        thestr += f"Training points: {len(self.training_data)}\n"
        thestr += f'Cutoffs: {self.cutoffs}\n'

        thestr += f'Number of hyperparameters: {len(self.hyps)}\n'
        thestr += f'Hyperparameters_array: {str(self.hyps)}\n'
        thestr += 'Hyperparameters: \n'
        if self.hyp_labels is None:
            # Put unlabeled hyperparameters on one line
            thestr = thestr[:-1]
            thestr += str(self.hyps) + '\n'
        else:
            for hyp, label in zip(self.hyps, self.hyp_labels):
                thestr += f"{label}: {hyp} \n"

        for k in self.hyps_mask:
            thestr += f'Hyps_mask {k}: {self.hyps_mask[k]} \n'

        return thestr


def rbcm_get_neg_like_grad(hyps, n_experts, name, kernel_grad, logger_name, cutoffs, hyps_mask, n_cpus, n_sample, per_expert_parallel):

    neg_like = 0
    neg_like_grad = None

    logger = logging.getLogger(logger_name)
    time0 = time.time()
    if per_expert_parallel and n_cpus > 1:

        with mp.Pool(processes=n_cpus) as pool:

            results = []
            for i in range(n_experts):
                results.append(
                    pool.apply_async(get_neg_like_grad,
                                     (hyps, f"{name}_{i}",
                                      kernel_grad, logger_name,
                                      cutoffs, hyps_mask, 1,
                                      n_sample)))
            for i in range(n_experts):
                chunk = results[i].get()
                neg_like_, neg_like_grad_ = chunk
                neg_like += neg_like_
                if neg_like_grad is None:
                    neg_like_grad = neg_like_grad_
                else:
                    neg_like_grad += neg_like_grad_
            pool.close()
            pool.join()
    else:
        for i in range(n_experts):
            neg_like_, neg_like_grad_ = get_neg_like_grad(hyps, f"{name}_{i}", kernel_grad, logger_name,
                                                          cutoffs, hyps_mask, n_cpus,
                                                          n_sample)
            neg_like += neg_like_
            if neg_like_grad is None:
                neg_like_grad = neg_like_grad_
            else:
                neg_like_grad += neg_like_grad_

    logger.info('')
    logger.info(f'Hyperparameters: {list(hyps)}')
    logger.info(f'Total Likelihood: {-neg_like}')
    logger.info(f'Total Likelihood Gradient: {list(neg_like_grad)}')
    logger.info(f"One step {time.time()-time0}")

    ohyps, label = Parameters.get_hyps(
        hyps_mask, hyps, constraint=False,
        label=True)
    if label:
        logger.info(f'oHyp_array: {list(ohyps)}')
        for i, l in enumerate(label):
            logger.info(f'oHyp {l}: {ohyps[i]}')

    return neg_like, neg_like_grad


def rbcm_get_neg_like(hyps, n_experts, name, force_kernel, logger_name, cutoffs, hyps_mask, n_cpus, n_sample, per_expert_parallel):

    neg_like = 0
    neg_like_grad = None

    logger = logging.getLogger(logger_name)
    time0 = time.time()
    if per_expert_parallel and n_cpus > 1:

        with mp.Pool(processes=n_cpus) as pool:

            results = []
            for i in range(n_experts):
                results.append(
                    pool.apply_async(get_neg_like,
                                     (hyps, f"{name}_{i}",
                                      force_kernel, logger_name,
                                      cutoffs, hyps_mask, 1,
                                      n_sample)))
            for i in range(n_experts):
                chunk = results[i].get()
                neg_like_ = chunk
                neg_like += neg_like_
            pool.close()
            pool.join()
    else:
        for i in range(n_experts):
            neg_like_ = get_neg_like(hyps, f"{name}_{i}", force_kernel, logger_name,
                                     cutoffs, hyps_mask, n_cpus,
                                     n_sample)
            neg_like += neg_like_

    logger.info('')
    logger.info(f'Hyperparameters: {list(hyps)}')
    logger.info(f'Total Likelihood: {-neg_like}')
    logger.info(f"One step {time.time()-time0}")

    ohyps, label = Parameters.get_hyps(
        hyps_mask, hyps, constraint=False,
        label=True)
    if label:
        logger.info(f'oHyp_array: {list(ohyps)}')
        for i, l in enumerate(label):
            logger.info(f'oHyp {l}: {ohyps[i]}')

    return neg_like
