import json
import json
import logging
import multiprocessing as mp
import pickle
import time
from collections import Counter
from copy import deepcopy
from typing import List, Union, Tuple

import numpy as np
from flare.env import AtomicEnvironment
from flare.gp_algebra import get_like_from_mats, get_neg_like_grad, \
    get_ky_mat_update, _global_training_data, _global_training_labels, \
    _global_training_structures, _global_energy_labels, get_Ky_mat, \
    get_kernel_vector, en_kern_vec, efs_kern_vec
from flare.kernels.utils import str_to_kernel_set, from_mask_to_args, \
    kernel_str_to_array
from flare.output import Output, set_logger
from flare.parameters import Parameters
from flare.struc import Structure
from flare.utils.element_coder import NumpyEncoder, Z_to_element
from numpy.random import random
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


class GaussianProcess:
    """Gaussian process force field. Implementation is based on Algorithm 2.1
    (pg. 19) of "Gaussian Processes for Machine Learning" by Rasmussen and
    Williams.

    Methods within GaussianProcess allow you to make predictions on
    AtomicEnvironment objects (see env.py) generated from
    FLARE Structures (see struc.py), and after data points are added,
    optimize hyperparameters based on available training data (train method).

    Args:
        kernels (list, optional): Determine the type of kernels. Example:
            ['twbody', 'threebody'], ['2', '3', 'mb'], ['2']. Defaults to [
            'twboody', 'threebody']
        component (str, optional): Determine single- ("sc") or multi-
            component ("mc") kernel to use. Defaults to "mc"
        hyps (np.ndarray, optional): Hyperparameters of the GP.
        cutoffs (Dict, optional): Cutoffs of the GP kernel. For simple hyper-
            parameter setups, formatted like {"twobody":7, "threebody":4.5}, 
            etc.
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
        name (str, optional): Name for the GP instance which dictates global
            memory access.
    """

    def __init__(self, kernels: List[str] = None,
                 component: str = 'mc',
                 hyps: 'ndarray' = None, cutoffs: dict = None,
                 hyps_mask: dict = None,
                 hyp_labels: List = None, opt_algorithm: str = 'L-BFGS-B',
                 maxiter: int = 10, parallel: bool = False,
                 per_atom_par: bool = True, n_cpus: int = 1,
                 n_sample: int = 100, output: Output = None,
                 name="default_gp",
                 energy_noise: float = 0.01, **kwargs, ):
        """Initialize GP parameters and training data."""

        # load arguments into attributes
        self.name = name

        self.output = output
        self.opt_algorithm = opt_algorithm

        self.per_atom_par = per_atom_par
        self.maxiter = maxiter

        # set up parallelization
        self.n_cpus = n_cpus
        self.n_sample = n_sample
        self.parallel = parallel

        self.component = component
        self.kernels = ['twobody', 'threebody'] if kernels is None else \
            kernel_str_to_array(''.join(kernels))
        self.cutoffs = {} if cutoffs is None else cutoffs
        self.hyp_labels = hyp_labels
        self.hyps_mask = {} if hyps_mask is None else hyps_mask
        self.hyps = hyps

        GaussianProcess.backward_arguments(kwargs, self.__dict__)
        GaussianProcess.backward_attributes(self.__dict__)

        # ------------  "computed" attributes ------------

        if self.output is None:
            self.logger_name = self.name + "GaussianProcess"
            set_logger(self.logger_name, stream=True,
                       fileout_name=None, verbose="info")
        else:
            self.logger_name = self.output.basename + 'log'

        if self.hyps is None:
            # If no hyperparameters are passed in, assume 2 hyps for each
            # kernel, plus one noise hyperparameter, and use a guess value
            self.hyps = np.array([0.1] * (1 + 2 * len(self.kernels)))
        else:
            self.hyps = np.array(self.hyps, dtype=np.float64)

        kernel, grad, ek, efk, efs_e, efs_f, efs_self = \
            str_to_kernel_set(self.kernels, self.component, self.hyps_mask)
        self.kernel = kernel
        self.kernel_grad = grad
        self.energy_force_kernel = efk
        self.energy_kernel = ek
        self.efs_energy_kernel = efs_e
        self.efs_force_kernel = efs_f
        self.efs_self_kernel = efs_self
        self.kernels = kernel_str_to_array(kernel.__name__)

        # parallelization
        if self.parallel:
            if self.n_cpus is None:
                self.n_cpus = mp.cpu_count()
            else:
                self.n_cpus = n_cpus
        else:
            self.n_cpus = 1

        self.training_data = []  # Atomic environments
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
        self.l_mat_inv = None
        self.alpha = None
        self.ky_mat_inv = None
        self.likelihood = None
        self.likelihood_gradient = None
        self.bounds = None

        # File used for reading / writing model if model is large
        self.ky_mat_file = None

        if self.logger_name is None:
            if self.output is None:
                self.logger_name = self.name+"GaussianProcess"
                set_logger(self.logger_name, stream=True,
                           fileout_name=None, verbose="info")
            else:
                self.logger_name = self.output.basename+'log'
        logger = logging.getLogger(self.logger_name)

        if self.cutoffs == {}:
            # If no cutoffs are passed in, assume 7 A for 2 body, 3.5 for
            # 3-body.
            cutoffs = {}
            if 'twobody' in self.kernels:
                cutoffs['twobody'] = 7
            if 'threebody' in self.kernels:
                cutoffs['threebody'] = 3.5
            if 'manybody' in self.kernels:
                raise ValueError("No cutoff was set for the manybody kernel."
                                 "A default value will not be set by default.")

            self.cutoffs = cutoffs
            logger.warning("Warning: No cutoffs were set for your GP."
                           "Default values have been assigned but you "
                           "should think carefully about which are "
                           "appropriate for your use case.")

        self.check_instantiation()

    def check_instantiation(self):
        """
        Runs a series of checks to ensure that the user has not supplied
        contradictory arguments which will result in undefined behavior
        with multiple hyperparameters.
        :return:
        """
        logger = logging.getLogger(self.logger_name)

        # check whether it's be loaded before
        loaded = False
        if self.name in _global_training_labels:
            if _global_training_labels.get(self.name,
                                           None) is not self.training_labels_np:
                loaded = True
        if self.name in _global_energy_labels:
            if _global_energy_labels.get(self.name,
                                         None) is not self.energy_labels_np:
                loaded = True

        if loaded:

            base = f'{self.name}'
            count = 2
            while self.name in _global_training_labels and count < 100:
                time.sleep(random())
                self.name = f'{base}_{count}'
                logger.debug("Specified GP name is present in global memory; "
                             "Attempting to rename the "
                             f"GP instance to {self.name}")
                count += 1
            if self.name in _global_training_labels:
                milliseconds = int(round(time.time() * 1000) % 10000000)
                self.name = f"{base}_{milliseconds}"
                logger.debug(
                    "Specified GP name still present in global memory: "
                    f"renaming the gp instance to {self.name}")
            logger.debug(f"Final name of the gp instance is {self.name}")

        self.sync_data()

        self.hyps_mask = Parameters.check_instantiation(
            hyps=self.hyps, cutoffs=self.cutoffs, kernels=self.kernels,
            param_dict=self.hyps_mask)

        self.bounds = deepcopy(self.hyps_mask.get('bounds', None))

    def update_kernel(self, kernels: List[str], component: str = "mc",
                      hyps=None, cutoffs: dict = None,
                      hyps_mask: dict = None):
        kernel, grad, ek, efk, _, _, _ = str_to_kernel_set(
            kernels, component, hyps_mask)
        self.kernel = kernel
        self.kernel_grad = grad
        self.energy_force_kernel = efk
        self.energy_kernel = ek
        self.kernels = kernel_str_to_array(kernel.__name__)

        if hyps_mask is not None:
            self.hyps_mask = hyps_mask
        # Cutoffs argument will override hyps mask's cutoffs key, if present
        if isinstance(hyps_mask, dict) and cutoffs is None:
            cutoffs = hyps_mask.get('cutoffs', None)

        if cutoffs is not None:
            if self.cutoffs != cutoffs:
                self.adjust_cutoffs(cutoffs, train=False,
                                    new_hyps_mask=hyps_mask)
            self.cutoffs = cutoffs

        if isinstance(hyps_mask, dict) and hyps is None:
            hyps = hyps_mask.get('hyps', None)

        if hyps is not None:
            self.hyps = hyps

    def update_db(self, struc: Structure, forces: List,
                  custom_range: List[int] = (), energy: float = None):
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
                    AtomicEnvironment(struc, atom, self.cutoffs,
                                      cutoffs_mask=self.hyps_mask)
                forces_curr = np.array(forces[atom])

                self.training_data.append(env_curr)
                self.training_labels.append(forces_curr)

            # create numpy array of training labels
            self.training_labels_np = np.hstack(self.training_labels)

        # If an energy is given, update the structure list.
        if energy is not None:
            structure_list = []  # Populate with all environments of the struc
            for atom in range(noa):
                env_curr = \
                    AtomicEnvironment(struc, atom, self.cutoffs,
                                      cutoffs_mask=self.hyps_mask)
                structure_list.append(env_curr)

            self.energy_labels.append(energy)
            self.training_structures.append(structure_list)
            self.energy_labels_np = np.array(self.energy_labels)

        # update list of all labels
        self.all_labels = np.concatenate((self.training_labels_np,
                                          self.energy_labels_np))
        self.sync_data()

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
        self.sync_data()

        # update list of all labels
        self.all_labels = np.concatenate((self.training_labels_np,
                                          self.energy_labels_np))

        if train:
            self.train(**kwargs)

    def train(self, logger_name: str = None, custom_bounds=None,
              grad_tol: float = 1e-4,
              x_tol: float = 1e-5,
              line_steps: int = 20,
              print_progress: bool = False):
        """Train Gaussian Process model on training data. Tunes the
        hyperparameters to maximize the likelihood, then computes L and alpha
        (related to the covariance matrix of the training set).

        Args:
            logger (logging.logger): logger object specifying where to write the
                progress of the optimization.
            custom_bounds (np.ndarray): Custom bounds on the hyperparameters.
            grad_tol (float): Tolerance of the hyperparameter gradient that
                determines when hyperparameter optimization is terminated.
            x_tol (float): Tolerance on the x values used to decide when
                Nelder-Mead hyperparameter optimization is terminated.
            line_steps (int): Maximum number of line steps for L-BFGS
                hyperparameter optimization.
                :param logger_name:
                :param print_progress:
        """

        verbose = "warning"
        if print_progress:
            verbose = "info"
        if logger_name is None:
            set_logger("gp_algebra", stream=True,
                       fileout_name="log.gp_algebra",
                       verbose=verbose)
            logger_name = "gp_algebra"

        disp = print_progress

        if len(self.training_data) == 0 or len(self.training_labels) == 0:
            raise Warning("You are attempting to train a GP with no "
                          "training data. Add environments and forces "
                          "to the GP and try again.")

        x_0 = self.hyps

        args = (self.name, self.kernel_grad, logger_name,
                self.cutoffs, self.hyps_mask,
                self.n_cpus, self.n_sample)

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
                               options={'disp': disp, 'gtol': grad_tol,
                                        'maxls': line_steps,
                                        'maxiter': self.maxiter})
            except np.linalg.LinAlgError:
                logger = logging.getLogger(self.logger_name)
                logger.warning("Algorithm for L-BFGS-B failed. Changing to "
                               "BFGS for remainder of run.")
                self.opt_algorithm = 'BFGS'

        if custom_bounds is not None:
            res = minimize(get_neg_like_grad, x_0, args,
                           method='L-BFGS-B', jac=True, bounds=custom_bounds,
                           options={'disp': disp, 'gtol': grad_tol,
                                    'maxls': line_steps,
                                    'maxiter': self.maxiter})

        elif self.opt_algorithm == 'BFGS':
            res = minimize(get_neg_like_grad, x_0, args,
                           method='BFGS', jac=True,
                           options={'disp': disp, 'gtol': grad_tol,
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
        size3 = len(self.training_data) * 3 + len(self.training_structures)

        # If model is empty, then just return
        if size3 == 0:
            return

        if self.alpha is None:
            self.update_L_alpha()
        elif size3 > self.alpha.shape[0]:
            self.update_L_alpha()
        elif size3 != self.alpha.shape[0]:
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

        self.sync_data()

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
        args = from_mask_to_args(self.hyps, self.cutoffs, self.hyps_mask)

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

        self.sync_data()

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

        self.sync_data()

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
        args = from_mask_to_args(self.hyps, self.cutoffs, self.hyps_mask)

        self_kern = self.energy_kernel(x_t, x_t, *args)

        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def predict_efs(self, x_t: AtomicEnvironment):
        """Predict the local energy, forces, and partial stresses of an
            atomic environment and their predictive variances."""

        # Kernel vector allows for evaluation of atomic environments.
        if self.parallel and not self.per_atom_par:
            n_cpus = self.n_cpus
        else:
            n_cpus = 1

        _global_training_data[self.name] = self.training_data
        _global_training_labels[self.name] = self.training_labels_np

        energy_vector, force_array, stress_array = \
            efs_kern_vec(self.name, self.efs_force_kernel,
                         self.efs_energy_kernel,
                         x_t, self.hyps, cutoffs=self.cutoffs,
                         hyps_mask=self.hyps_mask, n_cpus=n_cpus,
                         n_sample=self.n_sample)

        # Check that alpha is up to date with training set.
        self.check_L_alpha()

        # Compute mean predictions.
        en_pred = np.matmul(energy_vector, self.alpha)
        force_pred = np.matmul(force_array, self.alpha)
        stress_pred = np.matmul(stress_array, self.alpha)

        # Compute uncertainties.
        args = from_mask_to_args(self.hyps, self.cutoffs, self.hyps_mask)
        self_en, self_force, self_stress = self.efs_self_kernel(x_t, *args)

        en_var = self_en - \
                 np.matmul(np.matmul(energy_vector, self.ky_mat_inv),
                           energy_vector)
        force_var = self_force - \
                    np.diag(np.matmul(np.matmul(force_array, self.ky_mat_inv),
                                      force_array.transpose()))
        stress_var = self_stress - \
                     np.diag(
                         np.matmul(np.matmul(stress_array, self.ky_mat_inv),
                                   stress_array.transpose()))

        return en_pred, force_pred, stress_pred, en_var, force_var, stress_var

    def set_L_alpha(self):
        """
        Invert the covariance matrix, setting L (a lower triangular
        matrix s.t. L L^T = (K + sig_n^2 I)) and alpha, the inverse
        covariance matrix multiplied by the vector of training labels.
        The forces and variances are later obtained using alpha.
        """

        self.sync_data()

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
        self.sync_data()

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

        thestr = ''
        thestr += f'Number of cpu cores: {self.n_cpus}\n'
        thestr += f'Kernel: {self.kernels}\n'
        thestr += f"Training points: {len(self.training_data)}\n"
        thestr += f'Cutoffs: {self.cutoffs}\n'

        thestr += f'Number of hyperparameters: {len(self.hyps)}\n'
        thestr += f'Hyperparameter array: {str(self.hyps)}\n'

        if self.hyp_labels is None:
            # Put unlabeled hyperparameters on one line
            thestr = thestr[:-1]
            thestr += str(self.hyps) + '\n'
        else:
            for hyp, label in zip(self.hyps, self.hyp_labels):
                thestr += f"{label}: {hyp} \n"

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
                    'energy_force_kernel', 'efs_energy_kernel',
                    'efs_force_kernel', 'efs_self_kernel']:
            if out_dict.get(key) is not None:
                del out_dict[key]

        return out_dict

    def sync_data(self):
        _global_training_data[self.name] = self.training_data
        _global_training_labels[self.name] = self.training_labels_np
        _global_training_structures[self.name] = self.training_structures
        _global_energy_labels[self.name] = self.energy_labels_np

    @staticmethod
    def from_dict(dictionary):
        """Create GP object from dictionary representation."""

        GaussianProcess.backward_arguments(dictionary, dictionary)
        GaussianProcess.backward_attributes(dictionary)

        new_gp = GaussianProcess(**dictionary)

        # Save time by attempting to load in computed attributes
        if 'training_data' in dictionary:
            new_gp.training_data = [AtomicEnvironment.from_dict(env) for env in
                                    dictionary['training_data']]
            new_gp.training_labels = deepcopy(dictionary['training_labels'])
            new_gp.training_labels_np = deepcopy(
                dictionary['training_labels_np'])
            new_gp.sync_data()

        # Reconstruct training structures.
        if 'training_structures' in dictionary:
            new_gp.training_structures = []
            for n, env_list in enumerate(dictionary['training_structures']):
                new_gp.training_structures.append([])
                for env_curr in env_list:
                    new_gp.training_structures[n].append(
                        AtomicEnvironment.from_dict(env_curr))
            new_gp.energy_labels = deepcopy(dictionary['energy_labels'])
            new_gp.energy_labels_np = deepcopy(dictionary['energy_labels_np'])

        new_gp.all_labels = np.concatenate((new_gp.training_labels_np,
                                            new_gp.energy_labels_np))

        new_gp.likelihood = dictionary.get('likelihood', None)
        new_gp.likelihood_gradient = dictionary.get(
            'likelihood_gradient', None)

        new_gp.n_envs_prev = len(new_gp.training_data)

        # Save time by attempting to load in computed attributes
        if dictionary.get('ky_mat_file'):
            try:
                new_gp.ky_mat = np.load(dictionary['ky_mat_file'])
                new_gp.compute_matrices()
                new_gp.ky_mat_file = None

            except FileNotFoundError:
                new_gp.ky_mat = None
                new_gp.l_mat = None
                new_gp.alpha = None
                new_gp.ky_mat_inv = None
                filename = dictionary.get('ky_mat_file')
                logger = logging.getLogger(new_gp.logger_name)
                logger.warning("the covariance matrices are not loaded"
                               f"because {filename} cannot be found")
        else:
            new_gp.ky_mat = np.array(dictionary['ky_mat']) \
                if dictionary.get('ky_mat') is not None else None
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
        """
        When covariance matrix is known, reconstruct other matrices.
        Used in re-loading large GPs.
        :return:
        """
        ky_mat = self.ky_mat

        if ky_mat is None or \
                (isinstance(ky_mat, np.ndarray) and not np.any(
                    ky_mat)):
            Warning("Warning: Covariance matrix was not loaded but "
                    "compute_matrices was called. Computing covariance "
                    "matrix and proceeding...")
            self.set_L_alpha()

        else:
            self.l_mat = np.linalg.cholesky(ky_mat)
            self.l_mat_inv = np.linalg.inv(self.l_mat)
            self.ky_mat_inv = self.l_mat_inv.T @ self.l_mat_inv
            self.alpha = np.matmul(self.ky_mat_inv, self.all_labels)

    def adjust_cutoffs(self,
                       new_cutoffs: Union[list, tuple, 'np.ndarray'] = None,
                       reset_L_alpha=True, train=True, new_hyps_mask=None):
        """
        Loop through atomic environment objects stored in the training data,
        and re-compute cutoffs for each. Useful if you want to gauge the
        impact of cutoffs given a certain training set! Unless you know
        *exactly* what you are doing for some development or test purpose,
        it is **highly** suggested that you call set_L_alpha and
        re-optimize your hyperparameters afterwards as is default here.

        A helpful way to update the cutoffs and kernel for an extant
        GP is to perform the following commands:
        >> hyps_mask = pm.as_dict()
        >> hyps = hyps_mask['hyps']
        >> cutoffs = hyps_mask['cutoffs']
        >> kernels = hyps_mask['kernels']
        >> gp_model.update_kernel(kernels, 'mc', hyps, cutoffs, hyps_mask)

        :param reset_L_alpha:
        :param train:
        :param new_hyps_mask:
        :param new_cutoffs:
        :return:
        """

        if new_hyps_mask is not None:
            hm = new_hyps_mask
            self.hyps_mask = new_hyps_mask
        else:
            hm = self.hyps_mask
        if new_cutoffs is None:
            try:
                new_cutoffs = hm['cutoffs']
            except KeyError:
                raise KeyError("New cutoffs not found in the hyps_mask"
                               "dictionary via call to 'cutoffs' key.")

        # update environment
        nenv = len(self.training_data)
        for i in range(nenv):
            self.training_data[i].cutoffs = new_cutoffs
            self.training_data[i].cutoffs_mask = hm
            self.training_data[i].setup_mask(hm)
            self.training_data[i].compute_env()

        # Ensure that training data and labels are still consistent
        self.sync_data()

        self.cutoffs = new_cutoffs

        if reset_L_alpha:
            del self.l_mat
            del self.ky_mat
            self.set_L_alpha()

        if train:
            self.train()

    def remove_force_data(self, indexes: Union[int, List[int]],
                          update_matrices: bool = True) -> Tuple[
        List[Structure],
        List['ndarray']]:
        """
        Remove force components from the model. Convenience function which
        deletes individual data points.

        Matrices should *always* be updated if you intend to use the GP to make
        predictions afterwards. This might be time consuming for large GPs,
        so, it is provided as an option, but, only do so with extreme caution.
        (Undefined behavior may result if you try to make predictions and/or
        add to the training set afterwards).

        Returns training data which was removed akin to a pop method, in order
        of lowest to highest index passed in.

        :param indexes: Indexes of envs in training data to remove.
        :param update_matrices: If false, will not update the GP's matrices
            afterwards (which can be time consuming for large models).
            This should essentially always be true except for niche development
            applications.
        :return:
        """

        # Listify input even if one integer
        if isinstance(indexes, int):
            indexes = [indexes]

        if max(indexes) > len(self.training_data):
            raise ValueError("Index out of range of data")

        if len(indexes) == 0:
            return [], []

        # Get in reverse order so that modifying higher indexes doesn't affect
        # lower indexes
        indexes.sort(reverse=True)
        removed_data = []
        removed_labels = []
        for i in indexes:
            removed_data.append(self.training_data.pop(i))
            removed_labels.append(self.training_labels.pop(i))

        self.training_labels_np = np.hstack(self.training_labels)
        self.all_labels = np.concatenate((self.training_labels_np,
                                          self.energy_labels_np))
        self.sync_data()

        if update_matrices:
            self.set_L_alpha()

        # Put removed data in order of lowest to highest index
        removed_data.reverse()
        removed_labels.reverse()

        return removed_data, removed_labels

    def write_model(self, name: str, format: str = None,
                    split_matrix_size_cutoff: int = 5000):
        """
        Write model in a variety of formats to a file for later re-use.
        JSON files are open to visual inspection and are easier to use
        across different versions of FLARE or GP implementations. However,
        they are larger and loading them in takes longer (by setting up a
        new GP from the specifications). Pickled files can be faster to
        read & write, and they take up less memory.

        Args:
            name (str): Output name.
            format (str): Output format.
            split_matrix_size_cutoff (int): If there are more than this
            number of training points in the set, save the matrices seperately.
        """

        if len(self.training_data) > split_matrix_size_cutoff:
            np.save(f"{name}_ky_mat.npy", self.ky_mat)
            self.ky_mat_file = f"{name}_ky_mat.npy"

            temp_ky_mat = self.ky_mat
            temp_l_mat = self.l_mat
            temp_alpha = self.alpha
            temp_ky_mat_inv = self.ky_mat_inv

            self.ky_mat = None
            self.l_mat = None
            self.alpha = None
            self.ky_mat_inv = None

        # Automatically detect output format from name variable

        for detect in ['json', 'pickle', 'binary']:
            if detect in name.lower():
                format = detect
                break

        if format is None:
            format = 'json'

        supported_formats = ['json', 'pickle', 'binary']

        if format.lower() == 'json':
            if '.json' != name[-5:]:
                name += '.json'
            with open(name, 'w') as f:
                json.dump(self.as_dict(), f, cls=NumpyEncoder)

        elif format.lower() == 'pickle' or format.lower() == 'binary':
            if '.pickle' != name[-7:]:
                name += '.pickle'
            with open(name, 'wb') as f:
                pickle.dump(self, f)

        else:
            raise ValueError("Output format not supported: try from "
                             "{}".format(supported_formats))

        if len(self.training_data) > split_matrix_size_cutoff:
            self.ky_mat = temp_ky_mat
            self.l_mat = temp_l_mat
            self.alpha = temp_alpha
            self.ky_mat_inv = temp_ky_mat_inv

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

        elif '.pickle' in filename or 'pickle' in format:
            with open(filename, 'rb') as f:

                gp_model = pickle.load(f)

                GaussianProcess.backward_arguments(
                    gp_model.__dict__, gp_model.__dict__)

                GaussianProcess.backward_attributes(gp_model.__dict__)

                if hasattr(gp_model, 'ky_mat_file') and gp_model.ky_mat_file:
                    try:
                        gp_model.ky_mat = np.load(gp_model.ky_mat_file,
                                                  allow_pickle=True)
                        gp_model.compute_matrices()
                    except FileNotFoundError:
                        gp_model.ky_mat = None
                        gp_model.l_mat = None
                        gp_model.alpha = None
                        gp_model.ky_mat_inv = None
                        Warning("the covariance matrices are not loaded" \
                                f"it can take extra long time to recompute")

        else:
            raise ValueError("Warning: Format unspecieified or file is not "
                             ".json or .pickle format.")

        gp_model.check_instantiation()
        return gp_model

    @property
    def training_statistics(self) -> dict:
        """
        Return a dictionary with statistics about the current training data.
        Useful for quickly summarizing info about the GP.
        :return:
        """

        data = dict()

        data['N'] = len(self.training_data)

        # Count all of the present species in the atomic env. data
        present_species = []
        for env, _ in zip(self.training_data, self.training_labels):
            present_species.append(Z_to_element(env.structure.coded_species[
                                                    env.atom]))

        # Summarize the relevant information
        data['species'] = list(set(present_species))
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
        if self is None:
            return
        if self.name in _global_training_labels:
            return _global_training_data.pop(self.name, None), \
                   _global_training_labels.pop(self.name, None)

    @staticmethod
    def backward_arguments(kwargs, new_args={}):
        """
        update the initialize arguments that were renamed
        """

        if 'kernel_name' in kwargs:
            DeprecationWarning(
                "kernel_name is being replaced with kernels")
            new_args['kernels'] = kernel_str_to_array(
                kwargs['kernel_name'])
            kwargs.pop('kernel_name')
        if 'nsample' in kwargs:
            DeprecationWarning("nsample is being replaced with n_sample")
            new_args['n_sample'] = kwargs['nsample']
            kwargs.pop('nsample')
        if 'par' in kwargs:
            DeprecationWarning("par is being replaced with parallel")
            new_args['parallel'] = kwargs['par']
            kwargs.pop('par')
        if 'no_cpus' in kwargs:
            DeprecationWarning("no_cpus is being replaced with n_cpu")
            new_args['n_cpus'] = kwargs['no_cpus']
            kwargs.pop('no_cpus')
        if 'multihyps' in kwargs:
            DeprecationWarning("multihyps is removed")
            kwargs.pop('multihyps')

        return new_args

    @staticmethod
    def backward_attributes(dictionary):
        """
        add new attributes to old instance
        or update attribute types
        """

        if 'name' not in dictionary:
            dictionary['name'] = 'default_gp'
        if 'per_atom_par' not in dictionary:
            dictionary['per_atom_par'] = True
        if 'optimization_algorithm' not in dictionary:
            dictionary['opt_algorithm'] = 'L-BFGS-B'
        if 'hyps_mask' not in dictionary:
            dictionary['hyps_mask'] = None
        if 'parallel' not in dictionary:
            dictionary['parallel'] = False
        if 'component' not in dictionary:
            dictionary['component'] = 'mc'

        if 'training_structures' not in dictionary:
            # Environments of each structure
            dictionary['training_structures'] = []
            dictionary['energy_labels'] = []  # Energies of training structures
            dictionary['energy_labels_np'] = np.empty(0, )

        if 'training_labels' not in dictionary:
            dictionary['training_labels'] = []
            dictionary['training_labels_np'] = np.empty(0, )

        if 'energy_noise' not in dictionary:
            dictionary['energy_noise'] = 0.01

        if not isinstance(dictionary['cutoffs'], dict):
            dictionary['cutoffs'] = Parameters.cutoff_array_to_dict(
                dictionary['cutoffs'])

        dictionary['hyps_mask'] = Parameters.backward(
            dictionary['kernels'], deepcopy(dictionary['hyps_mask']))

        if 'logger_name' not in dictionary:
            dictionary['logger_name'] = None
