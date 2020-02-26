import time
import math
import pickle
import json

import numpy as np
import multiprocessing as mp

from copy import deepcopy
from typing import List, Callable
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.gp_algebra import get_neg_likelihood, \
                             get_like_from_ky_mat
from flare.gp_algebra import get_kernel_vector_par
from flare.gp_algebra import get_ky_mat_par
from flare.gp_algebra import get_ky_mat_update_par
from flare.gp_algebra import get_neg_like_grad
from flare.kernels.utils import str_to_kernels
from flare.util import NumpyEncoder
from flare.output import Output
import flare.cutoffs as cf


class GaussianProcess:
    """Gaussian process force field. Implementation is based on Algorithm 2.1
    (pg. 19) of "Gaussian Processes for Machine Learning" by Rasmussen and
    Williams.

    Args:
        kernel (Callable): Force/force kernel of the GP used to make force
            predictions.
        kernel_grad (Callable): Function that returns the gradient of the GP
            kernel with respect to the hyperparameters.
        hyps (np.ndarray): Hyperparameters of the GP.
        cutoffs (np.ndarray): Cutoffs of the GP kernel.
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
        par (bool, optional): If True, the covariance matrix K of the GP is
            computed in parallel. Defaults to False.
        n_cpus (int, optional): Number of cpus used for parallel
            calculations. Defaults to 1.
        output (Output, optional): Output object used to dump hyperparameters
            during optimization. Defaults to None.
    """

    def __init__(self, kernel: Callable,
                 kernel_grad: Callable, hyps: 'ndarray',
                 cutoffs: 'ndarray',
                 hyp_labels: List = None,
                 energy_force_kernel: Callable = None,
                 energy_kernel: Callable = None,
                 opt_algorithm: str = 'L-BFGS-B',
                 maxiter: int = 10, par: bool = False,
                 per_atom_par: bool = True,
                 n_cpus: int = 1, nsample: int = 100,
                 output: Output = None,
                 multihyps: bool = False, hyps_mask: dict = None):
        """Initialize GP parameters and training data."""

        self.kernel = kernel
        self.kernel_grad = kernel_grad
        self.energy_kernel = energy_kernel
        self.energy_force_kernel = energy_force_kernel
        self.kernel_name = kernel.__name__
        self.hyps = hyps
        self.hyp_labels = hyp_labels
        self.cutoffs = cutoffs
        self.algo = opt_algorithm
        self.bounds = None

        self.training_data = []
        self.training_labels = []
        self.training_labels_np = np.empty(0, )
        self.maxiter = maxiter
        self.par = par
        self.per_atom_par = per_atom_par
        self.n_cpus = n_cpus
        self.nsample = nsample
        self.output = output

        # Parameters set during training
        self.ky_mat = None
        self.l_mat = None
        self.alpha = None
        self.ky_mat_inv = None
        self.likelihood = None
        self.likelihood_gradient = None

        if multihyps is True and hyps_mask is None:
            raise ValueError("Warning! Multihyperparameter mode enabled,"
                             "but no configuration hyperparameter mask was "
                             "passed. Did you mean to set multihyps to False?")
        elif multihyps is False and hyps_mask is not None:
            raise ValueError("Warning! Multihyperparameter mode disabled,"
                             "but a configuration hyperparameter mask was "
                             "passed. Did you mean to set multihyps to True?")
        self.hyps_mask = None
        if (isinstance(hyps_mask, dict) and multihyps is True):
            self.multihyps = True

            assert 'nspec' in hyps_mask.keys(), "nspec key missing in " \
                                                "hyps_mask dictionary"
            assert 'spec_mask' in hyps_mask.keys(), "spec_mask key missing " \
                                                    "in hyps_mask dicticnary"

            self.hyps_mask = deepcopy(hyps_mask)

            nspec = hyps_mask['nspec']

            if ('nbond' in hyps_mask.keys()):
                n2b = self.hyps_mask['nbond']
                if (n2b>0):
                    assert (np.max(hyps_mask['bond_mask']) < n2b)
                    assert len(hyps_mask['bond_mask']) == nspec**2, \
                            f"wrong dimension of bond_mask: " \
                            f" {len(hyps_mask['bond_mask']) != {nspec**2}}"
            else:
                n2b = 0

            if ('ntriplet' in hyps_mask.keys()):
                n3b = self.hyps_mask['ntriplet']
                if (n3b>0):
                    assert (np.max(hyps_mask['triplet_mask']) < n3b)
                    assert len(hyps_mask['triplet_mask']) == nspec**3, \
                            f"wrong dimension of triplet_mask" \
                            f"{len(hyps_mask['triplet_mask']) != {nspec**3}}"
            else:
                n3b = 0

            assert ((n2b+n3b)>0)

            if ('map' in hyps_mask.keys()):
                assert ('original' in hyps_mask.keys()), \
                        "original hyper parameters have to be defined"
                # Ensure typed correctly as numpy array
                self.hyps_mask['original'] = np.array(hyps_mask['original'])

                assert (n2b*2+n3b*2+1) == len(hyps_mask['original']) , \
                        "the hyperparmeter length is inconsistent with the mask"
                assert len(hyps_mask['map']) == len(hyps), \
                        "the hyperparmeter length is inconsistent with the mask"
                if ((len(hyps_mask['original'])-1) not in hyps_mask['map']):
                    assert hyps_mask['train_noise'] is False, \
                            "train_noise should be False when noise is not in hyps"
            else:
                assert hyps_mask['train_noise'] is True, \
                       "train_noise should be True when map is not used"
                assert (n2b*2+n3b*2+1) == len(hyps), \
                        "the hyperparmeter length is inconsistent with the mask"

            if ('bounds' in hyps_mask.keys()):
                self.bounds = deepcopy(hyps_mask['bounds'])
        else:
            self.multihyps = False
            self.hyps_mask = None

    # TODO unit test custom range
    def update_db(self, struc: Structure, forces: List,
                  custom_range: List[int] = ()):
        """Given a structure and forces, add local environments from the
        structure to the training set of the GP.

        Args:
            struc (Structure): Input structure. Local environments of atoms
                in this structure will be added to the training set of the GP.

            forces (np.ndarray): Forces on atoms in the structure.

            custom_range (List[int]): Indices of atoms whose local
                environments will be added to the training set of the GP.
        """

        # By default, use all atoms in the structure
        noa = len(struc.positions)
        update_indices = custom_range or list(range(noa))

        for atom in update_indices:
            env_curr = AtomicEnvironment(struc, atom, self.cutoffs)
            forces_curr = np.array(forces[atom])

            self.training_data.append(env_curr)
            self.training_labels.append(forces_curr)

        # create numpy array of training labels
        self.training_labels_np = np.hstack(self.training_labels)

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

        x_0 = self.hyps

        args = (self.training_data, self.training_labels_np,
                self.kernel_grad, output,
                self.cutoffs, self.hyps_mask,
                self.n_cpus, self.nsample,
                print_progress)
        objective_func = get_neg_like_grad
        res = None

        if self.algo == 'L-BFGS-B':

            # bound signal noise below to avoid overfitting
            if (self.bounds is None):
                bounds = np.array([(1e-6, np.inf)] * len(x_0))
            else:
                bounds = self.bounds
            # bounds = np.array([(1e-6, np.inf)] * len(x_0))
            # bounds[-1] = [1e-6,np.inf]
            # Catch linear algebra errors and switch to BFGS if necessary
            try:
                res = minimize(get_neg_like_grad, x_0, args,
                               method='L-BFGS-B', jac=True, bounds=bounds,
                               options={'disp': False, 'gtol': grad_tol,
                                        'maxls': line_steps,
                                        'maxiter': self.maxiter})
            except:
                print("Warning! Algorithm for L-BFGS-B failed. Changing to "
                      "BFGS for remainder of run.")
                self.algo = 'BFGS'

        if custom_bounds is not None:
            res = minimize(get_neg_like_grad, x_0, args,
                           method='L-BFGS-B', jac=True, bounds=custom_bounds,
                           options={'disp': False, 'gtol': grad_tol,
                                    'maxls': line_steps,
                                    'maxiter': self.maxiter})

        elif self.algo == 'BFGS':
            res = minimize(get_neg_like_grad, x_0, args,
                           method='BFGS', jac=True,
                           options={'disp': False, 'gtol': grad_tol,
                                    'maxiter': self.maxiter})

        elif self.algo == 'nelder-mead':
            res = minimize(get_neg_likelihood, x_0, args,
                           method='nelder-mead',
                           options={'disp': False,
                                    'maxiter': self.maxiter,
                                    'xtol': x_tol})
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

        # check that alpha is up to date with training set
        if self.alpha is None or 3 * len(self.training_data) != len(
                self.alpha):
            self.update_L_alpha()

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

        # Kernel vector allows for evaluation of atomic environments.
        if (self.par and not self.per_atom_par):
            n_cpus = self.n_cpus
        else:
            n_cpus = 1

        k_v = get_kernel_vector_par(self.training_data, self.kernel,
                                    x_t, d,
                                    self.hyps,
                                    cutoffs=self.cutoffs,
                                    hyps_mask=self.hyps_mask,
                                    n_cpus=self.n_cpus,
                                    nsample=self.nsample)

        # Guarantee that alpha is up to date with training set
        if  self.alpha is None or\
                (3 * len(self.training_data) != len(self.alpha)):
            self.update_L_alpha()

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance without cholesky (possibly faster)
        # pass args to kernel based on if mult. hyperparameters in use
        if (self.multihyps):
             self_kern = self.kernel(x_t, x_t, d, d, self.hyps,
                                     self.cutoffs, hyps_mask=self.hyps_mask)
        else:
             self_kern = self.kernel(x_t, x_t, d, d, self.hyps,
                                     self.cutoffs)

        pred_var = self_kern - \
            np.matmul(np.matmul(k_v, self.ky_mat_inv), k_v)

        return pred_mean, pred_var

    def predict_local_energy(self, x_t: AtomicEnvironment) -> float:
        """Predict the local energy of a local environment.

        Args:
            x_t (AtomicEnvironment): Input local environment.

        Return:
            float: Local energy predicted by the GP.
        """

        k_v = self.en_kern_vec(x_t)
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

        # get kernel vector
        k_v = self.en_kern_vec(x_t)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        if (self.multihyps):
             self_kern = self.energy_kernel(x_t, x_t, self.hyps,
                                            self.cutoffs, hyps_mask=self.hyps_mask)
        else:
             self_kern = self.energy_kernel(x_t, x_t, self.hyps,
                                            self.cutoffs)
        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def get_kernel_vector(self, x: AtomicEnvironment,
                          d_1: int):
        """
        Compute kernel vector, comparing input environment to all environments
        in the GP's training set.

        Args:
            x (AtomicEnvironment): Local environment to compare against
                the training environments.
            d_1 (int): Cartesian component of the kernel (1=x, 2=y, 3=z).

        Return:
            np.ndarray: Kernel vector.
        """

        ds = [1, 2, 3]
        size = len(self.training_data) * 3
        k_v = np.zeros(size, )

        if (self.multihyps):
            for m_index in range(size):
                x_2 = self.training_data[int(math.floor(m_index / 3))]
                d_2 = ds[m_index % 3]
                k_v[m_index] = self.kernel(x, x_2, d_1, d_2,
                                           self.hyps, self.cutoffs,
                                           hyps_mask=self.hyps_mask)
        else:
            for m_index in range(size):
                x_2 = self.training_data[int(math.floor(m_index / 3))]
                d_2 = ds[m_index % 3]
                k_v[m_index] = self.kernel(x, x_2, d_1, d_2,
                                           self.hyps, self.cutoffs)

        return k_v

    def en_kern_vec(self, x: AtomicEnvironment):
        """Compute the vector of energy/force kernels between an atomic
        environment and the environments in the training set.

        Args:
            x (AtomicEnvironment): Local environment to compare against
                the training environments.

        Return:
            np.ndarray: Kernel vector.
        """

        ds = [1, 2, 3]
        size = len(self.training_data) * 3
        k_v = np.zeros(size, )

        if (self.multihyps):
            for m_index in range(size):
                x_2 = self.training_data[int(math.floor(m_index / 3))]
                d_2 = ds[m_index % 3]
                k_v[m_index] = self.energy_force_kernel(x_2, x, d_2,
                                                        self.hyps, self.cutoffs,
                                                        hyps_mask=self.hyps_mask)
        else:
            for m_index in range(size):
                x_2 = self.training_data[int(math.floor(m_index / 3))]
                d_2 = ds[m_index % 3]
                k_v[m_index] = self.energy_force_kernel(x_2, x, d_2,
                                                        self.hyps, self.cutoffs)

        return k_v

    def set_L_alpha(self):
        """
        Invert the covariance matrix, setting L (a lower triangular
        matrix s.t. L L^T = (K + sig_n^2 I)) and alpha, the inverse
        covariance matrix multiplied by the vector of training labels.
        The forces and variances are later obtained using alpha.
        """

        ky_mat = get_ky_mat_par(self.hyps,
                                self.training_data,
                                self.kernel,
                                cutoffs=self.cutoffs,
                                hyps_mask=self.hyps_mask,
                                n_cpus=self.n_cpus,
                                nsample=self.nsample)

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv

        self.likelihood = get_like_from_ky_mat(self.ky_mat, self.training_labels_np)

    def update_L_alpha(self):
        """
        Update the GP's L matrix and alpha vector without recalculating
        the entire covariance matrix K.
        """

        # Set L matrix and alpha if set_L_alpha has not been called yet
        if self.l_mat is None:
            self.set_L_alpha()
            return

        if (self.par and not self.per_atom_par):
            n_cpus=self.n_cpus
        else:
            n_cpus=1

        ky_mat = get_ky_mat_update_par(self.ky_mat, self.hyps,
                                       self.training_data,
                                       self.kernel,
                                       cutoffs=self.cutoffs,
                                       hyps_mask=self.hyps_mask,
                                       n_cpus=n_cpus,
                                       nsample=self.nsample)

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv

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

        if (self.multihyps):
            nspec = self.hyps_mask['nspec']
            thestr +=f'nspec: {nspec}\n'
            thestr +=f'spec_mask: \n'
            thestr += str(self.hyps_mask['spec_mask']) + '\n'

            nbond = self.hyps_mask['nbond']
            thestr +=f'nbond: {nbond}\n'
            if (nbond>0):
                thestr +=f'bond_mask: \n'
                thestr += str(self.hyps_mask['bond_mask']) + '\n'

            ntriplet = self.hyps_mask['ntriplet']
            thestr +=f'ntriplet: {ntriplet}\n'
            if (ntriplet>0):
                thestr +=f'triplet_mask: \n'
                thestr += str(self.hyps_mask['triplet_mask']) + '\n'

        return thestr

    def as_dict(self):
        """Dictionary representation of the GP model."""

        out_dict = deepcopy(dict(vars(self)))

        out_dict['training_data'] = [env.as_dict() for env in
                                     self.training_data]
        # Remove the callables
        del out_dict['kernel']
        del out_dict['kernel_grad']

        return out_dict

    @staticmethod
    def from_dict(dictionary):
        """Create GP object from dictionary representation."""

        multihyps = dictionary.get('multihyps', False)

        force_kernel, grad = str_to_kernels(dictionary['kernel_name'],
                                            multihyps,
                                            include_grad=True)

        if dictionary['energy_kernel'] is not None:
            energy_kernel = str_to_kernel(dictionary['energy_kernel'],
                                          multihyps)
        else:
            energy_kernel = None

        if dictionary['energy_force_kernel'] is not None:
            energy_force_kernel = \
                str_to_kernel(dictionary['energy_force_kernel'],
                              multihyps)
        else:
            energy_force_kernel = None

        new_gp = GaussianProcess(kernel=force_kernel,
                                 kernel_grad=grad,
                                 energy_kernel=energy_kernel,
                                 energy_force_kernel=energy_force_kernel,
                                 cutoffs=np.array(dictionary['cutoffs']),
                                 hyps=np.array(dictionary['hyps']),
                                 hyp_labels=dictionary['hyp_labels'],
                                 par=dictionary['par'],
                                 per_atom_par=dictionary.get('per_atom_par',True),
                                 n_cpus=dictionary.get('n_cpus') or dictionary.get('no_cpus'),
                                 maxiter=dictionary['maxiter'],
                                 opt_algorithm=dictionary['algo'],
                                 multihyps=multihyps,
                                 hyps_mask=dictionary.get('hyps_mask',None)
                                 )

        new_gp.training_data = [AtomicEnvironment.from_dict(env) for env in
                                dictionary['training_data']]
        new_gp.training_labels = deepcopy(dictionary['training_labels'])
        new_gp.training_labels_np = deepcopy(dictionary['training_labels_np'])

        new_gp.likelihood = dictionary['likelihood']
        new_gp.likelihood_gradient = dictionary['likelihood_gradient']
        new_gp.training_labels_np = np.hstack(new_gp.training_labels)

        # Save time by attempting to load in computed attributes
        if (len(new_gp.training_data)>5000):
            new_gp.ky_mat = np.load(dictionary['ky_mat_file'])
            new_gp.compute_matrices()
        else:
            new_gp.ky_mat_inv = np.array(dictionary['ky_mat_inv']) \
                if dictionary.get('ky_mat_inv') is not None else None
            new_gp.ky_mat = np.array(dictionary['ky_mat']) \
                if dictionary.get('ky_mat') is not None else None
            new_gp.l_mat = np.array(dictionary['l_mat']) \
                if dictionary.get('l_mat') is not None else None
            new_gp.alpha = np.array(dictionary['alpha']) \
                if dictionary.get('alpha') is not None  else None
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

    def write_model(self, name: str, format: str = 'json'):
        """
        Write model in a variety of formats to a file for later re-use.
        Args:
            name (str): Output name.
            format (str): Output format.
        """

        if (len(self.training_data)>5000):
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

        if (len(self.training_data)>5000):
            self.ky_mat = np.load(f"{name}_ky_mat.npy")
            self.compute_matrices()


    @staticmethod
    def from_file(filename: str, format: str=''):
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
                return GaussianProcess.from_dict(json.loads(f.readline()))

        elif '.pickle' in filename or 'pickle' in format:
            with open(filename, 'rb') as f:
                return pickle.load(f)

        else:
            raise ValueError("Warning: Format unspecified or file is not "
                             ".json or .pickle format.")
