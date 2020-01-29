"""Gaussian process model of the Born Oppenheimer potential energy surface."""
import time
import math
from copy import deepcopy
import pickle
import json

import numpy as np
import multiprocessing as mp

from typing import List, Callable
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.gp_algebra import get_neg_likelihood, \
                             get_like_from_ky_mat
from flare.gp_algebra_multi import get_kernel_vector_par
from flare.gp_algebra_multi import get_ky_mat_par
from flare.gp_algebra_multi import get_ky_mat_update_par
from flare.gp_algebra_multi import get_neg_like_grad
from flare.kernels import str_to_kernel
from flare.mc_simple import str_to_mc_kernel
from flare.mc_sephyps import str_to_mc_kernel as str_to_mc_sephyps_kernel
import flare.cutoffs as cf
from flare.util import NumpyEncoder

class GaussianProcess:
    """ Gaussian Process Regression Model.
    Implementation is based on Algorithm 2.1 (pg. 19) of
    "Gaussian Processes for Machine Learning" by Rasmussen and Williams"""

    def __init__(self, kernel: Callable,
                 kernel_grad: Callable, hyps,
                 cutoffs,
                 hyp_labels: List = None,
                 energy_force_kernel: Callable = None,
                 energy_kernel: Callable = None,
                 opt_algorithm: str = 'L-BFGS-B',
                 maxiter=10, par=False, per_atom_par=True,
                 ncpus=None, nsample=100,
                 output=None,
                 multihyps=False, hyps_mask=None):
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
        # self.data_store = []
        self.training_labels = []
        self.training_labels_np = np.empty(0, )
        self.maxiter = maxiter
        self.par = par
        self.per_atom_par = per_atom_par
        self.ncpus = ncpus
        self.nsample = 100
        self.output = output

        # Parameters set during training
        self.ky_mat = None
        self.l_mat = None
        self.alpha = None
        self.ky_mat_inv = None
        # self.l_mat_inv = None
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
                assert hyps_mask['train_noise'] is False, \
                       "train_noise should be False when map is not used"
                assert (n2b*2+n3b*2+1) == len(hyps), \
                        "the hyperparmeter length is inconsistent with the mask"

            if ('bounds' in hyps_mask.keys()):
                self.bounds = deepcopy(hyps_mask['bounds'])
        else:
            self.multihyps = False
            self.hyps_mask = None

    # TODO unit test custom range
    def update_db(self, struc: Structure, forces: list,
                  custom_range: List[int] = ()):
        """Given structure and forces, add to training set.
        :param struc: structure to add to db
        :type struc: Structure
        :param forces: list of corresponding forces to add to db
        :type forces: list<float>
        :param custom_range: Indices to use in lieu of the whole structure
        :type custom_range: List[int]
        """

        # By default, use all atoms in the structure
        noa = len(struc.positions)
        update_indices = custom_range or list(range(noa))

        for atom in update_indices:
            env_curr = AtomicEnvironment(struc, atom, self.cutoffs)
            forces_curr = np.array(forces[atom])
            # env_store = AtomicEnvironment(struc, atom, self.cutoffs)
            # del env_curr.structure

            # self.data_store.append(env_store)
            self.training_data.append(env_curr)
            self.training_labels.append(forces_curr)

        # create numpy array of training labels
        self.training_labels_np = np.hstack(self.training_labels)

    def add_one_env(self, env: AtomicEnvironment,
                    force, train: bool = False, **kwargs):
        """
        Tool to add a single environment / force pair into the training set.
        :param force:
        :param env:
        :param force: (x,y,z) component associated with environment
        :param train:
        :return:
        """
        # self.data_store.append(env)
        # light_env = deepcopy(env)
        # del light_env.structure
        # self.training_data.append(light_env)
        self.training_data.append(env)
        self.training_labels.append(force)
        self.training_labels_np = np.hstack(self.training_labels)

        if train:
            self.train(**kwargs)

    def train(self, output=None, custom_bounds=None,
              grad_tol: float = 1e-4,
              x_tol: float = 1e-5,
              line_steps: int = 20):
        """Train Gaussian Process model on training data. Tunes the \
hyperparameters to maximize the likelihood, then computes L and alpha \
(related to the covariance matrix of the training set)."""

        x_0 = self.hyps

        args = (self.training_data, self.training_labels_np,
                self.kernel_grad, output,
                self.cutoffs, self.hyps_mask,
                self.ncpus, self.nsample)
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
                res = minimize(objective_func, x_0, args,
                               method='L-BFGS-B', jac=True, bounds=bounds,
                               options={'disp': False, 'gtol': grad_tol,
                                        'maxls': line_steps,
                                        'maxiter': self.maxiter})
            except:
                print("Warning! Algorithm for L-BFGS-B failed. Changing to "
                      "BFGS for remainder of run.")
                self.algo = 'BFGS'

        if custom_bounds is not None:
            res = minimize(objective_func, x_0, args,
                           method='L-BFGS-B', jac=True, bounds=custom_bounds,
                           options={'disp': False, 'gtol': grad_tol,
                                    'maxls': line_steps,
                                    'maxiter': self.maxiter})

        elif self.algo == 'BFGS':
            res = minimize(objective_func, x_0, args,
                           method='BFGS', jac=True,
                           options={'disp': False, 'gtol': grad_tol,
                                    'maxiter': self.maxiter})

        elif self.algo == 'nelder-mead':
            res = minimize(objective_func, x_0, args,
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
        # Guarantee that alpha is up to date with training set
        if self.alpha is None or 3 * len(self.training_data) != len(
                self.alpha):
            self.update_L_alpha()

    def predict(self, x_t: AtomicEnvironment, d: int) -> [float, float]:
        """Predict force component of an atomic environment and its \
uncertainty."""

        # Kernel vector allows for evaluation of At. Env.
        if (self.par and not self.per_atom_par):
            ncpus = self.ncpus
        else:
            ncpus = 1

        k_v = get_kernel_vector_par(self.training_data, self.kernel,
                                    x_t, d,
                                    self.hyps,
                                    cutoffs=self.cutoffs,
                                    hyps_mask=self.hyps_mask,
                                    ncpus=self.ncpus,
                                    nsample=self.nsample)

        # Guarantee that alpha is up to date with training set
        assert ((self.alpha is not None) and (3 * len(self.training_data) == len(self.alpha)))

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance without cholesky (possibly faster)
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
        """Predict the local energy of an atomic environment.
        :param x_t: Atomic environment of test atom.
        :type x_t: AtomicEnvironment
        :return: local energy in eV (up to a constant).
        :rtype: float
        """

        k_v = self.en_kern_vec(x_t)
        pred_mean = np.matmul(k_v, self.alpha)

        return pred_mean

    def predict_local_energy_and_var(self, x_t: AtomicEnvironment):
        """Predict the local energy of an atomic environment and its \
uncertainty."""

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
        :param x: data point to compare against kernel matrix
        :type x: AtomicEnvironment
        :param d_1: Cartesian component of force vector to get (1=x,2=y,3=z)
        :type d_1: int
        :return: kernel vector
        :rtype: np.ndarray
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
        """Compute the vector of energy/force kernels between an atomic \
environment and the environments in the training set."""

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
        matrix s.t. L L^T = (K + sig_n^2 I) ) and alpha, the inverse
        covariance matrix multiplied by the vector of training labels.
        The forces and variances are later obtained using alpha.
        :return:
        """
        ky_mat = get_ky_mat_par(self.hyps,
                                self.training_data,
                                self.kernel,
                                cutoffs=self.cutoffs,
                                hyps_mask=self.hyps_mask,
                                ncpus=self.ncpus,
                                nsample=self.nsample)


        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv
        # self.l_mat_inv = l_mat_inv

        self.likelihood = get_like_from_ky_mat(self.ky_mat, self.training_labels_np)
        # self.likelihood_gradient = like_grad

    def update_L_alpha(self):
        """
        Update the GP's L matrix and alpha vector.
        """

        # Set L matrix and alpha if set_L_alpha has not been called yet
        if self.l_mat is None:
            self.set_L_alpha()
            return

        if (self.par and not self.per_atom_par):
            ncpus=self.ncpus
        else:
            ncpus=1
        ky_mat = get_ky_mat_update_par(self.ky_mat, self.hyps,
                                       self.training_data,
                                       self.kernel,
                                       cutoffs=self.cutoffs,
                                       hyps_mask=self.hyps_mask,
                                       ncpus=ncpus,
                                       nsample=self.nsample)

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv
        # self.l_mat_inv = l_mat_inv


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

        if 'mc' in dictionary['kernel_name']:
            if (dictionary.get('multihyps',False) is False):
                force_kernel, grad = \
                    str_to_mc_kernel(dictionary['kernel_name'], include_grad=True)
            else:
                force_kernel, grad = \
                    str_to_mc_sephyps_kernel(dictionary['kernel_name'],
                            include_grad=True)
        else:
            force_kernel, grad = str_to_kernel(dictionary['kernel_name'],
                                               include_grad=True)

        if dictionary['energy_kernel'] is not None:
            energy_kernel = str_to_kernel(dictionary['energy_kernel'])
        else:
            energy_kernel = None

        if dictionary['energy_force_kernel'] is not None:
            energy_force_kernel = \
                str_to_kernel(dictionary['energy_force_kernel'])
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
                                 ncpus=dictionary.get('ncpus') or dictionary.get('no_cpus'),
                                 maxiter=dictionary['maxiter'],
                                 opt_algorithm=dictionary['algo'],
                                 multihyps=dictionary.get('multihyps',False),
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
            new_gp.ky_mat_inv = np.array(dictionary.get('ky_mat_inv', None))
            new_gp.ky_mat = np.array(dictionary.get('ky_mat', None))
            new_gp.l_mat = np.array(dictionary.get('l_mat', None))
            new_gp.alpha = np.array(dictionary.get('alpha', None))


        return new_gp
    
    @staticmethod
    def from_file(filename):
        if '.json' in filename:
            with open(filename,'r') as f:
                return GaussianProcess.from_dict(json.loads(f.readline()))

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

        :param name: Output name
        :param format:
        :return:
        """

        if (len(self.training_data)>500):
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

        if (len(self.training_data)>500):
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
