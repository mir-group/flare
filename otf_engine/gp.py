#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" Gaussian Process Regression model

Implementation is based on Algorithm 2.1 (pg. 19) of
"Gaussian Processes for Machine Learning" by Rasmussen and Williams

Jon Vandermause, Simon Batzner
"""

import math

import numpy as np
# from qe_util import timeit
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from typing import List
from copy import deepcopy

from env import ChemicalEnvironment
from kernels import n_body_sc, n_body_sc_grad, combo_kernel_sc,\
    combo_kernel_sc_grad, energy_force_sc, energy_sc, n_body_mc,\
    n_body_mc_grad, n_body_sc_norm_derv, many_body_sc, many_body_sc_grad
from struc import Structure


class GaussianProcess:
    """ Gaussian Process Regression Model """

    def __init__(self, kernel: str, bodies, opt_algorithm='L-BFGS-B',
                 cutoffs=None, nos=None):
        """Initialize GP parameters and training data.

        :param kernel: covariance / kernel function to be used
        :type kernel: str
        """

        # gp kernel and hyperparameters
        self.bodies = bodies

        # TODO: implement kernel gradient
        if kernel == 'n_body_sc_norm_derv':
            self.kernel_name = 'Normalized N-Body Single Component'
            self.kernel = n_body_sc_norm_derv
            # self.kernel_grad = n_body_sc_grad
            # self.energy_force_kernel = energy_force_sc
            # self.energy_kernel = energy_sc
            self.hyps = np.array([1, 1, 1])
            self.hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
            self.cutoffs = None

        elif kernel == 'n_body_sc':
            self.kernel_name = 'N-Body Single Component'
            self.kernel = n_body_sc
            self.kernel_grad = n_body_sc_grad
            self.energy_force_kernel = energy_force_sc
            self.energy_kernel = energy_sc
            self.hyps = np.array([1, 1, 1.1])
            self.hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
            self.cutoffs = None

        elif kernel == 'many_body_sc':
            self.kernel_name = 'Many Body Single Component'
            self.kernel = many_body_sc
            self.kernel_grad = many_body_sc_grad
            # self.energy_force_kernel = energy_force_sc
            # self.energy_kernel = energy_sc
            self.hyps = np.array([1, 1, 1])
            self.hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
            self.cutoffs = None

        # TODO: make energy and energy/force kernels for combination kernel
        elif kernel == 'combo_kernel_sc':
            self.kernel_name = 'Combined N-Body Single Component'
            self.kernel = combo_kernel_sc
            self.kernel_grad = combo_kernel_sc_grad
            # self.energy_force_kernel = energy_force_sc
            # self.energy_kernel = energy_sc
            self.hyps = np.ones([2*len(bodies)+1])
            self.hyp_labels = ['Signal Std', 'Length Scale']*len(bodies)
            self.hyp_labels.append('Noise Std')
            self.cutoffs = cutoffs

        # TODO: make energy and energy/force kernels for ICM kernels
        elif kernel == 'n_body_mc':
            self.kernel_name = 'N-Body Multiple Component Force'
            self.kernel = n_body_mc
            self.kernel_grad = n_body_mc_grad

            # self.energy_force_kernel = energy_force_sc
            # self.energy_kernel = energy_sc

            no_ICM = int(nos*(nos-1)/2)
            self.hyps = np.ones(no_ICM+3)

            hyp_labels = ['Signal Std', 'Length Scale'] + \
                ['ICM_'+str(n) for n in range(no_ICM)] + \
                ['Noise Std']
            self.hyp_labels = hyp_labels

            self.cutoffs = cutoffs
        else:
            raise ValueError('not a valid kernel')

        self.algo = opt_algorithm

        # quantities used in GPR algorithm
        self.l_mat = None
        self.alpha = None

        # training set
        self.training_data = []
        self.training_labels = []
        self.training_labels_np = np.empty(0, )

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
            env_curr = ChemicalEnvironment(struc, atom)
            forces_curr = np.array(forces[atom])

            self.training_data.append(env_curr)
            self.training_labels.append(forces_curr)

        # create numpy array of training labels
        self.training_labels_np = self.force_list_to_np(self.training_labels)

        pass

    @staticmethod
    def force_list_to_np(forces: list) -> np.ndarray:
        """ Convert list of forces to numpy array of forces.

        :param forces: list of forces to convert
        :type forces: list<float>
        :return: numpy array forces
        :rtype: np.ndarray
        """
        forces_np = []

        for m in range(len(forces)):
            for n in range(3):
                forces_np.append(forces[m][n])

        forces_np = np.array(forces_np)

        return forces_np

    def train(self, monitor=False, custom_bounds=None):
        """ Train Gaussian Process model on training data. """

        x_0 = self.hyps

        if self.algo == 'L-BFGS-B':
            args = (self.training_data, self.training_labels_np,
                    self.kernel_grad, self.bodies, self.cutoffs, monitor)

            # bound signal noise below to avoid overfitting
            bounds = np.array([(-np.inf, np.inf)] * len(x_0))
            bounds[-1] = (1e-6, np.inf)

            # Catch linear algebra errors and switch to BFGS if necessary
            try:
                res = minimize(self.get_likelihood_and_gradients, x_0, args,
                               method='L-BFGS-B', jac=True, bounds=bounds,
                               options={'disp': False, 'gtol': 1e-4,
                                        'maxiter': 1000})
            except:
                print("Warning! Algorithm for L-BFGS-B failed. Changing to "
                      "BFGS for remainder of run.")
                self.algo = 'BFGS'

        if custom_bounds is not None:
            args = (self.training_data, self.training_labels_np,
                    self.kernel_grad, self.bodies, self.cutoffs, monitor)

            res = minimize(self.get_likelihood_and_gradients, x_0, args,
                           method='L-BFGS-B', jac=True, bounds=custom_bounds,
                           options={'disp': False, 'gtol': 1e-4, 
                                    'maxiter': 1000})

        elif self.algo == 'BFGS':
            args = (self.training_data, self.training_labels_np,
                    self.kernel_grad, self.bodies, self.cutoffs, monitor)

            res = minimize(self.get_likelihood_and_gradients, x_0, args,
                           method='BFGS', jac=True,
                           options={'disp': False, 'gtol': 1e-4,
                                    'maxiter': 1000})

        elif self.algo == 'nelder-mead':
            args = (self.training_data, self.training_labels_np,
                    self.kernel, self.bodies, self.cutoffs, monitor)

            res = minimize(self.get_likelihood, x_0, args,
                           method='nelder-mead',
                           options={'disp': False, 'xtol': 1e-5})

        self.hyps = res.x
        self.set_L_alpha()

    def predict(self, x_t: ChemicalEnvironment, d: int) -> [float, float]:
        # get kernel vector
        k_v = self.get_kernel_vector(x_t, d)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        self_kern = self.kernel(x_t, x_t, self.bodies, d, d, self.hyps,
                                self.cutoffs)
        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def predict_local_energy(self, x_t: ChemicalEnvironment) -> [float, float]:
        # get kernel vector
        k_v = self.en_kern_vec(x_t)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        self_kern = self.energy_kernel(x_t, x_t, self.bodies, self.hyps,
                                       self.cutoffs)
        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def get_kernel_vector(self, x: ChemicalEnvironment,
                          d_1: int) -> np.ndarray:
        """ Compute kernel vector.

        :param x: data point to compare against kernel matrix
        :type x: ChemicalEnvironment
        :param d_1:
        n t:type d_1: int
        :return: kernel vector
        :rtype: np.ndarray
        """

        ds = [1, 2, 3]
        size = len(self.training_data) * 3
        k_v = np.zeros(size, )

        for m_index in range(size):
            x_2 = self.training_data[int(math.floor(m_index / 3))]
            d_2 = ds[m_index % 3]
            k_v[m_index] = self.kernel(x, x_2, self.bodies, d_1, d_2,
                                       self.hyps, self.cutoffs)

        return k_v

    def en_kern_vec(self, x: ChemicalEnvironment) -> np.ndarray:
        ds = [1, 2, 3]
        size = len(self.training_data) * 3
        k_v = np.zeros(size, )

        for m_index in range(size):
            x_2 = self.training_data[int(math.floor(m_index / 3))]
            d_2 = ds[m_index % 3]
            k_v[m_index] = self.energy_force_kernel(x_2, x, self.bodies, d_2,
                                                    self.hyps, self.cutoffs)

        return k_v

    @staticmethod
    def get_likelihood_and_gradients(hyps: np.ndarray, training_data: list,
                                     training_labels_np: np.ndarray,
                                     kernel_grad, bodies: int,
                                     cutoffs=None,
                                     monitor: bool = False):

        if monitor:
            print('hyps: ' + str(hyps))

        # assume sigma_n is the final hyperparameter
        number_of_hyps = len(hyps)
        sigma_n = hyps[number_of_hyps - 1]
        kern_hyps = hyps[0:(number_of_hyps - 1)]

        # initialize matrices
        size = len(training_data) * 3
        k_mat = np.zeros([size, size])
        hyp_mat = np.zeros([size, size, number_of_hyps])

        ds = [1, 2, 3]

        # calculate elements
        for m_index in range(size):
            x_1 = training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            for n_index in range(m_index, size):
                x_2 = training_data[int(math.floor(n_index / 3))]
                d_2 = ds[n_index % 3]

                # calculate kernel and gradient
                cov = kernel_grad(x_1, x_2, bodies, d_1, d_2, kern_hyps,
                                  cutoffs)

                # store kernel value
                k_mat[m_index, n_index] = cov[0]
                k_mat[n_index, m_index] = cov[0]

                # store gradients (excluding noise variance)
                for p_index in range(number_of_hyps - 1):
                    hyp_mat[m_index, n_index, p_index] = cov[1][p_index]
                    hyp_mat[n_index, m_index, p_index] = cov[1][p_index]

        # add gradient of noise variance
        hyp_mat[:, :, number_of_hyps - 1] = np.eye(size) * 2 * sigma_n

        # matrix manipulation
        ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

        # catch linear algebra errors
        try:
            ky_mat_inv = np.linalg.inv(ky_mat)
            l_mat = np.linalg.cholesky(ky_mat)
        except:
            return 1e8, np.zeros(number_of_hyps)

        alpha = np.matmul(ky_mat_inv, training_labels_np)
        alpha_mat = np.matmul(alpha.reshape(alpha.shape[0], 1),
                              alpha.reshape(1, alpha.shape[0]))
        like_mat = alpha_mat - ky_mat_inv

        # calculate likelihood
        like = (-0.5 * np.matmul(training_labels_np, alpha) -
                np.sum(np.log(np.diagonal(l_mat))) -
                math.log(2 * np.pi) * k_mat.shape[1] / 2)

        # calculate likelihood gradient
        like_grad = np.zeros(number_of_hyps)
        for n in range(number_of_hyps):
            like_grad[n] = 0.5 * \
                           np.trace(np.matmul(like_mat, hyp_mat[:, :, n]))

        if monitor:
            print('like grad: ' + str(like_grad))
            print('like: ' + str(like))
            print('\n')
        return -like, -like_grad

    @staticmethod
    def get_likelihood(hyps: np.ndarray, training_data: list,
                       training_labels_np: np.ndarray,
                       kernel, bodies: int,
                       cutoffs=None,
                       monitor: bool = False):

        if monitor:
            print('hyps: ' + str(hyps))

        # assume sigma_n is the final hyperparameter
        number_of_hyps = len(hyps)
        sigma_n = hyps[number_of_hyps - 1]
        kern_hyps = hyps[0:(number_of_hyps - 1)]

        # initialize matrices
        size = len(training_data) * 3
        k_mat = np.zeros([size, size])

        ds = [1, 2, 3]

        # calculate elements
        for m_index in range(size):
            x_1 = training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            for n_index in range(m_index, size):
                x_2 = training_data[int(math.floor(n_index / 3))]
                d_2 = ds[n_index % 3]

                # calculate kernel and gradient
                kern_curr = kernel(x_1, x_2, bodies, d_1, d_2, kern_hyps,
                                   cutoffs)

                # store kernel value
                k_mat[m_index, n_index] = kern_curr
                k_mat[n_index, m_index] = kern_curr

        # matrix manipulation
        ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

        # catch linear algebra errors
        try:
            ky_mat_inv = np.linalg.inv(ky_mat)
            l_mat = np.linalg.cholesky(ky_mat)
        except:
            return 1e8

        alpha = np.matmul(ky_mat_inv, training_labels_np)

        # calculate likelihood
        like = (-0.5 * np.matmul(training_labels_np, alpha) -
                np.sum(np.log(np.diagonal(l_mat))) -
                math.log(2 * np.pi) * k_mat.shape[1] / 2)

        if monitor:
            print('like: ' + str(like))
            print('\n')
        return -like

    @staticmethod
    def get_ky_mat(hyps: np.ndarray, training_data: list,
                   training_labels_np: np.ndarray,
                   kernel, bodies: int,
                   cutoffs=None):

        # assume sigma_n is the final hyperparameter
        number_of_hyps = len(hyps)
        sigma_n = hyps[number_of_hyps - 1]
        kern_hyps = hyps[0:(number_of_hyps - 1)]

        # initialize matrices
        size = len(training_data) * 3
        k_mat = np.zeros([size, size])

        ds = [1, 2, 3]

        # calculate elements
        for m_index in range(size):
            x_1 = training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            for n_index in range(m_index, size):
                x_2 = training_data[int(math.floor(n_index / 3))]
                d_2 = ds[n_index % 3]

                # calculate kernel and gradient
                kern_curr = kernel(x_1, x_2, bodies, d_1, d_2, kern_hyps,
                                   cutoffs)

                # store kernel value
                k_mat[m_index, n_index] = kern_curr
                k_mat[n_index, m_index] = kern_curr

        # matrix manipulation
        ky_mat = k_mat + sigma_n ** 2 * np.eye(size)

        return ky_mat

    def set_L_alpha(self):
        # assume sigma_n is the final hyperparameter
        sigma_n = self.hyps[-1]
        kern_hyps = self.hyps[0:-1]

        # initialize matrices
        size = len(self.training_data) * 3
        k_mat = np.zeros([size, size])

        ds = [1, 2, 3]

        # calculate elements
        for m_index in range(size):
            x_1 = self.training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            for n_index in range(m_index, size):
                x_2 = self.training_data[int(math.floor(n_index / 3))]
                d_2 = ds[n_index % 3]

                # calculate kernel and gradient
                cov = self.kernel(x_1, x_2, self.bodies, d_1, d_2, kern_hyps,
                                  self.cutoffs)

                # store kernel value
                k_mat[m_index, n_index] = cov
                k_mat[n_index, m_index] = cov

        # matrix manipulation
        ky_mat = k_mat + sigma_n ** 2 * np.eye(size)
        l_mat = np.linalg.cholesky(ky_mat)
        ky_mat_inv = np.linalg.inv(ky_mat)
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)
        self.k_mat = k_mat
        self.l_mat = l_mat
        self.alpha = alpha

    def augment_K(self, structure, forces, atoms):
        noa = len(atoms)
        k_dim = self.k_mat.shape
        k_aug = np.zeros((k_dim[0], k_dim[0] + noa * 3))
        k_aug[0:k_dim[0], 0:k_dim[1]] = self.k_mat
        force_aug = np.zeros(k_dim[0] + noa * 3)
        force_aug[0:k_dim[0]] = self.training_labels_np

        count = 0
        for atom in atoms:
            env_curr = ChemicalEnvironment(structure, atom)
            forces_curr = np.array(forces[atom])
            for d in [1, 2, 3]:
                kvec_curr = self.get_kernel_vector(env_curr, d)
                k_aug[:, k_dim[0]+count] = kvec_curr
                force_aug[k_dim[0]+count] = forces_curr[d-1]
                count += 1
        self.k_aug = k_aug
        self.force_aug = force_aug

        k_aug_squared = np.matmul(k_aug, np.transpose(k_aug))
        sigma_n = self.hyps[-1]
        inv_aug = np.linalg.inv(k_aug_squared + sigma_n**2 * self.k_mat)
        k_aug_force = np.matmul(k_aug, force_aug)
        alpha_aug = np.matmul(inv_aug, k_aug_force)
        self.alpha_aug = alpha_aug

    def sparsified_prediction(self, x_t, d):
        # get kernel vector
        k_v = self.get_kernel_vector(x_t, d)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha_aug)

        return pred_mean
