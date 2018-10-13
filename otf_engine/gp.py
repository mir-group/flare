#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" Gaussian Process Regression model

Implementation is based on Algorithm 2.1 (pg. 19) of
"Gaussian Processes for Machine Learning" by Rasmussen and Williams

Simon Batzner, Jon Vandermause
"""

import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from typing import List

from env import ChemicalEnvironment
from kernels import two_body, three_body, n_body_sc, n_body_sc_grad
from kernels import get_likelihood_and_gradients, get_K_L_alpha
from kernels import get_likelihood
from struc import Structure


class GaussianProcess:
    """ Gaussian Process Regression Model """

    def __init__(self, kernel: str, bodies: int):
        """Initialize GP parameters and training data.

        :param kernel: covariance / kernel function to be used
        :type kernel: str
        """

        # gp kernel and hyperparameters
        self.bodies = bodies

        if kernel == 'n_body_sc':
            self.kernel = n_body_sc
            self.kernel_grad = n_body_sc_grad
            self.hyps = np.array([1, 1, 1])
        else:
            raise ValueError('not a valid kernel')

        # quantities used in GPR algorithm
        self.k_mat = None
        self.l_mat = None
        self.alpha = None

        # training set
        self.training_data = []
        self.training_labels = []
        self.training_labels_np = np.empty(0,)

    def update_db(self, struc: Structure, forces: list):
        """Given structure and forces, add to training set.

        :param struc: structure to add to db
        :type struc: Structure
        :param forces: list of corresponding forces to add to db
        :type forces: list<float>
        """

        noa = len(struc.positions)

        for atom in range(noa):
            env_curr = ChemicalEnvironment(struc, atom)
            forces_curr = forces[atom]

            self.training_data.append(env_curr)
            self.training_labels.append(forces_curr)

        # create numpy array of training labels
        self.training_labels_np = self.force_list_to_np(self.training_labels)

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

    def train(self):
        """ Train Gaussian Process model on training data. """

        x_0 = self.hyps
        args = (self.training_data, self.training_labels_np,
                self.kernel_grad, self.bodies)

        res = minimize(get_likelihood_and_gradients, x_0, args,
                       method='BFGS', jac=True,
                       options={'disp': False, 'gtol': 1e-2, 'maxiter': 1000})

        # args = (self.training_data, self.training_labels_np,
        #         self.kernel, self.bodies)

        # res = minimize(get_likelihood, x_0, args,
        #                method='nelder-mead',
        #                options={'disp': False, 'xtol': 1e-8})

        self.hyps = res.x
        self.k_mat, self.l_mat, self.alpha = \
            get_K_L_alpha(self.hyps, self.training_data,
                          self.training_labels_np, self.kernel, self.bodies)

    def predict(self, x_t: ChemicalEnvironment, d: int) -> [float, float]:
        """ Make GP prediction with SE kernel.

        :param x_t: data point to predict on
        :type x_t: ChemicalEnvironment
        :param d:
        :type d: int
        :return: predictive mean and predictive variance
        :rtype: [float, float]
        """

        # get kernel vector
        k_v = self.get_kernel_vector(x_t, d)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        self_kern = self.kernel(x_t, x_t, self.bodies, d, d, self.hyps)
        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def get_kernel_vector(self, x: ChemicalEnvironment, d_1: int)-> np.ndarray:
        """ Compute kernel vector.

        :param x: data point to compare against kernel matrix
        :type x: ChemicalEnvironment
        :param d_1:
        :type d_1: int
        :return: kernel vector
        :rtype: np.ndarray
        """

        ds = [1, 2, 3]
        size = len(self.training_data)*3
        k_v = np.zeros(size,)

        for m_index in range(size):
            x_2 = self.training_data[int(math.floor(m_index/3))]
            d_2 = ds[m_index % 3]
            k_v[m_index] = self.kernel(x, x_2, self.bodies, d_1, d_2,
                                       self.hyps)

        return k_v
