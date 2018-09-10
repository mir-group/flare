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

from env import ChemicalEnvironment, two_body, two_body_py, three_body,\
    three_body_py
from struc import Structure


class GaussianProcess:
    """ Gaussian Process Regression Model """

    def __init__(self, kernel: str):
        """Initialize GP parameters and training data.

        :param kernel: covariance / kernel function to be used
        :type kernel: str
        """

        # gp kernel and hyperparameters
        self.kernel_type = kernel
        self.length_scale = 1
        self.sigma_n = 1
        self.sigma_f = 0.01

        # quantities used in GPR algorithm
        self.k_mat = None
        self.l_mat = None
        self.alpha = None

        # training set
        self.training_data = []
        self.training_labels = []
        self.training_labels_np = np.empty(0,)

    def kernel(self, env1: ChemicalEnvironment, env2: ChemicalEnvironment,
               d1: int, d2: int, sig: float, length_scale: float):
        """Evaluate specified kernel.

        :param env1: first chemical environment to compare
        :type evn1: ChemicalEnvironment
        :param env2: second chemical environment to compare
        :type env2: ChemicalEnvironment
        :param d1:
        :type d1: int
        :param d2:
        :type d2: int
        :param sig: signal variance
        :type sig: float
        :param length_scale: length scale of kernel
        :type: length_scale: float
        """

        if self.kernel_type == 'two_body':
            return two_body(env1, env2, d1, d2, sig, length_scale)

        elif self.kernel_type == 'two_body_py':
            return two_body_py(env1, env2, d1, d2, sig, length_scale)

        elif self.kernel_type == 'three_body':
            return three_body(env1, env2, d1, d2, sig, length_scale)

        elif self.kernel_type == 'three_body_py':
            return three_body_py(env1, env2, d1, d2, sig, length_scale)

        else:
            raise ValueError('{} is not a valid kernel'.format(self.kernel))

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

        # optimize hyperparameters
        self.opt_hyper()

        # following: Algorithm 2.1 (pg. 19) of
        # "Gaussian Processes for Machine Learning" by Rasmussen and Williams
        self.set_kernel(sigma_f=self.sigma_f,
                        length_scale=self.length_scale,
                        sigma_n=self.sigma_n)

        # get alpha and likelihood
        self.set_alpha()

    def opt_hyper(self):
        """
        Optimize hyperparameters of GP by minimizing minus log likelihood. """
        # initial guess
        x_0 = np.array([self.sigma_f, self.length_scale, self.sigma_n])

        # nelder-mead optimization
        args = (self,)
        res = minimize(minus_like_hyp, x_0, args,
                       method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': False})

        self.sigma_f = res.x[0]
        self.length_scale = res.x[1]
        self.sigma_n = res.x[2]

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
        self_kern = self.kernel(x_t, x_t, d, d,
                                self.sigma_f, self.length_scale)
        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def like_hyp(self, hyp: List[float]) -> float:
        """ Get likelihood as a function of hyperparameters.

        :param hyp: hyperparameters to optimize
        :type hyp: list<float>
        :return: likelihood
        :rtype: float
        """

        # unpack hyperparameters
        sigma_f = hyp[0]
        length_scale = hyp[1]
        sigma_n = hyp[2]

        # calculate likelihood
        self.set_kernel(sigma_f=sigma_f,
                        length_scale=length_scale,
                        sigma_n=sigma_n)

        self.set_alpha()
        like = self.get_likelihood()

        return like

    def get_likelihood(self) -> float:
        """ Get log marginal likelihood.

        :return: likelihood
        :rtype: float
        """
        like = np.asscalar(-(1 / 2) *
                           np.matmul(self.training_labels_np.transpose(),
                                     self.alpha) -
                           np.sum(np.log(np.diagonal(self.l_mat))) -
                           np.log(2 * np.pi) * self.k_mat.shape[1] / 2)

        return like

    def set_kernel(self, sigma_f: float, length_scale: float, sigma_n: float):
        """ Compute 3Nx3N noiseless kernel matrix.

        :param sigma_f: signal variance
        :type sigma_f: float
        :param length_scale: length scale of the GP
        :type length_scale: float
        :param sigma_n: noise variance
        :type sigma_n: float

        """

        # initialize matrix
        size = len(self.training_data)*3
        k_mat = np.zeros([size, size])

        ds = [1, 2, 3]

        # calculate elements
        for m_index in range(size):
            x_1 = self.training_data[int(math.floor(m_index / 3))]
            d_1 = ds[m_index % 3]

            for n_index in range(m_index, size):
                x_2 = self.training_data[int(math.floor(n_index / 3))]
                d_2 = ds[n_index % 3]

                # calculate kernel
                cov = self.kernel(x_1, x_2, d_1, d_2, sigma_f, length_scale)
                k_mat[m_index, n_index] = cov
                k_mat[n_index, m_index] = cov

        # perform cholesky decomposition and store
        self.l_mat = np.linalg.cholesky(k_mat + sigma_n ** 2 * np.eye(size))
        self.k_mat = k_mat

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
            k_v[m_index] = self.kernel(x, x_2, d_1, d_2,
                                       self.sigma_f, self.length_scale)

        return k_v

    def set_alpha(self):
        """ Set weight vector alpha. """
        ts1 = solve_triangular(self.l_mat, self.training_labels_np, lower=True)
        self.alpha = solve_triangular(self.l_mat.transpose(), ts1)


def minus_like_hyp(hyp: List[float], gp: GaussianProcess,
                   verbose: bool = True) -> float:
    """Get minus likelihood as a function of hyperparameters.

    :param hyp: hyperparmeters to optimize: signal var, length scale, noise var
    :type hyp: [float, float, float]
    :param gp: gp object
    :type gp: GaussianProcess
    :param verbose: whether to print hyperparameters and likelihood
    :type verbose: bool
    :return: negative likelihood
    :rtype: float
    """
    like = gp.like_hyp(hyp)
    minus_like = -like

    if verbose:
        print(hyp)
        print(like)

    return minus_like
