#!/usr/bin/env python3

"""" Gaussian Process Regression model

Implementation is based on Algorithm 2.1 (pg. 19) of
"Gaussian Processes for Machine Learning" by Rasmussen and Williams

Simon Batzner
"""
import os

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from dev.two_body import two_body
from dev.kern_help import get_envs
from env import ChemicalEnvironment, two_body, two_body_py
from otf import parse_qe_input, parse_qe_forces, Structure


def get_outfiles(root_dir, out=True):
    """Find all files matching *.out or *.in.

    :param root_dir: dir to walk from
    :type root_dir: str
    :param out: whether to look for .out or .in files
    :type out: bool
    :return: files in root_dir ending with .out
    :rtype: list<str>
    """
    matching = []

    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:

            if out:
                if filename.endswith('.out'):
                    matching.append(os.path.join(root, filename))

            else:
                if filename.endswith('.in'):
                    matching.append(os.path.join(root, filename))

    return matching


class GaussianProcess:
    """ Gaussian Process Regression Model """

    def __init__(self, kernel):
        """Initialize GP parameters and training data.

        :param kernel: covariance/ kernel function used
        :type kernel: str
        """

        # predictive mean and variance
        self.pred_mean = None
        self.pred_var = None

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
        self.training_data = np.empty(0,)
        self.training_labels = np.empty(0,)

    def kernel(self, env1, env2, d1, d2, sig, ls):
        """Evaluate specified kernel."""

        if self.kernel_type == 'two_body':
            return two_body(env1, env2, d1, d2, sig, ls)

        elif self.kernel_type == 'two_body_py':
            return two_body_py(env1, env2, d1, d2, sig, ls)

        else:
            raise ValueError('{} is not a valid kernel'.format(self.kernel))

    def init_db(self, init_type='dir', **kwargs):
        """Initialize database from dir or directly from chemical envs

        :param init_type: whether to init from files in dir or pass envs
        :type init_type: str
        """

        if init_type == 'dir':
            pos, _, _, frcs = [], [], [], []

            for file in get_outfiles(root_dir=kwargs['root_dir'], out=False):
                pos = parse_qe_input(file)[0]

            for file in get_outfiles(root_dir=kwargs['root_dir'], out=True):
                frcs = parse_qe_forces(file)

            self.training_data = np.asarray(get_envs(
                                             pos=pos,
                                             brav_mat=kwargs['brav_mat'],
                                             brav_inv=kwargs['brav_inv'],
                                             vec1=kwargs['vec1'],
                                             vec2=kwargs['vec2'],
                                             vec3=kwargs['vec3'],
                                             cutoff=kwargs['cutoff']))
            self.training_labels = np.asarray(frcs)

        elif init_type == 'data':
            self.training_data = kwargs['envs']
            self.training_labels = np.asarray(kwargs['forces'])

    def train(self):
        """ Train Gaussian Process model on training data """

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
        Optimize hyperparameters of GP by minimizing minus log likelihood
        """
        # initial guess
        x_0 = np.array([self.sigma_f, self.length_scale, self.sigma_n])
        # x_0 = x_0.reshape(3, 1)

        # nelder-mead optimization
        res = minimize(fun=GaussianProcess.minus_like_hyp,
                       x0=x_0,
                       args=(self,),
                       method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': True})

        self.sigma_f = res.x[0]
        self.length_scale = res.x[1]
        self.sigma_n = res.x[2]

    def predict(self, x_t, d):
        """Make GP prediction with SE kernel

        :param x_t: data point to predict on
        :type x_t:
        :param d: kernel parameter
        :type d:
        """

        # get kernel vector
        k_v = self.get_kernel_vector(x=x_t, d_1=1)

        # get predictive mean
        self.pred_mean = np.matmul(k_v.transpose(), self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        self_kern = self.kernel(x_t, x_t, d, d,
                                self.sigma_f, self.length_scale)
        self.pred_var = self_kern - np.matmul(v_vec.transpose(), v_vec)

    @staticmethod
    def minus_like_hyp(hyp, gp):
        """Get minus likelihood as a function of hyperparameters

        """
        like = gp.like_hyp(hyp)
        minus_like = -like

        return minus_like

    def like_hyp(self, hyp):
        """ Get likelihood as a function of hyperparameters

        :param hyp: hyperparameters to optimize
        :type hyp: list<float>
        :return: likelihood
        :rtype:
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

    def get_likelihood(self):
        """
        Get log marginal likelihood

        :return: likelihood
        :rtype:
        """
        like = -(1 / 2) * \
            np.matmul(self.training_labels.transpose(), self.alpha) - \
            np.sum(np.log(np.diagonal(self.l_mat))) - np.log(2 * np.pi) * \
            self.k_mat.shape[1] / 2

        return like

    def set_kernel(self, sigma_f, length_scale, sigma_n):
        """ Compute 3Nx3N noiseless kernel matrix

        :param sigma_f: signal variance
        :type sigma_f: float
        :param length_scale: length scale of the GP
        :type length_scale: float
        :param sigma_n: noise variance
        :type sigma_n: float

        """

        # hard-coded for testing purposes
        d_1 = 1
        d_2 = 1

        # initialize matrix
        size = len(self.training_data)
        k_mat = np.zeros([size, size])

        # calculate elements
        for m_index in range(size):
            x_1 = self.training_data[m_index]

            for n_index in range(m_index, size):
                x_2 = self.training_data[n_index]

                # calculate kernel
                cov = self.kernel(x_1, x_2, d_1, d_2, sigma_f, length_scale)
                k_mat[m_index, n_index] = cov
                k_mat[n_index, m_index] = cov

        # perform cholesky decomposition and store
        self.l_mat = np.linalg.cholesky(k_mat + sigma_n ** 2 * np.eye(size))
        self.k_mat = k_mat

    def get_kernel_vector(self, x, d_1):
        """ Compute kernel vector

        :param x: data point to compare against kernel matrix
        :type x:
        :param d_1:
        :type d_1:
        :return: kernel vector
        :rtype:
        """
        size = len(self.training_data)
        k_v = np.zeros([size, 1])

        # hard-coded for testing purposes
        d_2 = 1

        for m_index in range(size):
            x_2 = self.training_data[m_index]
            k_v[m_index] = self.kernel(x, x_2, d_1, d_2,
                                       self.sigma_f, self.length_scale)

        return k_v

    def set_alpha(self):
        """ Set weight vector alpha """
        ts1 = solve_triangular(self.l_mat, self.training_labels, lower=True)
        self.alpha = solve_triangular(self.l_mat.transpose(), ts1)

if __name__ == "__main__":

    # set up a few test environments
    n_train = 10
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001
    train_envs = []
    species = ['B', 'A']
    atom = 0
    train_forces = []

    for i in range(n_train):

        # generate random positions and build chem env
        positions = [np.array([np.random.rand() * 0.5,
                               np.random.rand() * 0.5,
                               np.random.rand() * 0.5]),
                     np.array([np.random.rand() * 0.5,
                               np.random.rand() * 0.5,
                               np.random.rand() * 0.5])]

        test_structure = Structure(cell, species, positions, cutoff)
        train_envs.append(ChemicalEnvironment(test_structure, atom))

        # generate random force vectors
        train_forces.append(np.array([np.random.rand(),
                            np.random.rand(),
                            np.random.rand()]))

    # build gp and train
    gaussian = GaussianProcess(kernel='two_body_py')
    gaussian.init_db(init_type='data', envs=train_envs, forces=train_forces)
    gaussian.train()
