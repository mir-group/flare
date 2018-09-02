#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""" Gaussian Process Regression model

Implementation is based on Algorithm 2.1 (pg. 19) of
"Gaussian Processes for Machine Learning" by Rasmussen and Williams

Simon Batzner
"""
import os

import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from dev.two_body import two_body
from dev.kern_help import get_envs, fc_conv
from otf import parse_qe_input, parse_qe_forces
from dev.MD_Parser import parse_qe_pwscf_md_output


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
        self.length_scale = None
        self.sigma_n = None
        self.sigma_f = None

        # quantities used in GPR algorithm
        self.k_mat = None
        self.l_mat = None
        self.alpha = None

        # training set
        self.training_data = np.empty(0,)
        self.training_labels = np.empty(0,)

    def kernel(self, x1, x2, d1, d2, sig, ls):
        """Evaluate specified kernel."""
        if self.kernel_type == 'two_body':
            return two_body(x1, x2, d1, d2, sig, ls)
        else:
            raise ValueError('{} is not a valid kernel'.format(self.kernel))

    def init_db(self, init_type='dir', **kwargs):
        """Initialize database from root directory containing training data.

        :param init_type: whether to init from files in dir or pass data
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
            self.training_data = kwargs['positions']
            self.training_labels = kwargs['forces']

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

        # nelder-mead optimization
        res = minimize(self.minus_like_hyp, x_0, method='nelder-mead',
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

    def minus_like_hyp(self, hyp):
        """Get minus likelihood as a function of hyperparameters

        :param hyp: hyperparameters to optimize
        :type hyp: list<float>
        :return: negative likelihood
        :rtype:
        """
        like = self.like_hyp(hyp)
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
        d_s = ['xrel', 'yrel', 'zrel']

        # initialize matrix
        size = len(self.training_data) * 3
        k_mat = np.zeros([size, size])

        # calculate elements
        for m_index in range(size):
            x_1 = self.training_data[int(math.floor(m_index / 3))]
            d_1 = d_s[m_index % 3]
            for n_index in range(m_index, size):
                x_2 = self.training_data[int(math.floor(n_index / 3))]
                d_2 = d_s[n_index % 3]

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
        d_s = ['xrel', 'yrel', 'zrel']
        size = len(self.training_data) * 3
        k_v = np.zeros([size, 1])

        for m in range(size):
            x_2 = self.training_data[int(math.floor(m / 3))]
            d_2 = d_s[m % 3]
            k_v[m] = self.kernel(x, x_2, d_1, d_2,
                                 self.sigma_f, self.length_scale)

        return k_v

    def set_alpha(self):
        """ Set weight vector alpha """
        ts1 = solve_triangular(self.l_mat, self.training_labels, lower=True)
        alpha = solve_triangular(self.l_mat.transpose(), ts1)

        self.alpha = alpha

if __name__ == "__main__":
    outfile = os.path.join(os.environ['ML_HOME'],
                           'data/Si_Supercell_MD/si.md.out')
    Si_MD_Parsed = parse_qe_pwscf_md_output(outfile)

    # set crystal structure
    dim = 3
    alat = 5.431
    unit_cell = [[0.0, alat / 2, alat / 2],
                 [alat / 2, 0.0, alat / 2],
                 [alat / 2, alat / 2, 0.0]]

    unit_pos = [['Si', [0, 0, 0]],
                ['Si', [alat / 4, alat / 4, alat / 4]]]

    brav_mat = np.array([[0.0, alat / 2, alat / 2],
                         [alat / 2, 0.0, alat / 2],
                         [alat / 2, alat / 2, 0.0]]) * dim

    brav_inv = np.linalg.inv(brav_mat)

    # bravais vectors
    vec1 = brav_mat[:, 0].reshape(3, 1)
    vec2 = brav_mat[:, 1].reshape(3, 1)
    vec3 = brav_mat[:, 2].reshape(3, 1)

    # build force field from single snapshot
    cutoff = 4.5
    positions = np.asarray(Si_MD_Parsed[1]['positions'])
    envs = get_envs(positions, brav_mat, brav_inv, vec1, vec2, vec3, cutoff)
    fcs = fc_conv(Si_MD_Parsed[2]['forces'])

    # build gp and train
    gp = GaussianProcess(kernel='two_body')
    gp.init_db(init_type='data', positions=positions, forces=fcs)
    gp.train()
