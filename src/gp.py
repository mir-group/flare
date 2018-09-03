#!/usr/bin/env python3

"""" Gaussian Process Regression model

Implementation is based on Algorithm 2.1 (pg. 19) of
"Gaussian Processes for Machine Learning" by Rasmussen and Williams

Simon Batzner
"""
import os
import math
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

# from dev.two_body import two_body
# from dev.kern_help import get_envs
from env import ChemicalEnvironment, two_body, two_body_py
from otf import parse_qe_input, parse_qe_forces, Structure


def minus_like_hyp(hyp, gp):
    """Get minus likelihood as a function of hyperparameters

    """
    like = gp.like_hyp(hyp)
    minus_like = -like

    return minus_like


class GaussianProcess:
    """ Gaussian Process Regression Model """

    def __init__(self, kernel):
        """Initialize GP parameters and training data.

        :param kernel: covariance/ kernel function used
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

    def kernel(self, env1, env2, d1, d2, sig, ls):
        """Evaluate specified kernel."""

        if self.kernel_type == 'two_body':
            return two_body(env1, env2, d1, d2, sig, ls)

        elif self.kernel_type == 'two_body_py':
            return two_body_py(env1, env2, d1, d2, sig, ls)

        else:
            raise ValueError('{} is not a valid kernel'.format(self.kernel))

    def update_db(self, struc, forces):
        """Given structure and forces, add to training set.

        :param struc: [description]
        :type struc: [type]
        :param forces: [description]
        :type forces: [type]
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
    def force_list_to_np(forces):
        forces_np = []
        for m in range(len(forces)):
            for n in range(3):
                forces_np.append(forces[m][n])
        forces_np = np.array(forces_np)
        return forces_np

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
        args = (self)
        res = minimize(minus_like_hyp, x_0, args,
                       method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': False})

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
        k_v = self.get_kernel_vector(x_t, d)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        self_kern = self.kernel(x_t, x_t, d, d,
                                self.sigma_f, self.length_scale)
        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

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
            np.matmul(self.training_labels_np.transpose(), self.alpha) - \
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

    def get_kernel_vector(self, x, d_1):
        """ Compute kernel vector

        :param x: data point to compare against kernel matrix
        :type x:
        :param d_1:
        :type d_1:
        :return: kernel vector
        :rtype:
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
        """ Set weight vector alpha """
        ts1 = solve_triangular(self.l_mat, self.training_labels_np, lower=True)
        self.alpha = solve_triangular(self.l_mat.transpose(), ts1)

# # append data from directory
# # likely won't be used in this paper, so not necessary now

# def get_outfiles(root_dir, out=True):
#     """Find all files matching *.out or *.in.

#     :param root_dir: dir to walk from
#     :type root_dir: str
#     :param out: whether to look for .out or .in files
#     :type out: bool
#     :return: files in root_dir ending with .out
#     :rtype: list<str>
#     """
#     matching = []

#     for root, _, filenames in os.walk(root_dir):
#         for filename in filenames:

#             if out:
#                 if filename.endswith('.out'):
#                     matching.append(os.path.join(root, filename))

#             else:
#                 if filename.endswith('.in'):
#                     matching.append(os.path.join(root, filename))

#     return matching

# def update_db_from_directory(self):
#     pos, _, _, frcs = [], [], [], []

#     for file in get_outfiles(root_dir=kwargs['root_dir'], out=False):
#         pos = parse_qe_input(file)[0]

#     for file in get_outfiles(root_dir=kwargs['root_dir'], out=True):
#         frcs = parse_qe_forces(file)

#     self.training_data = np.asarray(get_envs(
#                                         pos=pos,
#                                         brav_mat=kwargs['brav_mat'],
#                                         brav_inv=kwargs['brav_inv'],
#                                         vec1=kwargs['vec1'],
#                                         vec2=kwargs['vec2'],
#                                         vec3=kwargs['vec3'],
#                                         cutoff=kwargs['cutoff']))
#     self.training_labels = np.asarray(frcs)

if __name__ == "__main__":

    # create a random test structure
    def get_random_structure(cell, unique_species, cutoff, noa):
        positions = []
        forces = []
        species = []
        for n in range(noa):
            positions.append(np.random.uniform(-1, 1, 3))
            forces.append(np.random.uniform(-1, 1, 3))
            species.append(unique_species[np.random.randint(0, 2)])

        test_structure = Structure(cell, species, positions, cutoff)

        return test_structure, forces

    cell = np.eye(3)
    unique_species = ['B', 'A']
    cutoff = 0.8
    noa = 10
    test_structure, forces = \
        get_random_structure(cell, unique_species, cutoff, noa)

    # create test point
    test_structure_2, _ = get_random_structure(cell, unique_species, cutoff,
                                               noa)
    test_pt = ChemicalEnvironment(test_structure_2, 0)

    # test update_db
    gaussian = GaussianProcess(kernel='two_body')
    gaussian.update_db(test_structure, forces)
    assert(len(gaussian.training_data) == noa)
    assert(len(gaussian.training_data) == len(gaussian.training_data))
    assert(len(gaussian.training_labels_np) == len(gaussian.training_data * 3))

    # test get_kernel
    gaussian.set_kernel(sigma_f=1, length_scale=1, sigma_n=0.1)
    db_pts = 3 * len(gaussian.training_data)
    assert(gaussian.k_mat.shape == (db_pts, db_pts))

    # test get_alpha
    gaussian.set_alpha()
    assert(gaussian.alpha.shape == (db_pts,))

    # test get_kernel_vector
    assert(gaussian.get_kernel_vector(test_pt, 1).shape ==
           (db_pts,))

    # test get_likelihood and like_hyp
    like = gaussian.get_likelihood()

    def like_performance(its, kernel_type, hyp, gp):
        # set kernel type
        gp.kernel_type = kernel_type

        # warm up jit
        like_hyp = gp.like_hyp(hyp)

        # test performance
        time0 = time.time()
        for n in range(its):
            like_hyp = gp.like_hyp(hyp)
        time1 = time.time()
        like_time = (time1 - time0) / its

        return like_time, like_hyp

    its = 10
    jit_kern = 'two_body'
    py_kern = 'two_body_py'
    hyp = np.array([1, 1, 0.1])
    jit_time, jit_like = like_performance(its, jit_kern, hyp, gaussian)
    py_time, py_like = like_performance(its, py_kern, hyp, gaussian)
    assert(py_time / jit_time > 1)
    assert(round(py_like, 8) == round(jit_like, 8))
    assert(round(like, 8) == round(jit_like, 8))

    # test train
    gaussian.kernel_type = 'two_body'
    gaussian.train()

    # test predict
    print(gaussian.predict(test_pt, 1))
