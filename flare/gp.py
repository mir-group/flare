import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from typing import List, Callable
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.gp_algebra import get_ky_mat, get_ky_and_hyp, \
    get_like_from_ky_mat, get_like_grad_from_mats, get_neg_likelihood, \
    get_neg_like_grad, get_ky_and_hyp_par


class GaussianProcess:
    """ Gaussian Process Regression Model.

    Implementation is based on Algorithm 2.1 (pg. 19) of
    "Gaussian Processes for Machine Learning" by Rasmussen and Williams"""

    def __init__(self, kernel: Callable,
                 kernel_grad: Callable, hyps: np.ndarray,
                 cutoffs: np.ndarray,
                 hyp_labels: List = None,
                 energy_force_kernel: Callable = None,
                 energy_kernel: Callable = None,
                 opt_algorithm: str = 'L-BFGS-B',
                 maxiter=10, par=False,
                 output=None):
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
        self.l_mat = None
        self.alpha = None
        self.training_data = []
        self.training_labels = []
        self.training_labels_np = np.empty(0, )
        self.maxiter = maxiter
        self.likelihood = None
        self.likelihood_gradient = None
        self.par = par
        self.output = output

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

            self.training_data.append(env_curr)
            self.training_labels.append(forces_curr)

        # create numpy array of training labels
        self.training_labels_np = self.force_list_to_np(self.training_labels)

        self.set_L_alpha()

    def add_one_env(self, env: AtomicEnvironment,
                    force: np.array, train: bool = False, **kwargs):
        """
        Tool to add a single environment / force pair into the training set
        :param force:
        :param env:
        :param force: (x,y,z) component associated with environment
        :param train:
        :return:
        """
        self.training_data.append(env)
        self.training_labels.append(force)
        self.training_labels_np = self.force_list_to_np(self.training_labels)

        if train:
            self.train(**kwargs)

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

    def train(self, output=None, custom_bounds=None,
              grad_tol: float = 1e-4,
              x_tol: float = 1e-5,
              line_steps: int = 20):
        """
        Train Gaussian Process model on training data.
        Tunes the hyperparameters to maximize the Bayesian likelihood,
        then computes L and alpha (related to the covariance matrix of the
        training set).
        """

        x_0 = self.hyps

        args = (self.training_data, self.training_labels_np,
                self.kernel_grad, self.cutoffs, output,
                self.par)

        if self.algo == 'L-BFGS-B':

            # bound signal noise below to avoid overfitting
            bounds = np.array([(1e-6, np.inf)] * len(x_0))
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

        self.hyps = res.x
        self.set_L_alpha()
        self.likelihood = -res.fun
        self.likelihood_gradient = -res.jac

    def predict(self, x_t: AtomicEnvironment, d: int) -> [float, float]:
        # Kernel vector allows for evaluation of At. Env.
        k_v = self.get_kernel_vector(x_t, d)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance without cholesky (possibly faster)
        self_kern = self.kernel(x_t, x_t, d, d, self.hyps,
                                self.cutoffs)
        pred_var = self_kern - \
                   np.matmul(np.matmul(k_v, self.ky_mat_inv), k_v)

        # # get predictive variance (possibly slow)
        # v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        # self_kern = self.kernel(x_t, x_t, self.bodies, d, d, self.hyps,
        #                         self.cutoffs)
        # pred_var = self_kern - np.matmul(v_vec, v_vec)

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
        # get kernel vector
        k_v = self.en_kern_vec(x_t)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)
        self_kern = self.energy_kernel(x_t, x_t, self.hyps,
                                       self.cutoffs)
        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def get_kernel_vector(self, x: AtomicEnvironment,
                          d_1: int) -> np.ndarray:
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

        for m_index in range(size):
            x_2 = self.training_data[int(math.floor(m_index / 3))]
            d_2 = ds[m_index % 3]
            k_v[m_index] = self.kernel(x, x_2, d_1, d_2,
                                       self.hyps, self.cutoffs)
        return k_v

    def en_kern_vec(self, x: AtomicEnvironment) -> np.ndarray:
        ds = [1, 2, 3]
        size = len(self.training_data) * 3
        k_v = np.zeros(size, )

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
        if self.par:
            hyp_mat, ky_mat = \
                get_ky_and_hyp_par(self.hyps, self.training_data,
                                   self.training_labels_np,
                                   self.kernel_grad, self.cutoffs)
        else:
            hyp_mat, ky_mat = \
                get_ky_and_hyp(self.hyps, self.training_data,
                               self.training_labels_np,
                               self.kernel_grad, self.cutoffs)

        like, like_grad = \
            get_like_grad_from_mats(ky_mat, hyp_mat, self.training_labels_np)
        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv
        self.l_mat_inv = l_mat_inv

        self.like = like
        self.like_grad = like_grad

    def update_L_alpha(self):
        """
        Update the GP's L matrix and alpha vector.
        """

        # Set L matrix and alpha if set_L_alpha has not been called yet
        if self.l_mat is None:
            self.set_L_alpha()
            return

        n = self.l_mat_inv.shape[0]
        N = len(self.training_data)
        m = N - n // 3  # number of new data added
        ky_mat = np.zeros((3 * N, 3 * N))
        ky_mat[:n, :n] = self.ky_mat
        # calculate kernels for all added data
        for i in range(m):
            ind = n // 3 + i
            x_t = self.training_data[ind]
            k_vi = np.array([self.get_kernel_vector(x_t, d + 1)
                             for d in range(3)]).T  # (n+3m) x 3
            ky_mat[:, 3 * ind:3 * ind + 3] = k_vi
            ky_mat[3 * ind:3 * ind + 3, :n] = k_vi[:n, :].T
        sigma_n = self.hyps[-1]
        ky_mat[n:, n:] += sigma_n ** 2 * np.eye(3 * m)

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv
        self.l_mat_inv = l_mat_inv
