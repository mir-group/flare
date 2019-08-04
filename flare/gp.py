import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from typing import List, Callable
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.gp_algebra import get_ky_mat, get_ky_and_hyp, get_like_from_ky_mat,\
    get_like_grad_from_mats, get_neg_likelihood, get_neg_like_grad


class GaussianProcess:
    """ Gaussian Process Regression Model.

    Implementation is based on Algorithm 2.1 (pg. 19) of
    "Gaussian Processes for Machine Learning" by Rasmussen and Williams"""

    def __init__(self, kernel: Callable,
                 kernel_grad: Callable,  hyps: np.ndarray,
                 cutoffs: np.ndarray,
                 hyp_labels: List=None,
                 energy_force_kernel: Callable=None,
                 energy_kernel: Callable=None,
                 opt_algorithm: str='L-BFGS-B',
                 maxiter=10, par=False):
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

        args = (self.training_data, self.training_labels_np,
                self.kernel_grad, self.cutoffs, monitor,
                self.par)

        if self.algo == 'L-BFGS-B':

            # bound signal noise below to avoid overfitting
            bounds = np.array([(-np.inf, np.inf)] * len(x_0))
            bounds[-1] = (1e-6, np.inf)

            # Catch linear algebra errors and switch to BFGS if necessary
            try:
                res = minimize(get_neg_like_grad, x_0, args,
                               method='L-BFGS-B', jac=True, bounds=bounds,
                               options={'disp': False, 'gtol': 1e-4,
                                        'maxiter': self.maxiter})
            except:
                print("Warning! Algorithm for L-BFGS-B failed. Changing to "
                      "BFGS for remainder of run.")
                self.algo = 'BFGS'

        if custom_bounds is not None:
            res = minimize(get_neg_like_grad, x_0, args,
                           method='L-BFGS-B', jac=True, bounds=custom_bounds,
                           options={'disp': False, 'gtol': 1e-4,
                                    'maxiter': self.maxiter})

        elif self.algo == 'BFGS':
            res = minimize(get_neg_like_grad, x_0, args,
                           method='BFGS', jac=True,
                           options={'disp': False, 'gtol': 1e-4,
                                    'maxiter': self.maxiter})

        elif self.algo == 'nelder-mead':
            res = minimize(get_neg_likelihood, x_0, args,
                           method='nelder-mead',
                           options={'disp': False,
                                    'maxiter': self.maxiter,
                                    'xtol': 1e-5})

        self.hyps = res.x
        self.set_L_alpha()
        self.likelihood = -res.fun
        self.likelihood_gradient = -res.jac

    def predict(self, x_t: AtomicEnvironment, d: int) -> [float, float]:
        # get kernel vector
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
        """Predict the sum of triplet energies that include the test atom.

        :param x_t: Atomic environment of test atom
        :type x_t: AtomicEnvironment
        :return: local energy in eV (up to a constant)
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
        """ Compute kernel vector.

        :param x: data point to compare against kernel matrix
        :type x: AtomicEnvironment
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

        hyp_mat, ky_mat = get_ky_and_hyp(self.hyps, self.training_data,
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
        n = self.l_mat_inv.shape[0]
        N = len(self.training_data)
        m = N - n//3  # number of data added
        ky_mat = np.zeros((3*N, 3*N))
        ky_mat[:n, :n] = self.ky_mat
        # calculate kernels for all added data
        for i in range(m):
            ind = n//3 + i
            x_t = self.training_data[ind]
            k_vi = np.array([self.get_kernel_vector(x_t, d+1)
                             for d in range(3)]).T  # (n+3m) x 3
            ky_mat[:,  3*ind:3*ind+3] = k_vi
            ky_mat[3*ind:3*ind+3, :n] = k_vi[:n, :].T
        sigma_n = self.hyps[-1]
        ky_mat[n:, n:] += sigma_n**2 * np.eye(3*m)

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv
        self.l_mat_inv = l_mat_inv

    def update_L_alpha_v1(self):
        '''
        1. This function is used right after "update_db".
        2. It can update the l_mat_inv, k_mat_inv and alpha
        without computing the whole kernel matrix.
        3. The condition is the hyps should be frozen.
        4. See notes for derivation and calculation details
        '''
        n = self.l_mat_inv.shape[0]
        m = len(self.training_data) - n//3  # number of data added
        sigma_n = self.hyps[-1]

        # update l_mat_inv => k_mat_inv
        k_v = np.array([[] for i in range(n)])
        V_mat = np.zeros((3*m, 3*m))

        # calculate kernels for all added data
        for i in range(m):
            x_t = self.training_data[-1-i]
            k_vi = np.array([self.get_kernel_vector(x_t, d+1)
                             for d in range(3)]).T  # (n+3m) x 3
            k_vi = k_vi[:n, :]
            k_v = np.hstack([k_v, k_vi])  # n x 3m
            for d1 in range(3):
                for j in range(i, m):
                    y_t = self.training_data[-1-j]
                    for d2 in range(3):
                        V_mat[3*i+d1, 3*j+d2] = \
                            self.kernel(x_t, y_t, d1+1, d2+1,
                                        self.hyps, self.cutoffs)
                        V_mat[3*j+d2, 3*i+d1] = V_mat[3*i+d1, 3*j+d2]
        v = self.l_mat_inv @ k_v  # n x 3m
        V_mat += sigma_n**2 * np.eye(3*m) - v.T @ v
        r_n1 = np.linalg.inv(np.linalg.cholesky(V_mat))  # 3m x 3m
        r = - self.l_mat_inv.T @ v @ r_n1.T  # n x 3m
        new_line1 = np.hstack([self.l_mat_inv, np.zeros((n, 3*m))])
        new_line2 = np.hstack([r.T, r_n1])
        self.l_mat_inv = np.vstack([new_line1, new_line2])
        self.ky_mat_inv = self.l_mat_inv.T @ self.l_mat_inv

        # update alpha
        self.alpha = self.ky_mat_inv @ self.training_labels_np
