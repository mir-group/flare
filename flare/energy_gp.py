import numpy as np
from flare.energy_gp_algebra import get_ky_mat
from flare.gp_algebra import get_like_from_ky_mat
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
from scipy.linalg import solve_triangular


class EnergyGP(GaussianProcess):
    def __init__(self, kernel, force_energy_kernel, energy_kernel,
                 hyps, energy_noise, cutoffs, hyp_labels=None,
                 opt_algorithm='L-BFGS-B', maxiter=10, par=False, output=None,
                 kernel_grad=None):

        GaussianProcess.__init__(self, kernel, kernel_grad, hyps, cutoffs,
                                 hyp_labels, force_energy_kernel,
                                 energy_kernel, opt_algorithm, maxiter,
                                 par, output)

        self.training_strucs = []
        self.training_atoms = []
        self.training_envs = []
        self.energy_noise = energy_noise

    def update_db(self, structure, force_range=[], sweep=1):
        """Add a structure to the training set."""

        # add structure and environments to training set
        self.training_strucs.append(structure)

        env_list = []
        for atom in range(structure.nat):
            env_curr = \
                AtomicEnvironment(structure, atom, self.cutoffs, sweep)
            env_list.append(env_curr)
        self.training_envs.append(env_list)

        # add energy to training labels if available
        if structure.energy is not None:
            self.training_labels_np = \
                np.append(self.training_labels_np,
                          structure.energy)

        # add forces to training set if available
        update_indices = []
        if structure.forces is not None:
            noa = len(structure.positions)
            update_indices = force_range or list(range(noa))

            for atom in update_indices:
                self.training_labels_np = \
                    np.append(self.training_labels_np,
                              structure.forces[atom])

        self.training_atoms.append(update_indices)

    def train(self, output=None, grad_tol=1e-4):
        """Train GP model."""
        pass

    def get_kernel_vector(self, test_env: AtomicEnvironment, d_1: int):
        """Get kernel vector between a force component and the training set."""

        kernel_vector = np.zeros(len(self.training_labels_np))
        index = 0

        for struc_no, structure in enumerate(self.training_strucs):
            if structure.energy is not None:
                en_kern = 0
                for train_env in self.training_envs[struc_no]:
                    en_kern += \
                        self.force_energy_kernel(test_env, train_env, d_1,
                                                 self.hyps, self.cutoffs)
                kernel_vector[index] = en_kern
                index += 1

            if structure.forces is not None:
                for atom in self.training_atoms[struc_no]:
                    train_env = self.training_envs[struc_no][atom]
                    for d_2 in range(3):
                        kernel_vector[index] = \
                            self.kernel(test_env, train_env, d_1, d_2 + 1,
                                        self.hyps, self.cutoffs)
                        index += 1

        return kernel_vector

    def en_kern_vec(self, test_env: AtomicEnvironment):
        """Get kernel vector between a local energy and the training set."""
        kernel_vector = np.zeros(len(self.training_labels_np))
        index = 0

        for struc_no, structure in enumerate(self.training_strucs):
            if structure.energy is not None:
                en_kern = 0
                for train_env in self.training_envs[struc_no]:
                    en_kern += \
                        self.energy_kernel(test_env, train_env, self.hyps,
                                           self.cutoffs)
                kernel_vector[index] = en_kern
                index += 1

            if structure.forces is not None:
                for atom in self.training_atoms[struc_no]:
                    train_env = self.training_envs[struc_no][atom]
                    for d_2 in range(3):
                        kernel_vector[index] = \
                            self.force_energy_kernel(train_env, test_env,
                                                     d_2 + 1, self.hyps,
                                                     self.cutoffs)
                        index += 1

        return kernel_vector

    def global_en_kern_vec(self, test_struc: Structure):
        """Get kernel vector between a local energy and the training set."""
        kernel_vector = np.zeros(len(self.training_labels_np))

        for n in range(test_struc.nat):
            index = 0
            test_env = \
                AtomicEnvironment(test_struc, n, self.cutoffs)

            for struc_no, structure in enumerate(self.training_strucs):
                if structure.energy is not None:
                    en_kern = 0
                    for train_env in self.training_envs[struc_no]:
                        en_kern += \
                            self.energy_kernel(test_env, train_env, self.hyps,
                                               self.cutoffs)
                    kernel_vector[index] += en_kern
                    index += 1

                if structure.forces is not None:
                    for atom in self.training_atoms[struc_no]:
                        train_env = self.training_envs[struc_no][atom]
                        for d_2 in range(3):
                            kernel_vector[index] += \
                                self.force_energy_kernel(train_env, test_env,
                                                         d_2 + 1, self.hyps,
                                                         self.cutoffs)
                            index += 1

        return kernel_vector

    def predict_global_energy(self, test_struc: Structure):
        """Predict the global energy of a structure and its uncertainty."""

        # get kernel vector
        k_v = self.global_en_kern_vec(test_struc)

        # get predictive mean
        pred_mean = np.matmul(k_v, self.alpha)

        # get predictive variance
        v_vec = solve_triangular(self.l_mat, k_v, lower=True)

        self_kern = 0
        for m in range(test_struc.nat):
            x_1 = AtomicEnvironment(test_struc, m, self.cutoffs)
            for n in range(test_struc.nat):
                x_2 = AtomicEnvironment(test_struc, n, self.cutoffs)
                self_kern += self.energy_kernel(x_1, x_2, self.hyps,
                                                self.cutoffs)
        pred_var = self_kern - np.matmul(v_vec, v_vec)

        return pred_mean, pred_var

    def set_L_alpha(self):
        """Set L matrix and alpha vector based on the current training set."""

        ky_mat = \
            get_ky_mat(self.hyps, self.energy_noise, self.training_strucs,
                       self.training_envs, self.training_atoms,
                       self.training_labels_np, self.kernel,
                       self.force_energy_kernel, self.energy_kernel,
                       self.cutoffs)

        like = \
            get_like_from_ky_mat(ky_mat, self.training_labels_np)

        l_mat = np.linalg.cholesky(ky_mat)
        l_mat_inv = np.linalg.inv(l_mat)
        ky_mat_inv = l_mat_inv.T @ l_mat_inv
        alpha = np.matmul(ky_mat_inv, self.training_labels_np)

        self.ky_mat = ky_mat
        self.l_mat = l_mat
        self.alpha = alpha
        self.ky_mat_inv = ky_mat_inv
        self.l_mat_inv = l_mat_inv
        self.likelihood = like

if __name__ == '__main__':
    pass
