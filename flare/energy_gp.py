import numpy as np
from gp import GaussianProcess
from flare.env import AtomicEnvironment


class EnergyGP(GaussianProcess):
    def __init__(self, kernel, kernel_grad, hyps, cutoffs, hyp_labels=None,
                 energy_force_kernel=None, energy_kernel=None,
                 opt_algorithm='L-BFGS-B', maxiter=10, par=False, output=None):

        GaussianProcess.__init__(self, kernel, kernel_grad, cutoffs,
                                 hyp_labels, energy_force_kernel,
                                 energy_kernel, opt_algorithm, maxiter,
                                 par, output)

        self.training_strucs = []
        self.training_atoms = []
        self.training_envs = []

    def update_db(self, structure, force_range=[]):
        """Add a structure to the training set."""

        # add structure and environments to training set
        self.training_strucs.append(structure)

        env_list = []
        for atom in structure.nat:
            env_curr = \
                AtomicEnvironment(structure, atom, self.cutoffs)
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

    def predict(self, x_t: AtomicEnvironment, d: int):
        """Predict force on atomic environment."""
        pass

    def predict_local_energy(self, x_t: AtomicEnvironment):
        """Predict local energy of atomic environment."""
        pass

    def get_kernel_vector(self, x: AtomicEnvironment, d_1: int):
        """Get kernel vector between a force component and the training set."""
        pass

        # for structure in self.training_strucs:
        #     if structure.energy is not None:

                
        # ds = [1, 2, 3]
        # size = len(self.training_data) * 3
        # k_v = np.zeros(size, )

        # for m_index in range(size):
        #     x_2 = self.training_data[int(math.floor(m_index / 3))]
        #     d_2 = ds[m_index % 3]
        #     k_v[m_index] = self.kernel(x, x_2, d_1, d_2,
        #                                self.hyps, self.cutoffs)
        # return k_v

    def en_kern_vec(self, x: AtomicEnvironment):
        """Get kernel vector between a local energy and the training set."""
        pass

    def set_L_alpha(self):
        """Set L matrix and alpha vector based on the current training set."""
        pass
