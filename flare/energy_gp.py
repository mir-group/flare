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

    def update_db(self, struc, force_range=[]):
        """Add a structure to the training set."""

        self.training_data.append(struc)

        # add energy to training labels if available
        if struc.energy is not None:
            self.training_labels_np = \
                np.append(self.training_labels_np,
                          struc.energy)

        # add forces to training set if available
        env_list = []
        update_indices = []
        if struc.forces is not None:
            noa = len(struc.positions)
            update_indices = force_range or list(range(noa))

            for atom in update_indices:
                env_curr = \
                    AtomicEnvironment(struc, atom, self.cutoffs)
                env_list.append(env_curr)
                self.training_labels_np = \
                    np.append(self.training_labels_np,
                              struc.forces[atom])

        self.training_atoms.append(update_indices)
        self.training_envs.append(env_list)

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

    def en_kern_vec(self, x: AtomicEnvironment):
        """Get kernel vector between a local energy and the training set."""
        pass

    def set_L_alpha(self):
        """Set L matrix and alpha vector based on the current training set."""
        pass
