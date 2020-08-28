import numpy as np
from _C_flare import SparseGP_DTC, StructureDescriptor
from scipy.optimize import minimize
from typing import List


class SparseGP:
    """Wrapper class used to make the C++ sparse GP object compatible with
    OTF. Methods and properties are designed to mirror the GP class."""

    def __init__(
        self,
        kernels: List,
        descriptor_calculators: List,
        cutoff: float,
        many_body_cutoffs,
        sigma_e: float,
        sigma_f: float,
        sigma_s: float,
        species_map: dict,
        single_atom_energies: dict = None,
        energy_training=True,
        force_training=True,
        stress_training=True,
        max_iterations=10,
    ):

        self.sparse_gp = SparseGP_DTC(kernels, sigma_e, sigma_f, sigma_s)
        self.descriptor_calculators = descriptor_calculators
        self.cutoff = cutoff
        self.many_body_cutoffs = many_body_cutoffs
        self.hyps_mask = None
        self.species_map = species_map
        self.single_atom_energies = single_atom_energies
        self.energy_training = energy_training
        self.force_training = force_training
        self.stress_training = stress_training
        self.max_iterations = max_iterations

        # Make placeholder hyperparameter labels.
        self.hyp_labels = []
        for n in range(len(self.hyps)):
            self.hyp_labels.append("Hyp" + str(n))

    @property
    def training_data(self):
        return self.sparse_gp.training_structures

    @property
    def hyps(self):
        return self.sparse_gp.hyperparameters

    @property
    def hyps_and_labels(self):
        return self.hyps, self.hyp_labels

    @property
    def likelihood(self):
        return self.sparse_gp.log_marginal_likelihood

    @property
    def likelihood_gradient(self):
        return self.sparse_gp.likelihood_gradient

    @property
    def force_noise(self):
        return self.sparse_gp.sigma_f

    def __str__(self):
        return "Sparse GP model"

    def check_L_alpha(self):
        pass

    def write_model(self, name: str):
        pass

    def update_db(
        self,
        structure,
        forces,
        custom_range=(),
        energy: float = None,
        stress: "ndarray" = None,
    ):

        # Convert coded species to 0, 1, 2, etc.
        coded_species = []
        for spec in structure.coded_species:
            coded_species.append(self.species_map[spec])

        # Convert flare structure to structure descriptor.
        structure_descriptor = StructureDescriptor(
            structure.cell,
            coded_species,
            structure.positions,
            self.cutoff,
            self.many_body_cutoffs,
            self.descriptor_calculators,
        )

        # Add labels to structure descriptor.
        if (energy is not None) and (self.energy_training):
            # Sum up single atom energies.
            single_atom_sum = 0
            if self.single_atom_energies is not None:
                for spec in coded_species:
                    single_atom_sum += self.single_atom_energies[spec]

            # Correct the energy label and assign to structure.
            corrected_energy = energy - single_atom_sum
            structure_descriptor.energy = np.array([[corrected_energy]])

        if (forces is not None) and (self.force_training):
            structure_descriptor.forces = forces.reshape(-1)

        if (stress is not None) and (self.stress_training):
            structure_descriptor.stresses = stress

        # Assemble sparse environments.
        sparse_environments = []
        for sparse_index in custom_range:
            sparse_environments.append(
                structure_descriptor.local_environments[sparse_index]
            )

        # Update the sparse GP.
        self.sparse_gp.add_training_structure(structure_descriptor)
        self.sparse_gp.add_sparse_environments(sparse_environments)
        self.sparse_gp.update_matrices_QR()

    def set_L_alpha(self):
        # Taken care of in the update_db method.
        pass

    def train(self, logger_name=None):
        optimize_hyperparameters(self.sparse_gp, max_iterations=self.max_iterations)


def compute_negative_likelihood(hyperparameters, sparse_gp):
    """Compute the negative log likelihood and gradient with respect to the
    hyperparameters."""

    assert len(hyperparameters) == len(sparse_gp.hyperparameters)

    negative_likelihood = -sparse_gp.compute_likelihood_gradient(hyperparameters)
    negative_likelihood_gradient = -sparse_gp.likelihood_gradient

    # print("hyperparameters:")
    # print(hyperparameters)
    # print("likelihood gradient:")
    # print(-negative_likelihood_gradient)
    # print("likelihood:")
    # print(-negative_likelihood)
    # print("\n")

    return negative_likelihood, negative_likelihood_gradient


def optimize_hyperparameters(
    sparse_gp, display_results=True, gradient_tolerance=1e-4, max_iterations=10
):
    """Optimize the hyperparameters of a sparse GP model."""

    # Optimize the hyperparameters with BFGS.
    initial_guess = sparse_gp.hyperparameters
    arguments = sparse_gp

    optimization_result = minimize(
        compute_negative_likelihood,
        initial_guess,
        arguments,
        method="BFGS",
        jac=True,
        options={
            "disp": display_results,
            "gtol": gradient_tolerance,
            "maxiter": max_iterations,
        },
    )

    # Set the hyperparameters to the optimal value.
    sparse_gp.set_hyperparameters(optimization_result.x)
    sparse_gp.log_marginal_likelihood = -optimization_result.fun
    sparse_gp.likelihood_gradient = -optimization_result.jac

    return optimization_result


if __name__ == "__main__":
    pass
