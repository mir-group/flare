import json
from time import time
import numpy as np
from flare_pp import _C_flare
from flare_pp._C_flare import SparseGP, Structure, NormalizedDotProduct
from scipy.optimize import minimize
from typing import List
import warnings
from flare import struc
from flare.ase.atoms import FLARE_Atoms
from flare.utils.element_coder import NumpyEncoder


class SGP_Wrapper:
    """Wrapper class used to make the C++ sparse GP object compatible with
    OTF. Methods and properties are designed to mirror the GP class."""

    def __init__(
        self,
        kernels: List,
        descriptor_calculators: List,
        cutoff: float,
        sigma_e: float,
        sigma_f: float,
        sigma_s: float,
        species_map: dict,
        variance_type: str = "SOR",
        single_atom_energies: dict = None,
        energy_training=True,
        force_training=True,
        stress_training=True,
        max_iterations=10,
        opt_method="BFGS",
        bounds=None,
    ):

        self.sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s)
        self.descriptor_calculators = descriptor_calculators
        self.cutoff = cutoff
        self.hyps_mask = None
        self.species_map = species_map
        self.variance_type = variance_type
        self.single_atom_energies = single_atom_energies
        self.energy_training = energy_training
        self.force_training = force_training
        self.stress_training = stress_training
        self.max_iterations = max_iterations
        self.opt_method = opt_method
        self.bounds = bounds

        # Make placeholder hyperparameter labels.
        self.hyp_labels = []
        for n in range(len(self.hyps)):
            self.hyp_labels.append("Hyp" + str(n))

        # prepare a new sGP for variance mapping
        self.sgp_var = None
        if isinstance(
            kernels[0], NormalizedDotProduct
        ):  # TODO: adapt this to multiple kernels
            if kernels[0].power == 1:
                self.sgp_var_flag = "self"
                self.sgp_var = self.sparse_gp
            else:
                self.sgp_var_flag = "new"
        else:
            warnings.warn(
                "kernels[0] should be NormalizedDotProduct for variance mapping"
            )
            self.sgp_var_flag = None

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
        return self.sparse_gp.force_noise

    def __str__(self):
        gp_str = ""
        gp_str += f"Number of hyperparameters: {len(self.hyps)}\n"
        gp_str += f"Hyperparameter array: {str(self.hyps)}\n"

        if self.hyp_labels is None:
            # Put unlabeled hyperparameters on one line
            gp_str = gp_str[:-1]
            gp_str += str(self.hyps) + "\n"
        else:
            for hyp, label in zip(self.hyps, self.hyp_labels):
                gp_str += f"{label}: {hyp} \n"

        return gp_str

    def __len__(self):
        return len(self.training_data)

    def check_L_alpha(self):
        pass

    def write_model(self, name: str):
        """
        Write to .json file
        """
        if ".json" != name[-5:]:
            name += ".json"
        with open(name, "w") as f:
            json.dump(self.as_dict(), f, cls=NumpyEncoder)

    def as_dict(self):
        out_dict = {}
        for key in vars(self):
            if key not in ["sparse_gp", "sgp_var", "descriptor_calculators"]:
                out_dict[key] = getattr(self, key, None)

        # save descriptor_settings
        desc_calc = self.descriptor_calculators
        assert (len(desc_calc) == 1) and (isinstance(desc_calc[0], _C_flare.B2))
        b2_calc = desc_calc[0]
        b2_dict = {
            "type": "B2",
            "radial_basis": b2_calc.radial_basis,
            "cutoff_function": b2_calc.cutoff_function,
            "radial_hyps": b2_calc.radial_hyps,
            "cutoff_hyps": b2_calc.cutoff_hyps,
            "descriptor_settings": b2_calc.descriptor_settings,
        }
        out_dict["descriptor_calculators"] = [b2_dict]

        # save hyps
        out_dict["hyps"], out_dict["hyp_labels"] = self.hyps_and_labels

        # only save kernel type and hyps
        kernel_list = []
        for kern in self.sparse_gp.kernels:
            if isinstance(kern, NormalizedDotProduct):
                kernel_list.append(("NormalizedDotProduct", kern.sigma, kern.power))
            else:
                raise NotImplementedError
        out_dict["kernels"] = kernel_list

        kernel_list = []
        if getattr(self, "sgp_var_kernels", None):
            for kern in self.sgp_var_kernels:
                if isinstance(kern, NormalizedDotProduct):
                    kernel_list.append(("NormalizedDotProduct", kern.sigma, kern.power))
                else:
                    raise NotImplementedError
            out_dict["sgp_var_kernels"] = kernel_list

        out_dict["training_structures"] = []
        for s in range(len(self.training_data)):
            custom_range = self.sparse_gp.sparse_indices[0][s]
            struc_cpp = self.training_data[s]

            # invert mapping of species
            inv_species_map = {v: k for k, v in self.species_map.items()}
            species = [inv_species_map[s] for s in struc_cpp.species]

            # build training structure
            train_struc = struc.Structure(
                struc_cpp.cell,
                species,
                struc_cpp.positions,
            )
            train_struc.forces = struc_cpp.forces
            train_struc.stress = struc_cpp.stresses

            # Add back the single atom energies to dump the original energy
            single_atom_sum = 0
            if self.single_atom_energies is not None:
                for spec in struc_cpp.species:
                    single_atom_sum += self.single_atom_energies[spec]

            train_struc.energy = struc_cpp.energy + single_atom_sum

            out_dict["training_structures"].append(train_struc.as_dict())

        out_dict["sparse_indice"] = self.sparse_gp.sparse_indices
        return out_dict

    @staticmethod
    def from_dict(in_dict):
        """
        Need an initialized GP
        """
        # Recover kernel from checkpoint.
        kernel_list = in_dict["kernels"]
        assert len(kernel_list) == 1
        kernel_hyps = kernel_list[0]
        assert kernel_hyps[0] == "NormalizedDotProduct"
        sigma = float(kernel_hyps[1])
        power = int(kernel_hyps[2])
        kernel = NormalizedDotProduct(sigma, power)
        kernels = [kernel]

        # Recover descriptor from checkpoint.
        desc_calc = in_dict["descriptor_calculators"]
        assert len(desc_calc) == 1
        b2_dict = desc_calc[0]
        assert b2_dict["type"] == "B2"
        calc = _C_flare.B2(
            b2_dict["radial_basis"],
            b2_dict["cutoff_function"],
            b2_dict["radial_hyps"],
            b2_dict["cutoff_hyps"],
            b2_dict["descriptor_settings"],
        )

        # change the keys of single_atom_energies and species_map to int
        if in_dict["single_atom_energies"] is not None:
             sae_dict = {int(k): v for k, v in in_dict["single_atom_energies"].items()}
        else:
             sae_dict = None
        species_map = {int(k): v for k, v in in_dict["species_map"].items()}

        gp = SGP_Wrapper(
            kernels=[kernel],
            descriptor_calculators=[calc],
            cutoff=in_dict["cutoff"],
            sigma_e=in_dict["hyps"][-3],
            sigma_f=in_dict["hyps"][-2],
            sigma_s=in_dict["hyps"][-1],
            species_map=species_map,
            variance_type=in_dict["variance_type"],
            single_atom_energies=sae_dict,
            energy_training=in_dict["energy_training"],
            force_training=in_dict["force_training"],
            stress_training=in_dict["stress_training"],
            max_iterations=in_dict["max_iterations"],
            opt_method=in_dict["opt_method"],
            bounds=in_dict["bounds"],
        )

        # update db
        training_data = in_dict["training_structures"]
        for s in range(len(training_data)):
            custom_range = in_dict["sparse_indice"][0][s]
            train_struc = struc.Structure.from_dict(training_data[s])

            if len(train_struc.energy) > 0:
                energy = train_struc.energy[0]
            else:
                energy = None

            gp.update_db(
                train_struc,
                train_struc.forces,
                custom_range=custom_range,
                energy=energy,
                stress=train_struc.stress,
                mode="specific",
                sgp=None,
                update_qr=False,
            )

        gp.sparse_gp.update_matrices_QR()
        return gp, kernels

    @staticmethod
    def from_file(filename: str):
        with open(filename, "r") as f:
            in_dict = json.loads(f.readline())
        return SGP_Wrapper.from_dict(in_dict)

    def update_db(
        self,
        structure,
        forces,
        custom_range=(),
        energy: float = None,
        stress: "ndarray" = None,
        mode: str = "specific",
        sgp: SparseGP = None,  # for creating sgp_var
        update_qr=True,
    ):

        # Convert coded species to 0, 1, 2, etc.
        if isinstance(structure, (struc.Structure, FLARE_Atoms)):
            coded_species = []
            for spec in structure.coded_species:
                coded_species.append(self.species_map[spec])
        elif isinstance(structure, Structure):
            coded_species = structure.species
        else:
            raise Exception

        # Convert flare structure to structure descriptor.
        structure_descriptor = Structure(
            structure.cell,
            coded_species,
            structure.positions,
            self.cutoff,
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

        # Update the sparse GP.
        if sgp is None:
            sgp = self.sparse_gp

        sgp.add_training_structure(structure_descriptor)
        if mode == "all":
            if not custom_range:
                sgp.add_all_environments(structure_descriptor)
            else:
                raise Exception("Set mode='specific' for a user-defined custom_range")
        elif mode == "uncertain":
            if len(custom_range) == 1:  # custom_range gives n_added
                n_added = custom_range
                sgp.add_uncertain_environments(structure_descriptor, n_added)
            else:
                raise Exception(
                    "The custom_range should be set as [n_added] if mode='uncertain'"
                )
        elif mode == "specific":
            if not custom_range:
                warnings.warn(
                    "The mode='specific' but no custom_range is given, will not add sparse envs"
                )
            else:
                sgp.add_specific_environments(structure_descriptor, custom_range)
        elif mode == "random":
            if len(custom_range) == 1:  # custom_range gives n_added
                n_added = custom_range
                sgp.add_random_environments(structure_descriptor, n_added)
            else:
                raise Exception(
                    "The custom_range should be set as [n_added] if mode='random'"
                )
        else:
            raise NotImplementedError

        if update_qr:
            sgp.update_matrices_QR()

    def set_L_alpha(self):
        # Taken care of in the update_db method.
        pass

    def train(self, logger_name=None):
        optimize_hyperparameters(
            self.sparse_gp,
            max_iterations=self.max_iterations,
            method=self.opt_method,
            bounds=self.bounds,
        )

    def write_mapping_coefficients(self, filename, contributor, kernel_idx):
        self.sparse_gp.write_mapping_coefficients(filename, contributor, kernel_idx)

    def write_varmap_coefficients(self, filename, contributor, kernel_idx):
        old_kernels = self.sparse_gp.kernels
        assert (len(old_kernels) == 1) and (
            kernel_idx == 0
        ), "Not support multiple kernels"
        assert isinstance(old_kernels[0], NormalizedDotProduct)

        power = 1
        new_kernels = [NormalizedDotProduct(old_kernels[0].sigma, power)]

        # Build a power=1 SGP from scratch
        if self.sgp_var is None:
            print("Build a new SGP with power = 1")
            self.sgp_var, new_kernels = self.duplicate(new_kernels=new_kernels)
            self.sgp_var_kernels = new_kernels

        # Check hyperparameters and training data, if not match, construct a new SGP
        assert len(self.sgp_var.kernels) == 1
        assert np.allclose(self.sgp_var.kernels[0].power, 1.0)
        is_same_hyps = np.allclose(
            self.sgp_var.hyperparameters, self.sparse_gp.hyperparameters
        )
        n_sgp = len(self.training_data)
        n_sgp_var = len(self.sgp_var.training_structures)
        is_same_data = n_sgp == n_sgp_var 

        # Add new data if sparse_gp has more data than sgp_var
        if not is_same_data:
            n_add = n_sgp - n_sgp_var
            assert n_add > 0, "sgp_var has more training data than sgp"
            print("Training data not match, adding", n_add, "structures")
            for s in range(n_add):
                custom_range = self.sparse_gp.sparse_indices[0][s + n_sgp_var]
                struc_cpp = self.training_data[s + n_sgp_var]

                if len(struc_cpp.energy) > 0:
                    energy = struc_cpp.energy[0]
                else:
                    energy = None

                self.update_db(
                    struc_cpp,
                    struc_cpp.forces,
                    custom_range=custom_range,
                    energy=energy,
                    stress=struc_cpp.stresses,
                    mode="specific",
                    sgp=self.sgp_var,
                    update_qr=False,
                )

            self.sgp_var.update_matrices_QR()

        if not is_same_hyps:
            print("Hyps not match, set hyperparameters")
            self.sgp_var.set_hyperparameters(self.sparse_gp.hyperparameters)
            self.sgp_var_kernels = self.sgp_var.kernels

        new_kernels = self.sgp_var.kernels
        print("Map with current sgp_var")

        self.sgp_var.write_varmap_coefficients(filename, contributor, kernel_idx)

        return new_kernels

    def duplicate(self, new_hyps=None, new_kernels=None, new_powers=None):
        # TODO: change to __copy__ method
        # TODO: add compatibility with other kernels

        if new_hyps is None:
            hyps = self.sparse_gp.hyperparameters
        else:
            hyps = new_hyps

        if new_kernels is None:
            assert len(hyps) == len(self.sparse_gp.kernels) + 3
            kernels = []
            for k, kern in enumerate(self.sparse_gp.kernels):
                assert isinstance(kern, NormalizedDotProduct)
                if new_powers is not None:
                    power = new_powers[k]
                else:
                    power = kern.power
                kernels.append(NormalizedDotProduct(hyps[k], power))
        else:
            kernels = new_kernels

        n_kern = len(kernels)
        new_gp = SparseGP(kernels, hyps[n_kern], hyps[n_kern + 1], hyps[n_kern + 2])

        # add training data
        sparse_indices = self.sparse_gp.sparse_indices
        assert len(sparse_indices) == len(kernels)
        assert len(sparse_indices[0]) == len(self.training_data)

        for s in range(len(self.training_data)):
            custom_range = sparse_indices[0][s]
            struc_cpp = self.training_data[s]

            if len(struc_cpp.energy) > 0:
                energy = struc_cpp.energy[0]
            else:
                energy = None

            self.update_db(
                struc_cpp,
                struc_cpp.forces,
                custom_range=custom_range,
                energy=energy,
                stress=struc_cpp.stresses,
                mode="specific",
                sgp=new_gp,
                update_qr=False,
            )

        new_gp.update_matrices_QR()
        return new_gp, kernels


def compute_negative_likelihood(hyperparameters, sparse_gp,
                                print_vals=False):
    """Compute the negative log likelihood and gradient with respect to the
    hyperparameters."""

    assert len(hyperparameters) == len(sparse_gp.hyperparameters)

    sparse_gp.set_hyperparameters(hyperparameters)
    sparse_gp.compute_likelihood()
    negative_likelihood = -sparse_gp.log_marginal_likelihood

    if print_vals:
        print_hyps(hyperparameters, negative_likelihood)

    return negative_likelihood


def compute_negative_likelihood_grad(hyperparameters, sparse_gp,
                                     print_vals=False):
    """Compute the negative log likelihood and gradient with respect to the
    hyperparameters."""

    assert len(hyperparameters) == len(sparse_gp.hyperparameters)

    negative_likelihood = \
        -sparse_gp.compute_likelihood_gradient(hyperparameters)
    negative_likelihood_gradient = -sparse_gp.likelihood_gradient

    if print_vals:
        print_hyps_and_grad(
            hyperparameters, negative_likelihood_gradient, negative_likelihood
            )

    return negative_likelihood, negative_likelihood_gradient


def print_hyps(hyperparameters, neglike):
    print("Hyperparameters:")
    print(hyperparameters)
    print("Likelihood:")
    print(-neglike)
    print("\n")


def print_hyps_and_grad(hyperparameters, neglike_grad, neglike):
    print("Hyperparameters:")
    print(hyperparameters)
    print("Likelihood gradient:")
    print(-neglike_grad)
    print("Likelihood:")
    print(-neglike)
    print("\n")


def optimize_hyperparameters(
    sparse_gp,
    display_results=False,
    gradient_tolerance=1e-4,
    max_iterations=10,
    bounds=None,
    method="BFGS",
):
    """Optimize the hyperparameters of a sparse GP model."""

    initial_guess = sparse_gp.hyperparameters
    arguments = sparse_gp

    if method == "BFGS":
        optimization_result = minimize(
            compute_negative_likelihood_grad,
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

        # Assign likelihood gradient.
        sparse_gp.likelihood_gradient = -optimization_result.jac

    elif method == "L-BFGS-B":
        optimization_result = minimize(
            compute_negative_likelihood_grad,
            initial_guess,
            arguments,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "disp": display_results,
                "gtol": gradient_tolerance,
                "maxiter": max_iterations,
            },
        )

        # Assign likelihood gradient.
        sparse_gp.likelihood_gradient = -optimization_result.jac

    elif method == "nelder-mead":
        optimization_result = minimize(
            compute_negative_likelihood,
            initial_guess,
            arguments,
            method="nelder-mead",
            options={
                "disp": display_results,
                "maxiter": max_iterations,
            },
        )

    # Set the hyperparameters to the optimal value.
    sparse_gp.set_hyperparameters(optimization_result.x)
    sparse_gp.log_marginal_likelihood = -optimization_result.fun

    return optimization_result


if __name__ == "__main__":
    pass
