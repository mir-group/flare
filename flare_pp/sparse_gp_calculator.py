from ase.calculators.calculator import Calculator, all_changes
from flare.utils.element_coder import NumpyEncoder
from flare_pp._C_flare import Structure
from flare_pp.sparse_gp import SGP_Wrapper
import numpy as np
import time, json
from copy import deepcopy

class SGP_Calculator(Calculator):

    implemented_properties = ["energy", "forces", "stress", "stds"]

    def __init__(self, sgp_model):
        super().__init__()
        self.gp_model = sgp_model
        self.results = {}
        self.use_mapping = False
        self.mgp_model = None

    # TODO: Figure out why this is called twice per MD step.
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties including: energy, local energies, forces,
            stress, uncertainties.
        """

        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if properties is None:
            properties = self.implemented_properties

        # Convert coded species to 0, 1, 2, etc.
        coded_species = []
        for spec in atoms.coded_species:
            coded_species.append(self.gp_model.species_map[spec])

        # Create structure descriptor.
        structure_descriptor = Structure(
            atoms.cell,
            coded_species,
            atoms.positions,
            self.gp_model.cutoff,
            self.gp_model.descriptor_calculators,
        )

        # Predict on structure.
        if self.gp_model.variance_type == "SOR":
            self.gp_model.sparse_gp.predict_SOR(structure_descriptor)
        elif self.gp_model.variance_type == "DTC":
            self.gp_model.sparse_gp.predict_DTC(structure_descriptor)
        elif self.gp_model.variance_type == "local":
            self.gp_model.sparse_gp.predict_local_uncertainties(structure_descriptor)

        # Set results.
        self.results["energy"] = deepcopy(structure_descriptor.mean_efs[0])
        self.results["forces"] = deepcopy(structure_descriptor.mean_efs[1:-6].reshape(-1, 3))

        # Convert stress to ASE format.
        flare_stress = deepcopy(structure_descriptor.mean_efs[-6:])
        ase_stress = -np.array(
            [
                flare_stress[0],
                flare_stress[3],
                flare_stress[5],
                flare_stress[4],
                flare_stress[2],
                flare_stress[1],
            ]
        )
        self.results["stress"] = ase_stress

        # Report negative variances, which can arise if there are numerical
        # instabilities.
        if (self.gp_model.variance_type == "SOR") or (
            self.gp_model.variance_type == "DTC"
        ):
            variances = structure_descriptor.variance_efs[1:-6]
            stds = np.zeros(len(variances))
            for n in range(len(variances)):
                var = variances[n]
                if var > 0:
                    stds[n] = np.sqrt(var)
                else:
                    stds[n] = -np.sqrt(np.abs(var))
            self.results["stds"] = stds.reshape(-1, 3)
        # The "local" variance type should be used only if the model has a
        # single atom-centered descriptor.
        # TODO: Generalize this variance type to multiple descriptors.
        elif self.gp_model.variance_type == "local":
            variances = structure_descriptor.local_uncertainties[0]
            sorted_variances = sort_variances(structure_descriptor, variances)
            stds = np.zeros(len(sorted_variances))
            for n in range(len(sorted_variances)):
                var = sorted_variances[n]
                if var > 0:
                    stds[n] = np.sqrt(var)
                else:
                    stds[n] = -np.sqrt(np.abs(var))
            stds_full = np.zeros((len(sorted_variances), 3))

            # Divide by the signal std to get a unitless value.
            stds_full[:, 0] = stds / self.gp_model.hyps[0]
            self.results["stds"] = stds_full

    def get_uncertainties(self, atoms):
        return self.get_property("stds", atoms)

    def calculation_required(self, atoms, quantities):
        return True

    def as_dict(self):
        out_dict = dict(vars(self))
        out_dict["gp_model"] = self.gp_model.as_dict()
        out_dict.pop("atoms")

        if "get_spin_polarized" in out_dict:
            out_dict.pop("get_spin_polarized")

        return out_dict

    @staticmethod
    def from_dict(dct):
        sgp, _ = SGP_Wrapper.from_dict(dct["gp_model"])
        calc = SGP_Calculator(sgp)
        calc.results = dct["results"]
        return calc

    def write_model(self, name):
        if ".json" != name[-5:]:
            name += ".json"
        with open(name, "w") as f:
            json.dump(self.as_dict(), f, cls=NumpyEncoder)

    @staticmethod
    def from_file(name):
        with open(name, "r") as f:
            gp_dict = json.loads(f.readline())
        sgp, kernels = SGP_Wrapper.from_dict(gp_dict["gp_model"])
        calc = SGP_Calculator(sgp)

        return calc, kernels


def sort_variances(structure_descriptor, variances):
    # Check that the variance length matches the number of atoms.
    assert len(variances) == structure_descriptor.noa
    sorted_variances = np.zeros(len(variances))

    # Sort the variances by atomic order.
    descriptor_values = structure_descriptor.descriptors[0]
    atom_indices = descriptor_values.atom_indices
    n_types = descriptor_values.n_types
    assert n_types == len(atom_indices)

    v_count = 0
    for s in range(n_types):
        for n in range(len(atom_indices[s])):
            atom_index = atom_indices[s][n]
            sorted_variances[atom_index] = variances[v_count]
            v_count += 1

    return sorted_variances
