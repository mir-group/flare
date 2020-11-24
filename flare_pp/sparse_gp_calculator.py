from ase.calculators.calculator import Calculator
from _C_flare import SparseGP_DTC, Structure
import numpy as np


class SGP_Calculator(Calculator):
    def __init__(self, sgp_model):
        super().__init__()
        self.gp_model = sgp_model
        self.results = {}
        self.use_mapping = False
        self.mgp_model = None

    def calculate(self, atoms):
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
        self.gp_model.sparse_gp.predict_on_structure(structure_descriptor)

        # Set results.
        self.results["energy"] = structure_descriptor.mean_efs[0]
        self.results["forces"] = structure_descriptor.mean_efs[1:-6].reshape(-1, 3)

        # Convert stress to ASE format.
        flare_stress = structure_descriptor.mean_efs[-6:]
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

        self.results["stds"] = np.sqrt(
            structure_descriptor.variance_efs[1:-6].reshape(-1, 3)
        )

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.results.keys():
            if not allow_calculation:
                return None
            self.calculate(atoms)
        return self.results[name]

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.get_property("energy", atoms)

    def get_forces(self, atoms):
        return self.get_property("forces", atoms)

    def get_stress(self, atoms):
        return self.get_property("stress", atoms)

    def get_uncertainties(self, atoms):
        return self.get_property("stds", atoms)

    def calculation_required(self, atoms, quantities):
        return True
