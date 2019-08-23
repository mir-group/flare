import numpy as np
import multiprocessing as mp
import concurrent.futures
import copy
import sys
from flare import gp, env, struc, kernels, otf


class GPCalculator:
    def __init__(self, gp_model):
        self.gp_model = gp_model

    def get_potential_energy(self, atoms=None, force_consistent=False):
        nat = len(atoms)
        species = atoms.get_chemical_symbols()

        struc_curr = struc.Structure(atoms.cell[:], species,
                                     atoms.positions)
        local_energies = np.zeros(nat)

        for n in range(nat):
            chemenv = env.AtomicEnvironment(struc_curr, n,
                                            self.gp_model.cutoffs)
            local_energies[n] = self.gp_model.predict_local_energy(chemenv)

        return np.sum(local_energies)

    def get_forces(self, atoms):
        nat = len(atoms)
        species = atoms.get_chemical_symbols()

        struc_curr = struc.Structure(atoms.cell[:], species,
                                     atoms.positions)

        forces = np.zeros((nat, 3))

        for n in range(nat):
            chemenv = env.AtomicEnvironment(struc_curr, n,
                                            self.gp_model.cutoffs)
            for i in range(3):
                force, _ = self.gp_model.predict(chemenv, i + 1)
                forces[n][i] = float(force)

        return forces

    def get_uncertainties(self, atoms):
        nat = len(atoms)
        species = atoms.get_chemical_symbols()

        struc_curr = struc.Structure(atoms.cell[:], species,
                                     atoms.positions)

        uncertainties = np.zeros((nat, 3))

        for n in range(nat):
            chemenv = env.AtomicEnvironment(struc_curr, n,
                                            self.gp_model.cutoffs)
            for i in range(3):
                _, uncertainty = self.gp_model.predict(chemenv, i + 1)
                uncertainties[n][i] = float(uncertainty)

        return uncertainties

    def calculation_required(self, atoms, quantities):
        return True
