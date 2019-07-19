import numpy as np
import multiprocessing as mp
import concurrent.futures
import copy
import sys
sys.path.append('..')
from flare import gp, env, struc, kernels, otf
from ase.calculators.calculator import Calculator

class GPCalculator(Calculator):
    def __init__(self, gp_model):
        super().__init__() # all set to default values,TODO: change
        self.gp_model = gp_model

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return 0
        nat = len(atoms)
        struc_curr = struc.Structure(atoms.cell, ['A']*nat,
                                     atoms.positions)
        local_energies = np.zeros(nat)

        for n in range(nat):
            chemenv = env.AtomicEnvironment(struc_curr, n,
                                            self.gp_model.cutoffs)
            local_energies[n] = self.gp_model.predict_local_energy(chemenv)

        return np.sum(local_energies)

    def get_forces(self, atoms):
        nat = len(atoms)
        struc_curr = struc.Structure(atoms.cell, ['A']*nat,
                                     atoms.positions)

        forces = np.zeros((nat, 3))
        stds = np.zeros((nat, 3))
        for n in range(nat):
            chemenv = env.AtomicEnvironment(struc_curr, n,
                                            self.gp_model.cutoffs)
            for i in range(3):
                force, std = self.gp_model.predict(chemenv, i + 1)
                forces[n][i] = float(force)
                stds[n][i] = np.sqrt(np.absolute(std))

        self.results['stds'] = stds
        atoms.get_uncertainties = self.get_uncertainties

        return forces

    def get_stress(self, atoms):
        return np.eye(3)

    def calculation_required(self, atoms, quantities):
        return True

    def get_uncertainties(self):
        return self.results['stds']
