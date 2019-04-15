import numpy as np
import multiprocessing as mp
import concurrent.futures
import copy
import sys
sys.path.append('../../flare/otf_engine')
import gp, env, struc, kernels, otf


class GPCalculator:
    def __init__(self, gp_model):
        self.gp_model = gp_model

    def get_potential_energy(self, atoms=None, force_consistent=False):
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

        for n in range(nat):
            chemenv = env.AtomicEnvironment(struc_curr, n,
                                            self.gp_model.cutoffs)
            for i in range(3):
                force, _ = self.gp_model.predict(chemenv, i + 1)
                forces[n][i] = float(force)

        return forces

    def calculation_required(self, atoms, quantities):
        return True
