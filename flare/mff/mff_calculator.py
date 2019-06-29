import numpy as np
import multiprocessing as mp
import concurrent.futures
import copy
import sys
sys.path.append('..')
from flare.env import AtomicEnvironment
from flare.struc import Structure

class MFFCalculator:
    def __init__(self, mff_model):
        self.mff_model = mff_model

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return 0

    def get_forces(self, atoms):
        nat = len(atoms)
        struc_curr = struc.Structure(atoms.cell, ['A']*nat,
                                     atoms.positions)

        forces = np.zeros((nat, 3))

        for n in range(nat):
            chemenv = AtomicEnvironment(struc_curr, n,
                                        self.mff_model.GP.cutoffs)
            force, _ = self.mff_model.predict(chemenv, mean_only=True)
        return forces

    def calculation_required(self, atoms, quantities):
        return True
