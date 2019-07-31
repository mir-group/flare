import numpy as np
import multiprocessing as mp
import concurrent.futures
import copy
import sys
sys.path.append('..')
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.mff.mff_new import MappedForceField
from ase.calculators.calculator import Calculator

class FLARE_Calculator(Calculator):
    def __init__(self, gp_model, mff_model, use_mapping=False):
        super().__init__() # all set to default values,TODO: change
        self.mff_model = mff_model
        self.gp_model = gp_model
        self.use_mapping = use_mapping
        self.results = {}

    def get_potential_energy(self, atoms=None, force_consistent=False):
        # TODO: to be implemented
        return 1

        nat = len(atoms)
        struc_curr = Structure(atoms.cell, ['A']*nat,
                                     atoms.positions)
        local_energies = np.zeros(nat)

        for n in range(nat):
            chemenv = AtomicEnvironment(struc_curr, n,
                                        self.gp_model.cutoffs)
            local_energies[n] = self.gp_model.predict_local_energy(chemenv)

        return np.sum(local_energies)

    def get_forces(self, atoms):
        if self.use_mapping:
            return self.get_forces_mff(atoms)
        else:
            return self.get_forces_gp(atoms)

    def get_forces_gp(self, atoms):
        nat = len(atoms)
        struc_curr = Structure(atoms.cell, ['A']*nat,
                                     atoms.positions)

        forces = np.zeros((nat, 3))
        stds = np.zeros((nat, 3))
        for n in range(nat):
            chemenv = AtomicEnvironment(struc_curr, n,
                                        self.gp_model.cutoffs)
            for i in range(3):
                force, std = self.gp_model.predict(chemenv, i + 1)
                forces[n][i] = float(force)
                stds[n][i] = np.sqrt(np.absolute(std))

        self.results['stds'] = stds
        atoms.get_uncertainties = self.get_uncertainties

        return forces

    def get_forces_mff(self, atoms):
        nat = len(atoms)
        struc_curr = Structure(atoms.cell, ['A']*nat,
                                     atoms.positions)

        forces = np.zeros((nat, 3))
        stds = np.zeros((nat, 3))
        for n in range(nat):
            chemenv = AtomicEnvironment(struc_curr, n,
                                        self.mff_model.GP.cutoffs)
            f, v = self.mff_model.predict(chemenv, mean_only=False)
            forces[n] = f
            stds[n] = np.sqrt(np.absolute(v))

        self.results['stds'] = stds
        atoms.get_uncertainties = self.get_uncertainties
        return forces

    def get_stress(self, atoms):
        return np.eye(3)

    def calculation_required(self, atoms, quantities):
        return True

    def get_uncertainties(self):
        return self.results['stds']

    def train_gp(self, monitor=True):
        self.gp_model.train(monitor)

    def build_mff(self, skip=True):
        # l_bound not implemented

        if skip and (self.curr_step in self.non_mapping_steps):
            return 1

        # set svd rank based on the training set, grid number and threshold 1000
        grid_params = self.mff_model.grid_params
        struc_params = self.mff_model.struc_params

        train_size = len(self.gp_model.training_data)
        rank_2 = np.min([1000, grid_params['grid_num_2'], train_size*3])
        rank_3 = np.min([1000, grid_params['grid_num_3'][0]**3, train_size*3])
        grid_params['svd_rank_2'] = rank_2
        grid_params['svd_rank_3'] = rank_3
       
        self.mff_model = MappedForceField(self.gp_model, grid_params, struc_params)

