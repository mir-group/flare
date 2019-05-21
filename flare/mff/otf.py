from copy import deepcopy
import os
import sys
sys.path.append('../flare')
import datetime
import time
import multiprocessing as mp
from typing import List
import numpy as np

from mff import MappedForceField
from flare.otf import OTF
from flare.struc import Structure
import flare.gp as gp
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.qe_util import run_espresso, parse_qe_input, \
    qe_input_to_structure, parse_qe_forces

from ase import Atoms
from ase.calculators.eam import EAM
from ase.calculators.lammpsrun import LAMMPS

class MFFOTF(OTF):

    def __init__(self, qe_input: str, dt: float, number_of_steps: int,
                 gp_model: gp.GaussianProcess, pw_loc: str,
                 std_tolerance_factor: float = 1,
                 prev_pos_init: np.ndarray=None, par: bool=False,
                 skip: int=0, init_atoms: List[int]=None,
                 calculate_energy=False, output_name='otf_run.out',
                 max_atoms_added=None, freeze_hyps=False,
                 rescale_steps=[], rescale_temps=[], add_all=False,
                 no_cpus=1, use_mapping=True,
                 grid_params: dict={}, struc_params: dict={}):
                     
        super().__init__(qe_input, dt, number_of_steps,
                         gp_model, pw_loc, 
                         std_tolerance_factor, 
                         prev_pos_init, par,
                         skip, init_atoms, 
                         calculate_energy, output_name, 
                         max_atoms_added, freeze_hyps,
                         rescale_steps, rescale_temps, add_all,
                         no_cpus, use_mapping)
        
        self.grid_params = grid_params
        self.struc_params = struc_params
        if par:
            self.pool = mp.Pool(processes=no_cpus)

    def predict_on_structure_par_mff(self):
        args_list = [(atom, self.structure, self.gp.cutoffs, self.mff) for atom in self.atom_list]
        results = self.pool.starmap(predict_on_atom_mff, args_list)
        for atom in self.atom_list:
            res = results[atom]
            self.structure.forces[atom] = res[0]
            self.structure.stds[atom] = res[1]
            self.local_energies[atom] = res[2]
        self.structure.dft_forces = False


    def predict_on_structure_mff(self): # changed
        """
        Assign forces to self.structure based on self.gp
        """

        for n in range(self.structure.nat):
            chemenv = AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            force, var = self.mff.predict(chemenv)
            self.structure.forces[n][:] = force
            self.structure.stds[n][:] = np.sqrt(np.absolute(var))
        self.structure.dft_forces = False

    def train_mff(self):
        if self.grid_params['svd_rank'] <= 1000:
            self.grid_params['svd_rank'] += 3
        self.mff = MappedForceField(self.gp, self.grid_params, self.struc_params)                                   


def predict_on_atom_mff(atom, structure, cutoffs, mff):
    chemenv = AtomicEnvironment(structure, atom, cutoffs)
    # predict force components and standard deviations
    force, var = mff.predict(chemenv)
    comps = force
    stds = np.sqrt(np.absolute(var))

    # predict local energy
#     local_energy = self.gp.predict_local_energy(chemenv)
    local_energy = 0
    return comps, stds, local_energy


