from copy import deepcopy
import os
import sys
import datetime
import time
import multiprocessing as mp
from typing import List
import numpy as np

from flare.mgp.mgp import MappedGaussianProcess
from flare.otf import OTF
from flare.struc import Structure
import flare.gp as gp
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.qe_util import run_espresso, parse_qe_input, \
    qe_input_to_structure, parse_qe_forces
from flare import output

class MGPOTF(OTF):

    def __init__(self, qe_input: str, dt: float, number_of_steps: int,
                 gp_model: gp.GaussianProcess, pw_loc: str,
                 std_tolerance_factor: float = 1,
                 prev_pos_init: np.ndarray=None, par: bool=False,
                 skip: int=0, init_atoms: List[int]=None,
                 calculate_energy=False, output_name='otf_run.out',
                 backup_name='otf_run_backup.out',
                 max_atoms_added=None, freeze_hyps=False,
                 rescale_steps=[], rescale_temps=[], add_all=False,
                 no_cpus=1, use_mapping=True, non_mapping_steps=[],
                 l_bound=None, two_d=False,
                 grid_params: dict={}, struc_params: dict={}):
                     
        '''
        On-the-fly training with MGP inserted, added parameters:
        :param non_mapping_steps: steps that not to use MGP
        :param l_bound: initial minimal interatomic distance
        :param two_d: if the system is a 2-D system
        :param grid_params: see class `MappedGaussianProcess`
        :param struc_params: see class `MappedGaussianProcess`
        '''

        super().__init__(qe_input, dt, number_of_steps,
                         gp_model, pw_loc, 
                         std_tolerance_factor, 
                         prev_pos_init, par,
                         skip, init_atoms, 
                         calculate_energy, output_name, 
                         backup_name,
                         max_atoms_added, freeze_hyps,
                         rescale_steps, rescale_temps, add_all,
                         no_cpus, use_mapping, non_mapping_steps,
                         l_bound, two_d)
        
        self.grid_params = grid_params
        self.struc_params = struc_params
        if par:
            self.pool = mp.Pool(processes=no_cpus)

    def predict_on_structure_par_mgp(self):
        args_list = [(atom, self.structure, self.gp.cutoffs, self.mgp) for atom in self.atom_list]
        results = self.pool.starmap(predict_on_atom_mgp, args_list)
        for atom in self.atom_list:
            res = results[atom]
            self.structure.forces[atom] = res[0]
            self.structure.stds[atom] = res[1]
            self.local_energies[atom] = res[2]
        self.structure.dft_forces = False


    def predict_on_structure_mgp(self): # changed
        """
        Assign forces to self.structure based on self.gp
        """

        output.write_to_output('\npredict with mapping:\n', self.output_name)
        for n in range(self.structure.nat):
            chemenv = AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            force, var = self.mgp.predict(chemenv)
            self.structure.forces[n][:] = force
            self.structure.stds[n][:] = np.sqrt(np.absolute(var))
        self.structure.dft_forces = False

    def train_mgp(self, skip=True):
        t0 = time.time()

        if self.l_bound < self.grid_params['bounds_2'][0,0]:
            self.grid_params['bounds_2'][0,0] = self.l_bound - 0.01
            self.grid_params['bounds_3'][0,:2] = np.ones(2)*self.l_bound - 0.01

        if skip and (self.curr_step in self.non_mapping_steps):
            return 1

        # set svd rank based on the training set, grid number and threshold 1000
        train_size = len(self.gp.training_data)
        rank_2 = np.min([1000, self.grid_params['grid_num_2'], train_size*3])
        rank_3 = np.min([1000, self.grid_params['grid_num_3'][0]**3, train_size*3])
        self.grid_params['svd_rank_2'] = rank_2
        self.grid_params['svd_rank_3'] = rank_3
       
        output.write_to_output('\ntraining set size: {}\n'.format(train_size),
                                self.output_name)
        output.write_to_output('lower bound: {}\n'.format(self.l_bound))
        output.write_to_output('mgp l_bound: {}\n'.format(self.grid_params['bounds_2'][0,0]))
        output.write_to_output('Constructing mapped force field...\n', 
                                self.output_name)
        self.mgp = MappedGaussianProcess(self.gp, self.grid_params, self.struc_params)
        output.write_to_output('building mapping time: {}'.format(time.time()-t0),
                                self.output_name)

        self.is_mgp_built = True

def predict_on_atom_mgp(atom, structure, cutoffs, mgp):
    chemenv = AtomicEnvironment(structure, atom, cutoffs)
    # predict force components and standard deviations
    force, var = mgp.predict(chemenv)
    comps = force
    stds = np.sqrt(np.absolute(var))

    # predict local energy
#     local_energy = self.gp.predict_local_energy(chemenv)
    local_energy = 0
    return comps, stds, local_energy


