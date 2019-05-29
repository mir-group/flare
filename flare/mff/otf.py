from copy import deepcopy
import os
os.environ['LAMMPS_COMMAND'] = '/n/home08/xiey/lammps-16Mar18/src/lmp_mpi'
import sys
sys.path.append('../flare')
import datetime
import time
import multiprocessing as mp
from typing import List
import numpy as np

from flare.mff.mff import MappedForceField
from flare.otf import OTF
from flare.struc import Structure
import flare.gp as gp
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.qe_util import run_espresso, parse_qe_input, \
    qe_input_to_structure, parse_qe_forces
from flare import output

from ase import Atoms
from ase.calculators.eam import EAM
from ase.calculators.lammpsrun import LAMMPS
from ase.lattice.hexagonal import Graphene
from ase.build import make_supercell

class MFFOTF(OTF):

    def __init__(self, qe_input: str, dt: float, number_of_steps: int,
                 gp_model: gp.GaussianProcess, pw_loc: str,
                 std_tolerance_factor: float = 1,
                 prev_pos_init: np.ndarray=None, par: bool=False,
                 skip: int=0, init_atoms: List[int]=None,
                 calculate_energy=False, output_name='otf_run.out',
                 max_atoms_added=None, freeze_hyps=False,
                 rescale_steps=[], rescale_temps=[], add_all=False,
                 no_cpus=1, use_mapping=True, non_mapping_steps=[],
                 grid_params: dict={}, struc_params: dict={}):

        super().__init__(qe_input, dt, number_of_steps,
                         gp_model, pw_loc,
                         std_tolerance_factor,
                         prev_pos_init, par,
                         skip, init_atoms,
                         calculate_energy, output_name,
                         max_atoms_added, freeze_hyps,
                         rescale_steps, rescale_temps, add_all,
                         no_cpus, use_mapping, non_mapping_steps)

        self.grid_params = grid_params
        self.struc_params = struc_params
        if par:
            self.pool = mp.Pool(processes=no_cpus)

    def predict_on_structure_par_mff(self):
        args_list = [(atom, self.structure, self.gp.cutoffs, self.mff) for atom
                     in self.atom_list]
        results = self.pool.starmap(predict_on_atom_mff, args_list)
        for atom in self.atom_list:
            res = results[atom]
            self.structure.forces[atom] = res[0]
            self.structure.stds[atom] = res[1]
            self.local_energies[atom] = res[2]
        self.structure.dft_forces = False

    def predict_on_structure_mff(self):  # changed
        """
        Assign forces to self.structure based on self.gp
        """

        output.write_to_output('\npredict with mapping:\n', self.output_name)
        for n in range(self.structure.nat):
            chemenv = AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            force, var = self.mff.predict(chemenv)
            self.structure.forces[n][:] = force
            self.structure.stds[n][:] = np.sqrt(np.absolute(var))
        self.structure.dft_forces = False

    def train_mff(self):
        t0 = time.time()
        if self.curr_step in self.non_mapping_steps:
            return 1

        # set svd rank based on the training set, grid number and threshold 1000
        train_size = len(self.gp.training_data)
        rank_2 = np.min([1000, self.grid_params['grid_num_2'], train_size*3])
        rank_3 = np.min([1000, self.grid_params['grid_num_2'], train_size*3])
        self.grid_params['svd_rank_2'] = rank_2
        self.grid_params['svd_rank_3'] = rank_3
       
        output.write_to_output('\ntraining set size: {}\n'.format(train_size),
                                self.output_name)
        output.write_to_output('Constructing mapped force field...\n', 
                                self.output_name)
        self.mff = MappedForceField(self.gp, self.grid_params, self.struc_params)
        output.write_to_output('building mapping time: {}'.format(time.time()-t0),
                                self.output_name)

    def run_dft(self):
        output.write_to_output('\nCalling Lammps...\n',
                               self.output_name)

        # build ASE unitcell
        species = self.structure.species[0]
        positions = self.structure.positions
        nat = len(positions)
        cell = self.structure.cell
        symbols = species + str(nat)
        a = 2.46
        c = cell[2][2]
        unit_cell = Graphene(species, latticeconstant={'a':a,'c':c})
        multiplier = np.array([[7,0,0],[0,7,0],[0,0,1]])
        super_cell = make_supercell(unit_cell, multiplier)
        super_cell.positions = positions
        super_cell.cell = cell

        # calculate Lammps forces
        pot_path = '/n/home08/xiey/lammps-16Mar18/potentials/' 
        parameters = {'pair_style': 'airebo 5.0',
                      'pair_coeff': ['* * '+pot_path+'CH.airebo C'],
                      'mass': ['* 12.0107']}
        files = [pot_path+'CH.airebo']
        
#        calc = LAMMPS(keep_tmp_files=True, tmp_dir='lmp_tmp/', parameters=parameters, files=files)
        calc = LAMMPS(parameters=parameters, files=files)
        super_cell.set_calculator(calc)

        # calculate LAMMPS forces
        forces = super_cell.get_forces()
        self.structure.forces = forces

        # write wall time of DFT calculation
        self.dft_count += 1
        output.write_to_output('QE run complete.\n', self.output_name)
        time_curr = time.time() - self.start_time
        output.write_to_output('number of DFT calls: %i \n' % self.dft_count,
                               self.output_name)
        output.write_to_output('wall time from start: %.2f s \n' % time_curr,
                               self.output_name)


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
