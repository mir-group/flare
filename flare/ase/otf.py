'''
:class:`OTF` is the on-the-fly training module for ASE, WITHOUT molecular dynamics engine. 
It needs to be used adjointly with ASE MD engine. Please refer to our 
`OTF MD module <https://flare.readthedocs.io/en/latest/flare/ase/otf_md.html>`_ for the
complete training module with OTF and MD.
'''
import os
import sys
import inspect
from time import time
from copy import deepcopy

from flare.struc import Structure
from flare.gp import GaussianProcess
from flare.util import is_std_in_bound
from flare.mgp.utils import get_l_bound

import numpy as np
from ase import units
from ase.calculators.espresso import Espresso


class OTF:
    """
    OTF (on-the-fly) training with the ASE interface. 
    
    Note: Dft calculator is set outside of the otf module, and input as 
        dft_calc, so that different calculators can be used

    Args:
        dft_calc (ASE Calculator): the ASE DFT calculator (see ASE documentaion)
        dft_count (int): initial number of DFT calls
        std_tolerance_factor (float): the threshold of calling DFT = noise * 
            std_tolerance_factor
        init_atoms (list): the list of atoms in the first DFT call to add to
            the training set, since there's no uncertainty prediction initially
        calculate_energy (bool): if True, the energy will be calculated;
            otherwise, only forces will be predicted
        max_atoms_added (int): the maximal number of atoms to add to the 
            training set after each DFT calculation
        freeze_hyps (int or None): the hyperparameters will only be trained for
            the first `freeze_hyps` DFT calls, and will be fixed after that
        restart_from (str or None): the path of the directory that stores the
            training data from last OTF run, and this OTF will restart from it

    Other Parameters:
        use_mapping (bool): if True, the MGP will be used
        non_mapping_steps (list): a list of steps that MGP will not be 
            constructed and used
        l_bound (float): the lower bound of the interatomic distance, used for 
            MGP construction
        two_d (bool): used in the calculation of l_bound. If 2-D material is 
            considered, set to True, then the atomic environment construction 
            will only search the x & y periodic boundaries to save time
    """

    def __init__(self, 
            # on-the-fly parameters
            dft_calc=None, dft_count=None, std_tolerance_factor: float=1, 
            skip: int=0, init_atoms: list=[], calculate_energy=False, 
            max_atoms_added=1, freeze_hyps=1, restart_from=None,
            # mgp parameters
            use_mapping: bool=False, non_mapping_steps: list=[],
            l_bound: float=None, two_d: bool=False):

        # get all arguments as attributes 
        arg_dict = inspect.getargvalues(inspect.currentframe())[3]
        del arg_dict['self']
        self.__dict__.update(arg_dict)

        if dft_count is None:
            self.dft_count = 0
        self.noa = len(self.atoms.positions)

        # initialize local energies
        if calculate_energy:
            self.local_energies = np.zeros(self.noa)
        else:
            self.local_energies = None

        # initialize otf
        if init_atoms is None:
            self.init_atoms = [int(n) for n in range(self.noa)]

    def otf_run(self, steps, rescale_temp=[], rescale_steps=[]):
        """
        Use `otf_run` intead of `run` to perform a number of time steps.

        Args:
            steps (int): the number of time steps

        Other Parameters:
            rescale_temp (list): a list of temepratures that rescale the system
            rescale_steps (list): a list of step numbers that the temperature
                rescaling in `rescale_temp` is done

        Example:
            # rescale temperature to 500K and 1000K at the 100th and 200th step
            rescale_temp = [500, 1000]
            rescale_steps = [100, 200]
        """

        self.start_time = time()
        # observers
        for i, obs in enumerate(self.observers):
            if obs[0].__class__.__name__ == "OTFLogger":
                self.logger_ind = i
                self.output = self.observers[self.logger_ind][0]
                self.output.write_header_info()
                break

        # initialize gp by a dft calculation
        calc = self.atoms.calc
        calc.mgp_updated = False

        # restart from previous OTF training
        if self.restart_from is not None:
            self.restart()
            f = self.atoms.calc.results['forces']

        if not calc.gp_model.training_data:
            self.dft_count = 0
            self.stds = np.zeros((self.noa, 3))
            dft_forces = self.run_dft()
            f = dft_forces
   
            # update gp model
            curr_struc = Structure.from_ase_atoms(self.atoms)
            self.l_bound = get_l_bound(100, curr_struc, self.two_d)
            print('l_bound:', self.l_bound)

            calc.gp_model.update_db(curr_struc, dft_forces,
                           custom_range=self.init_atoms)

            # train calculator
            for atom in self.init_atoms:
                # the observers[0][0] is the logger
                self.output.add_atom_info(atom, 
                    self.stds[atom])
            self.train_gp()

            if self.use_mapping:
                self.build_mgp()

  
        if self.md_engine == 'NPT':
            if not self.initialized:
                self.initialize()
            else:
                if self.have_the_atoms_been_changed():
                    raise NotImplementedError(
                        "You have modified the atoms since the last timestep.")

        step_0 = self.nsteps
        for i in range(step_0, steps):
            print('step:', i)

            calc.results = {} # clear the calculation from last step
            self.stds = np.zeros((self.noa, 3))

            # temperature rescaling
            if self.nsteps in rescale_steps:
                temp = rescale_temp[rescale_steps.index(self.nsteps)]
                curr_velocities = self.atoms.get_velocities()
                curr_temp = self.atoms.get_temperature()
                self.atoms.set_velocities(curr_velocities *\
                                          np.sqrt(temp/curr_temp))

            if self.md_engine == 'NPT':
                self.step()
            else:
                f = self.step(f)
            self.nsteps += 1
            self.stds = self.atoms.get_uncertainties(self.atoms)

            # figure out if std above the threshold
            self.call_observers() 
            curr_struc = Structure.from_ase_atoms(self.atoms)
            self.l_bound = get_l_bound(self.l_bound, curr_struc, self.two_d)
            curr_struc.stds = np.copy(self.stds)
            noise = calc.gp_model.hyps[-1]
            std_in_bound, target_atoms = \
                    is_std_in_bound(self.std_tolerance_factor,
                                    noise, curr_struc, 
                                    self.max_atoms_added)

            if not self.std_in_bound:
                # call dft/eam
                print('calling dft')
                dft_forces = self.run_dft()

                # compute mae and write to output
                gp_forces = self.atoms.get_forces(self.atoms)
                mae = np.mean(np.abs(gp_forces - dft_forces))
                mac = np.mean(np.abs(dft_forces))

                self.output.logfile.write_to_log('\nmean absolute error:'
                                         ' %.4f eV/A \n' % mae)
                self.output.logfile.write_to_log('mean absolute dft component:'
                                             ' %.4f eV/A \n' % mac)

                # update gp
                print('updating gp')
                self.update_gp(target_atoms, dft_forces)
                self.train_gp()
                calc.mgp_updated = False

            if self.use_mapping:
                self.build_mgp()

        self.output.run_complete()

    
    def build_mgp(self):
        # build mgp
        calc = self.atoms.calc
        if self.nsteps in self.non_mapping_steps:
            calc.use_mapping = False
            skip = True
        else: 
            calc.use_mapping = True

            if calc.mgp_updated:
                skip = True
            else:
                skip = False
                calc.mgp_updated = True

        calc.build_mgp(skip)


    def run_dft(self):
        # change from FLARE to DFT calculator
        self.dft_calc.nsteps = self.nsteps
        prev_calc = self.atoms.calc
        calc = deepcopy(self.dft_calc)
        self.atoms.set_calculator(calc)

        self.output.logfile.write_to_log('\nCalling DFT...\n')

        # calculate DFT forces 
        forces = self.atoms.get_forces()

        # write configuration to files
        self.call_observers()

        # set back to flare calculator
        self.atoms.set_calculator(prev_calc)

        # write wall time of DFT calculation
        self.dft_count += 1
        self.output.conclude_dft(self.dft_count, self.start_time)

        return forces

    def update_gp(self, train_atoms, dft_forces):
        # write added atoms to logfile
        self.output.logfile.add_atom_info(train_atoms, self.stds)
 
        # build gp structure from atoms
        atom_struc = Structure.from_ase_atoms(self.atoms)

        # update gp model
        gp_model = self.atoms.calc.gp_model
        gp_model.update_db(atom_struc, dft_forces,
                           custom_range=train_atoms)

        if gp_model.alpha is None:
            gp_model.set_L_alpha()
        else:
            gp_model.update_L_alpha()

        self.output.added_atoms_dat.write('\n')

    def train_gp(self, skip=False):
        gp_model = self.atoms.calc.gp_model
        if (self.dft_count-1) < self.freeze_hyps:
            gp_model.train(self.output.logfile)
            self.output.write_hyps()
        else:
            gp_model.set_L_alpha()

        # save gp_model everytime after training
        gp_model.write_model('otf_data/gp_model', format='pickle')

    def restart(self):
        # Recover atomic configuration: positions, velocities, forces
        positions, self.nsteps = self.read_frame('positions.xyz', -1)
        self.atoms.set_positions(positions)
        self.atoms.set_velocities(self.read_frame('velocities.dat', -1)[0])
        self.atoms.calc.results['forces'] = self.read_frame('forces.dat', -1)[0]
        print('Last frame recovered')

        # Recover FLARE calculator
        gp_pickle = self.restart_from + '/gp_model.pickle'
        self.atoms.calc.gp_model = GaussianProcess.from_file(gp_pickle)

        if self.atoms.calc.use_mapping:
            for map_3 in self.atoms.calc.mgp_model.maps_3:
                map_3.load_grid = self.restart_from + '/'
            self.atoms.calc.build_mgp(skip=False)
            self.atoms.calc.mgp_updated = True
        print('GP and MGP ready')

        self.l_bound = 10

    def read_all_frames(self, filename, nat, header=2, elem_type='xyz'):
        frames = []
        with open(self.restart_from+'/'+filename) as f:
            lines = f.readlines()
            frame_num = len(lines) // (nat+header)
            for i in range(frame_num):
                start = (nat+header) * i + header
                curr_frame = lines[start:start+nat]
                properties = []
                for line in curr_frame:
                    line = line.split()
                    if elem_type == 'xyz':
                        xyz = [float(l) for l in line[1:]]
                        properties.append(xyz)
                    elif elem_type == 'int':
                        properties = [int(l) for l in line]
                frames.append(properties)
        return np.array(frames)


    def read_frame(self, filename, frame_num):
        nat = len(self.atoms.positions)
        with open(self.restart_from+'/'+filename) as f:
            lines = f.readlines()
            if frame_num == -1: # read the last frame
                start_line = - (nat+2)
                frame = lines[start_line:]
            else:
                start_line = frame_num * (nat+2)
                end_line = (frame_num+1) * (nat+2)
                frame = f.lines[start_line:end_line]

            properties = []
            for line in frame[2:]:
                line = line.split()
                properties.append([float(d) for d in line[1:]])
        return np.array(properties), len(lines)//(nat+2)



