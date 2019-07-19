import os
import sys
sys.path.append('..')
from flare.struc import Structure

import numpy as np
from ase.calculators.espresso import Espresso
from ase.calculators.eam import EAM
from ase.md.md import MolecularDynamics
from ase import units


class OTF(MolecularDynamics):
    def __init__(self, atoms, timestep,  
            trajectory=None, 
            # on-the-fly parameters
            pw_loc=None, dft_input={}, 
            std_tolerance_factor: float=1, 
            prev_pos_init: np.ndarray=None, par:bool=False,
            skip: int=0, init_atoms: list=[], 
            calculate_energy=False,
            max_atoms_added=1, freeze_hyps=1, 
            rescale_steps=[], rescale_temps=[], add_all=False,
            no_cpus=1, npool=1, 
            # mff parameters
            use_mapping: bool=False,
            non_mapping_steps: list=[],
            l_bound: float=None, two_d: bool=False):

        '''
        output_name => logfile
        npool: added
        prev_pos_init: relaunch mode not implemented
        par: consider merging into gp_calculator instead of here
        skip: not implemented
        calculate_energy: not implemented
        l_bound: mff update l_bound, not implemented

        TODO:
        1. replace Structure with the ase internal Atoms
        2. mff_calculator: self.gp_model
        '''

        MolecularDynamics.__init__(self, atoms, timestep, trajectory)
                                   
        self.std_tolerance = std_tolerance_factor
        self.noa = len(atoms.positions)
        self.max_atoms_added = max_atoms_added
        self.freeze_hyps = freeze_hyps
        self.dft_input = dft_input

        # set up pw.x command
        pwi_file = self.dft_input['label'] + '.pwi'
        pwo_file = self.dft_input['label'] + '.pwo'
        os.environ['ASE_ESPRESSO_COMMAND'] = 'srun -n {0} --mpi=pmi2 {1} -npool {2} < {3} > {4}'.format(no_cpus, pw_loc, npool, pwi_file, pwo_file)

        # params for mapped force field
        self.use_mapping = use_mapping
        self.non_mapping_steps = non_mapping_steps
        self.two_d = two_d

        # initialize local energies
        if calculate_energy:
            self.local_energies = np.zeros(self.noa)
        else:
            self.local_energies = None

        # initialize otf
        if init_atoms is None:
            self.init_atoms = [int(n) for n in range(self.noa)]
        else:
            self.init_atoms = init_atoms

        # rescale temperatures
        self.rescale_steps = rescale_steps
        self.rescale_temps = rescale_temps


    def is_std_in_bound(self, atom_list):
        # set uncertainty threshold
        if self.std_tolerance == 0:
            self.std_in_bound = True
            self.target_atom = -1
        elif self.std_tolerance > 0:
            noise = np.abs(self.atoms.calc.gp_model.hyps[-1])
            threshold = self.std_tolerance * noise
        else:
            threshold = np.abs(self.std_tolerance)

        # find max std
        max_std = 0
        for atom, std in enumerate(self.stds):
            std_curr = np.max(std)

            if std_curr > max_std:
                max_std = std_curr
                target_atom = atom

        # if above threshold, return atom
        if max_std > threshold and atom not in atom_list:
            self.std_in_bound = False
            self.target_atom = target_atom
        else:
            self.std_in_bound = True
            self.target_atom = -1

    def otf_run(self, steps):
        """Perform a number of time steps."""
        # initialize gp by a dft calculation
        if not self.atoms.calc.gp_model.training_data:
            self.dft_count = 0
            self.std_in_bound = False
            self.target_atom = 0
            self.stds = []
            dft_forces = self.call_DFT()
   
            # update gp model
            atom_struc = Structure(self.atoms.cell, 
                    ['A']*len(self.atoms.positions), 
                    self.atoms.positions)
            gp_model = self.atoms.calc.gp_model
            gp_model.update_db(atom_struc, dft_forces,
                           custom_range=self.init_atoms)

            # train calculator
            self.train()

        if not self.initialized:
            self.initialize()
        else:
            if self.have_the_atoms_been_changed():
                raise NotImplementedError(
                    "You have modified the atoms since the last timestep.")

        for i in range(steps):
            print('step:', i)
            self.step()
            self.nsteps += 1
            self.stds = self.atoms.get_uncertainties()

            # figure out if std above the threshold
            self.call_observers() 
            self.is_std_in_bound([])

            if not self.std_in_bound:
                # call dft/eam
                print('calling dft')
                dft_forces = self.call_DFT()

                # update gp
                print('updating gp')
                self.update_GP(dft_forces)

    
    def call_DFT(self):
        prev_calc = self.atoms.calc
        pseudopotentials = self.dft_input['pseudopotentials']
#        label = self.dft_input['label']
#        input_data = self.dft_input['input_data']
#        kpts = self.dft_input['kpts']
#        calc = Espresso(pseudopotentials=pseudopotentials, label=label, 
#                        tstress=True, tprnfor=True, nosym=True, 
#                        input_data=input_data, kpts=kpts) 
        calc = EAM(potential=pseudopotentials)        
        self.atoms.set_calculator(calc)
        forces = self.atoms.get_forces()
        self.call_observers()
        self.atoms.set_calculator(prev_calc)
        self.dft_count += 1
        return forces

    def update_GP(self, dft_forces):
        atom_count = 0
        atom_list = []
        gp_model = self.atoms.calc.gp_model
        while (not self.std_in_bound and atom_count <
               self.max_atoms_added):
            # build gp structure from atoms
            atom_struc = Structure(self.atoms.cell, 
                    ['A']*len(self.atoms.positions), 
                    self.atoms.positions)

            # update gp model
            gp_model.update_db(atom_struc, dft_forces,
                           custom_range=[self.target_atom])
    
            if gp_model.alpha is None:
                gp_model.set_L_alpha()
            else:
                gp_model.update_L_alpha()

            atom_list.append(self.target_atom)
            forces = self.atoms.calc.get_forces_gp(self.atoms) # needed before get_uncertainties
            self.stds = self.atoms.get_uncertainties()

            # write added atom to the log file, 
            # refer to ase.optimize.optimize.Dynamics
            self.observers[0][0].add_atom_info(self.target_atom,
                    self.stds[self.target_atom])
           
            self.is_std_in_bound(atom_list)
            atom_count += 1

        self.train()

    def train(self, monitor=True, skip=False):
        calc = self.atoms.calc
        if (self.dft_count-1) < self.freeze_hyps:
            calc.gp_model.train(monitor)
            self.observers[0][0].write_hyps(calc.gp_model.hyp_labels, 
                            calc.gp_model.hyps, calc.gp_model.like, 
                            calc.gp_model.like_grad)
        else:
            calc.gp_model.set_L_alpha()
    
        # build mapped force field
        if self.use_mapping:
            calc.build_mff(skip)

