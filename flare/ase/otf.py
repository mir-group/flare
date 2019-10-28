import os
import sys
from copy import deepcopy

from flare.struc import Structure
from flare.util import is_std_in_bound

import numpy as np
from ase.md.md import MolecularDynamics
from ase import units


class OTF(MolecularDynamics):
    """output_name => logfile
    npool: added
    prev_pos_init: relaunch mode not implemented
    skip: not implemented
    l_bound: mgp update l_bound, not implemented

    dft calculator is set outside of the otf module, and input as dft_calc, 
    so that different calculators can be used"""

    def __init__(self, atoms, timestep, trajectory=None, 
            # on-the-fly parameters
            dft_calc=None, dft_count=None, std_tolerance_factor: float=1, 
            prev_pos_init: np.ndarray=None, par:bool=False, skip: int=0, 
            init_atoms: list=[], calculate_energy=False, max_atoms_added=1, 
            freeze_hyps=1, no_cpus=1,  
            # mgp parameters
            use_mapping: bool=False, non_mapping_steps: list=[],
            l_bound: float=None, two_d: bool=False):

        MolecularDynamics.__init__(self, atoms, timestep, trajectory)
                                   
        self.std_tolerance = std_tolerance_factor
        self.noa = len(atoms.positions)
        self.max_atoms_added = max_atoms_added
        self.freeze_hyps = freeze_hyps
        self.dft_calc = dft_calc
        if dft_count is None:
            self.dft_count = 0
        else:
            self.dft_count = dft_count

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

    def otf_run(self, steps):
        """Perform a number of time steps."""
        # initialize gp by a dft calculation
        if not self.atoms.calc.gp_model.training_data:
            self.dft_count = 0
            self.std_in_bound = False
            self.target_atom = 0
            self.stds = []
            dft_forces = self.call_DFT()
            f = dft_forces
   
            # update gp model
            atom_struc = Structure(np.array(self.atoms.cell), 
                                   self.atoms.get_atomic_numbers(), 
                                   self.atoms.positions)
            self.atoms.calc.gp_model.update_db(atom_struc, dft_forces,
                           custom_range=self.init_atoms)

            # train calculator
            self.train()
            print('mgp model:', self.atoms.calc.mgp_model)

        if self.md_engine == 'NPT':
            if not self.initialized:
                self.initialize()
            else:
                if self.have_the_atoms_been_changed():
                    raise NotImplementedError(
                        "You have modified the atoms since the last timestep.")

        for i in range(steps):
            print('step:', i)
            if self.md_engine == 'NPT':
                self.step()
            else:
                f = self.step(f)
            self.nsteps += 1
            self.stds = self.atoms.get_uncertainties()

            # figure out if std above the threshold
            self.call_observers() 
            curr_struc = Structure.from_ase_atoms(self.atoms)
            curr_struc.stds = self.stds
            noise = self.atoms.calc.gp_model.hyps[-1]
            self.std_in_bound, self.target_atoms = is_std_in_bound(\
                    noise, self.std_tolerance, curr_struc, self.max_atoms_added)

            #self.is_std_in_bound([])

            if not self.std_in_bound:
                # call dft/eam
                print('calling dft')
                dft_forces = self.call_DFT()

                # update gp
                print('updating gp')
                self.update_GP(dft_forces)

        self.observers[0][0].run_complete()

    
    def call_DFT(self):
        prev_calc = self.atoms.calc
        calc = deepcopy(self.dft_calc)
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
            atom_struc = Structure(np.array(self.atoms.cell), 
                    self.atoms.get_atomic_numbers(), 
                    self.atoms.positions)

            # update gp model
            gp_model.update_db(atom_struc, dft_forces,
                           custom_range=[self.target_atom])
    
            if gp_model.alpha is None:
                gp_model.set_L_alpha()
            else:
                gp_model.update_L_alpha()

            atom_list.append(self.target_atom)
            # force calculation needed before get_uncertainties
            forces = self.atoms.calc.get_forces_gp(self.atoms) 
            self.stds = self.atoms.get_uncertainties()

            # write added atom to the log file, 
            # refer to ase.optimize.optimize.Dynamics
            self.observers[0][0].add_atom_info(self.target_atom,
                    self.stds[self.target_atom])
           
            #self.is_std_in_bound(atom_list)
            atom_count += 1

        self.train()

    def train(self, output=None, skip=False):
        calc = self.atoms.calc
        if (self.dft_count-1) < self.freeze_hyps:
            #TODO: add other args to train()
            calc.gp_model.train(output=output)
            self.observers[0][0].write_hyps(calc.gp_model.hyp_labels, 
                            calc.gp_model.hyps, calc.gp_model.likelihood, 
                            calc.gp_model.likelihood_gradient)
        else:
            calc.gp_model.set_L_alpha()

        # build mgp
        if self.use_mapping:
            if self.get_time() in self.non_mapping_steps:
                skip = True

            calc.build_mgp(skip)

