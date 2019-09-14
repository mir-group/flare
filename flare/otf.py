import sys
import numpy as np
import datetime
import time
from typing import List
import copy
import multiprocessing as mp
import subprocess
import concurrent.futures
from flare import struc, gp, env, qe_util, md
from flare.output import Output


class OTF(object):
    def __init__(self, qe_input: str, dt: float, number_of_steps: int,
                 gp: gp.GaussianProcess, pw_loc: str,
                 std_tolerance_factor: float = 1,
                 prev_pos_init: np.ndarray=None, par: bool=False,
                 skip: int=0, init_atoms: List[int]=None,
                 calculate_energy=False, output_name='otf_run',
                 max_atoms_added=1, freeze_hyps=10,
                 rescale_steps=[], rescale_temps=[],
                 no_cpus=1):

        self.qe_input = qe_input
        self.dt = dt
        self.number_of_steps = number_of_steps
        self.gp = gp
        self.pw_loc = pw_loc
        self.std_tolerance = std_tolerance_factor
        self.skip = skip
        self.dft_step = True
        self.freeze_hyps = freeze_hyps

        # parse input file
        positions, species, cell, masses = \
            qe_util.parse_qe_input(self.qe_input)

        _, coded_species = struc.get_unique_species(species)

        self.structure = struc.Structure(cell=cell, species=coded_species,
                                         positions=positions,
                                         mass_dict=masses,
                                         prev_positions=prev_pos_init,
                                         species_labels=species)

        self.noa = self.structure.positions.shape[0]
        self.atom_list = list(range(self.noa))
        self.curr_step = 0

        self.max_atoms_added = max_atoms_added

        # initialize local energies
        if calculate_energy:
            self.local_energies = np.zeros(self.noa)
        else:
            self.local_energies = None

        # set atom list for initial dft run
        if init_atoms is None:
            self.init_atoms = [int(n) for n in range(self.noa)]
        else:
            self.init_atoms = init_atoms

        self.dft_count = 0

        # set pred function
        if not par and not calculate_energy:
            self.pred_func = self.predict_on_structure
        elif par and not calculate_energy:
            self.pred_func = self.predict_on_structure_par
        elif not par and calculate_energy:
            self.pred_func = self.predict_on_structure_en
        elif par and calculate_energy:
            self.pred_func = self.predict_on_structure_par_en
        self.par = par

        # set rescale attributes
        self.rescale_steps = rescale_steps
        self.rescale_temps = rescale_temps

        self.output = Output(output_name)

        # set number of cpus for qe runs
        self.no_cpus = no_cpus

    def run(self):
        self.output.write_header(self.gp.cutoffs, self.gp.kernel_name,
                                 self.gp.hyps, self.gp.algo,
                                 self.dt, self.number_of_steps,
                                 self.structure,
                                 self.std_tolerance)
        counter = 0
        self.start_time = time.time()

        while self.curr_step < self.number_of_steps:
            print('curr_step:', self.curr_step)
            # run DFT and train initial model if first step and DFT is on
            if self.curr_step == 0 and self.std_tolerance != 0:
                # call dft and update positions
                self.run_dft()
                dft_frcs = copy.deepcopy(self.structure.forces)
                new_pos = md.update_positions(self.dt, self.noa,
                                              self.structure)
                self.update_temperature(new_pos)
                self.record_state()

                # make initial gp model and predict forces
                self.update_gp(self.init_atoms, dft_frcs)
                if (self.dft_count-1) < self.freeze_hyps:
                    self.train_gp()

            # after step 1, try predicting with GP model
            else:
                self.pred_func()
                self.dft_step = False
                new_pos = md.update_positions(self.dt, self.noa,
                                              self.structure)

                # get max uncertainty atoms
                std_in_bound, target_atoms = self.is_std_in_bound()

                if not std_in_bound:
                    # record GP forces
                    self.update_temperature(new_pos)
                    self.record_state()
                    gp_frcs = copy.deepcopy(self.structure.forces)

                    # run DFT and record forces
                    self.dft_step = True
                    self.run_dft()
                    dft_frcs = copy.deepcopy(self.structure.forces)
                    new_pos = md.update_positions(self.dt, self.noa,
                                                  self.structure)
                    self.update_temperature(new_pos)
                    self.record_state()

                    # compute mae and write to output
                    mae = np.mean(np.abs(gp_frcs - dft_frcs))
                    mac = np.mean(np.abs(dft_frcs))

                    self.output.write_to_log('\nmean absolute error:'
                                             ' %.4f eV/A \n' % mae)
                    self.output.write_to_log('mean absolute dft component:'
                                           ' %.4f eV/A \n' % mac)

                    # add max uncertainty atoms to training set
                    self.update_gp(target_atoms, dft_frcs)
                    if (self.dft_count-1) < self.freeze_hyps:
                        self.train_gp()

            # write gp forces
            if counter >= self.skip and not self.dft_step:
                self.update_temperature(new_pos)
                self.record_state()
                counter = 0

            counter += 1
            self.update_positions(new_pos)
            self.curr_step += 1

        self.output.conclude_run()

    def predict_on_atom(self, atom):
        chemenv = env.AtomicEnvironment(self.structure, atom, self.gp.cutoffs)
        comps = []
        stds = []
        # predict force components and standard deviations
        for i in range(3):
            force, var = self.gp.predict(chemenv, i+1)
            comps.append(float(force))
            stds.append(np.sqrt(np.abs(var)))

        return comps, stds

    def predict_on_atom_en(self, atom):
        chemenv = env.AtomicEnvironment(self.structure, atom, self.gp.cutoffs)
        comps = []
        stds = []
        # predict force components and standard deviations
        for i in range(3):
            force, var = self.gp.predict(chemenv, i+1)
            comps.append(float(force))
            stds.append(np.sqrt(np.abs(var)))

        # predict local energy
        local_energy = self.gp.predict_local_energy(chemenv)
        return comps, stds, local_energy

    def predict_on_structure_par(self):
        n = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(self.predict_on_atom, self.atom_list):
                for i in range(3):
                    self.structure.forces[n][i] = res[0][i]
                    self.structure.stds[n][i] = res[1][i]
                n += 1

    def predict_on_structure_par_en(self):
        n = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(self.predict_on_atom_en, self.atom_list):
                for i in range(3):
                    self.structure.forces[n][i] = res[0][i]
                    self.structure.stds[n][i] = res[1][i]
                self.local_energies[n] = res[2]
                n += 1

    def predict_on_structure(self):
        for n in range(self.structure.nat):
            chemenv = env.AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            for i in range(3):
                force, var = self.gp.predict(chemenv, i + 1)
                self.structure.forces[n][i] = float(force)
                self.structure.stds[n][i] = np.sqrt(np.abs(var))

    def predict_on_structure_en(self):
        for n in range(self.structure.nat):
            chemenv = env.AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            for i in range(3):
                force, var = self.gp.predict(chemenv, i + 1)
                self.structure.forces[n][i] = float(force)
                self.structure.stds[n][i] = np.sqrt(np.abs(var))
            self.local_energies[n] = self.gp.predict_local_energy(chemenv)

    def run_dft(self):
        self.output.write_to_log('\nCalling Quantum Espresso...\n')

        # calculate DFT forces
        forces = qe_util.run_espresso_par(self.qe_input, self.structure,
                                          self.pw_loc, self.no_cpus)
        self.structure.forces = forces

        # write wall time of DFT calculation
        self.dft_count += 1
        self.output.write_to_log('QE run complete.\n')
        time_curr = time.time() - self.start_time
        self.output.write_to_log('number of DFT calls: %i \n' % self.dft_count)
        self.output.write_to_log('wall time from start: %.2f s \n' % time_curr)

    def update_gp(self, train_atoms, dft_frcs):
        self.output.write_to_log('\nAdding atom {} to the training set.\n'
                                 .format(train_atoms))
        self.output.write_to_log('Uncertainty: {}.\n'
                                 .format(self.structure.stds[train_atoms[0]]))

        # update gp model
        self.gp.update_db(self.structure, dft_frcs,
                          custom_range=train_atoms)

        self.gp.set_L_alpha()

        # if self.curr_step == 0:
        #     self.gp.set_L_alpha()
        # else:
        #     self.gp.update_L_alpha()

    def train_gp(self):
        self.gp.train(self.output)
        self.output.write_hyps(self.gp.hyp_labels, self.gp.hyps,
                               self.start_time,
                               self.gp.like, self.gp.like_grad)

    def is_std_in_bound(self):
        # set uncertainty threshold
        if self.std_tolerance == 0:
            return True, -1
        elif self.std_tolerance > 0:
            threshold = self.std_tolerance * np.abs(self.gp.hyps[-1])
        else:
            threshold = np.abs(self.std_tolerance)

        # sort max stds
        max_stds = np.zeros((self.noa))
        for atom, std in enumerate(self.structure.stds):
            max_stds[atom] = np.max(std)
        stds_sorted = np.argsort(max_stds)
        target_atoms = list(stds_sorted[-self.max_atoms_added:])

        # if above threshold, return atom
        if max_stds[stds_sorted[-1]] > threshold:
            return False, target_atoms
        else:
            return True, [-1]

    def update_positions(self, new_pos):
        if self.curr_step in self.rescale_steps:
            rescale_ind = self.rescale_steps.index(self.curr_step)
            temp_fac = self.rescale_temps[rescale_ind] / self.temperature
            vel_fac = np.sqrt(temp_fac)
            self.structure.prev_positions = \
                new_pos - self.velocities * self.dt * vel_fac
        else:
            self.structure.prev_positions = self.structure.positions
        self.structure.positions = new_pos
        self.structure.wrap_positions()

    def update_temperature(self, new_pos):
        KE, temperature, velocities = \
                md.calculate_temperature(new_pos, self.structure, self.dt,
                                         self.noa)
        self.KE = KE
        self.temperature = temperature
        self.velocities = velocities

    def record_state(self):
        self.output.write_md_config(self.dt, self.curr_step, self.structure,
                                    self.temperature, self.KE,
                                    self.local_energies, self.start_time,
                                    self.dft_step,
                                    self.velocities)
        self.output.write_xyz_config(self.curr_step, self.structure,
                                     self.dft_step)
