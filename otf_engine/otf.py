import sys
import numpy as np
import datetime
import time
from typing import List
import struc
import gp
import env
import qe_util
import multiprocessing as mp
import concurrent.futures
import md
import output


class OTF(object):
    def __init__(self, qe_input: str, dt: float, number_of_steps: int,
                 gp: gp.GaussianProcess, pw_loc: str,
                 std_tolerance_factor: float = 1,
                 prev_pos_init: np.ndarray=None, par: bool=False,
                 skip: int=0, freeze_hyps=False, init_atoms: List[int]=None,
                 calculate_energy=False, output_name='otf_run.out'):

        self.qe_input = qe_input
        self.dt = dt
        self.number_of_steps = number_of_steps
        self.gp = gp
        self.pw_loc = pw_loc
        self.std_tolerance = std_tolerance_factor
        self.skip = skip
        self.freeze_hyps = freeze_hyps

        # parse input file
        positions, species, cell, masses = \
            qe_util.parse_qe_input(self.qe_input)

        self.structure = struc.Structure(cell=cell, species=species,
                                         positions=positions,
                                         mass_dict=masses,
                                         prev_positions=prev_pos_init)

        self.noa = self.structure.positions.shape[0]
        self.atom_list = list(range(self.noa))
        self.curr_step = 0

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

        self.output_name = output_name

    def run(self):
        output.write_header(self.gp.cutoffs, self.gp.kernel_name, self.gp.hyps,
                            self.gp.algo, self.dt, self.number_of_steps,
                            self.structure, self.output_name)
        counter = 0
        self.start_time = time.time()

        while self.curr_step < self.number_of_steps:
            # run DFT and train initial model if first step and DFT is on
            if self.curr_step == 0 and self.std_tolerance != 0:
                self.run_dft()
                self.update_gp(self.init_atoms)
                new_pos = md.update_positions(self.dt, self.noa,
                                              self.structure)
                self.update_temperature(new_pos)
                self.record_state()
                self.update_positions(new_pos)

            # otherwise, try predicting with GP model
            else:
                self.pred_func()
                new_pos = md.update_positions(self.dt, self.noa,
                                              self.structure)

                # TODO: add option to repeat std check until below threshold
                std_in_bound, target_atom = self.is_std_in_bound()
                if not std_in_bound:
                    self.dft_count += 1

                    # record GP forces
                    self.update_temperature(new_pos)
                    self.record_state()

                    # record DFT forces
                    self.run_dft()
                    self.update_gp([target_atom])
                    new_pos = md.update_positions(self.dt, self.noa,
                                                  self.structure)
                    self.update_temperature(new_pos)
                    self.record_state()

                # write gp forces only when counter equals skip
                if counter >= self.skip and self.structure.dft_forces is False:
                    self.update_temperature(new_pos)
                    self.record_state()
                    counter = 0

                counter += 1
                self.update_positions(new_pos)

            # update step variable
            self.curr_step += 1

        output.conclude_run(self.output_name)

    def predict_on_atom(self, atom):
        chemenv = env.AtomicEnvironment(self.structure, atom, self.gp.cutoffs)
        comps = []
        stds = []
        # predict force components and standard deviations
        for i in range(3):
            force, var = self.gp.predict(chemenv, i+1)
            comps.append(float(force))
            stds.append(np.sqrt(np.absolute(var)))

        return comps, stds

    def predict_on_atom_en(self, atom):
        chemenv = env.AtomicEnvironment(self.structure, atom, self.gp.cutoffs)
        comps = []
        stds = []
        # predict force components and standard deviations
        for i in range(3):
            force, var = self.gp.predict(chemenv, i+1)
            comps.append(float(force))
            stds.append(np.sqrt(np.absolute(var)))

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
        self.structure.dft_forces = False

    def predict_on_structure_par_en(self):
        n = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(self.predict_on_atom_en, self.atom_list):
                for i in range(3):
                    self.structure.forces[n][i] = res[0][i]
                    self.structure.stds[n][i] = res[1][i]
                self.local_energies[n] = res[2]
                n += 1
        self.structure.dft_forces = False

    def predict_on_structure(self):
        for n in range(self.structure.nat):
            chemenv = env.AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            for i in range(3):
                force, var = self.gp.predict(chemenv, i + 1)
                self.structure.forces[n][i] = float(force)
                self.structure.stds[n][i] = np.sqrt(np.absolute(var))

        self.structure.dft_forces = False

    def predict_on_structure_en(self):
        for n in range(self.structure.nat):
            chemenv = env.AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            for i in range(3):
                force, var = self.gp.predict(chemenv, i + 1)
                self.structure.forces[n][i] = float(force)
                self.structure.stds[n][i] = np.sqrt(np.absolute(var))
            self.local_energies[n] = self.gp.predict_local_energy(chemenv)

        self.structure.dft_forces = False

    def run_dft(self):
        output.write_to_output('=' * 20 + '\n', self.output_name)
        output.write_to_output('Calling Quantum Espresso. ', self.output_name)

        # calculate DFT forces
        forces = qe_util.run_espresso(self.qe_input, self.structure,
                                      self.pw_loc)
        self.structure.forces = forces
        self.structure.stds = [np.zeros(3) for n in range(self.structure.nat)]
        self.structure.dft_forces = True

        # write wall time of DFT calculation
        output.write_to_output('QE run complete.\n', self.output_name)
        time_curr = time.time() - self.start_time
        output.write_to_output('wall time from start: %.2f s \n' % time_curr,
                               self.output_name)

    def update_gp(self, train_atoms):
        output.write_to_output('Training atoms: {}.\n'.format(train_atoms),
                               self.output_name)

        # update gp model
        self.gp.update_db(self.structure, self.structure.forces,
                          custom_range=train_atoms)
        if self.freeze_hyps is False:
            self.gp.train()
        else:
            self.gp.set_L_alpha()
        output.write_hyps(self.gp.hyp_labels, self.gp.hyps, self.start_time,
                          self.dft_count, self.output_name)

    def is_std_in_bound(self) -> (bool, int):
        stds = self.structure.stds
        max_std = np.nanmax(stds)

        # negative tolerance is a hard user-defined std cutoff
        # positive is a multiple of the noise std
        # 0 means no failure condition (DFT is off)
        tol = self.std_tolerance

        if tol == 0:
            return True, -1
        constant_cutoff_trip = tol < 0 and max_std >= np.abs(tol)
        rel_cutoff_trip = tol > 0 and max_std >= tol * np.abs(self.gp.hyps[-1])

        if constant_cutoff_trip or rel_cutoff_trip:
            nat = self.structure.nat
            target_atom = np.argmax([np.max(stds[i]) for i in range(nat)])
            return False, target_atom
        else:
            return True, -1

    def update_positions(self, new_pos):
        self.structure.prev_positions = self.structure.positions
        self.structure.positions = new_pos
        self.structure.wrap_positions()

    def update_temperature(self, new_pos):
        KE, temperature = \
                md.calculate_temperature(new_pos, self.structure, self.dt,
                                         self.noa)
        self.KE = KE
        self.temperature = temperature

    def record_state(self):
        output.write_md_config(self.dt, self.curr_step, self.structure,
                               self.temperature, self.KE,
                               self.local_energies, self.start_time,
                               self.output_name)
