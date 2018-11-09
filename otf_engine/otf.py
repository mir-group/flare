#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
OTF engine

Steven Torrisi, Jon Vandermause, Simon Batzner
"""

import sys
import numpy as np
import datetime
import time

from typing import List
from copy import deepcopy

from struc import Structure
from gp import GaussianProcess
from env import ChemicalEnvironment
from punchout import punchout
from qe_util import run_espresso, parse_qe_input, timeit, \
    qe_input_to_structure, parse_qe_forces

sys.path.append('../modules')
from analyze_otf import parse_header_information, parse_dft_information, \
                        parse_hyperparameter_information


class OTF(object):
    def __init__(self, qe_input: str, dt: float, number_of_steps: int,
                 kernel: str, bodies: int, cutoff: float, std_tolerance_factor:
                 float = 1, opt_algo : str =  'L-BFGS-B',
                 prev_pos_init: List[np.ndarray] = None,
                 punchout_settings: dict = None):
        """
        On-the-fly learning engine, containing methods to run OTF calculation

        :param qe_input: str, Location of QE input
        :param dt: float, Timestep size, Picoseconds
        :param number_of_steps: int, Number of steps
        :param kernel: Type of kernel for GP regression model
        :param cutoff: Cutoff radius for kernel in angstrom
        :param std_tolerance_factor: Sensitivity factor for calling DFT run,
                if positive, will be relative to signal variance 
                hyperparameter, if negative, will be absolute
        :param punchout_settings: Settings for punchout mode
        """

        self.qe_input = qe_input
        self.dt = dt
        self.Nsteps = number_of_steps
        self.gp = GaussianProcess(kernel, bodies,opt_algorithm=opt_algo)
        self.cutoff = cutoff
        self.std_tolerance = std_tolerance_factor

        self.punchout_settings = punchout_settings

        positions, species, cell, masses = parse_qe_input(self.qe_input)
        self.structure = Structure(lattice=cell, species=species,
                                   positions=positions, cutoff=cutoff,
                                   mass_dict=masses,
                                   prev_positions=prev_pos_init)

        self.curr_step = 0
        self.train_structure = None

        # Instantiate time attributes
        self.start_time = 0
        self.end_time = 0
        self.run_stats = {'dft_calls': 0}

        self.write_header()

    def run(self):
        """
        Performs main loop of OTF engine.
        :return:
        """
        self.start_time = time.time()

        # Bootstrap first training point if DFT is on
        if self.std_tolerance != 0:
            self.run_and_train()

            # If not in punchout mode, take first step with DFT forces
            if not self.punchout_settings:
                self.write_md_config()
                self.update_positions()

        # Main loop
        while self.curr_step < self.Nsteps:

            # Assign forces to atoms in structure
            self.predict_on_structure(time_log=self.run_stats)

            # Check error before proceeding
            std_in_bound, target_atom = self.is_std_in_bound()
            if not std_in_bound:
                self.write_md_config()
                self.run_and_train(target_atom)

                # Re-evaluate forces if in punchout; else use DFT forces
                if self.punchout_settings:
                    continue

            # Print and propagate
            self.write_md_config()
            self.update_positions()

        self.conclude_run()

    @timeit
    def predict_on_structure(self):
        """
        Assign forces to self.structure based on self.gp
        """

        for n in range(self.structure.nat):
            chemenv = ChemicalEnvironment(self.structure, n)
            for i in range(3):
                force, var = self.gp.predict(chemenv, i + 1)
                self.structure.forces[n][i] = float(force)
                self.structure.stds[n][i] = np.sqrt(np.absolute(var))

        self.structure.dft_forces = False

    def run_and_train(self, target_atom: int = None):
        """
        Runs QE on the current self.structure config and re-trains self.GP.
        :return:
        """

        # Run espresso and write out results

        self.write_to_output('=' * 20 + '\n')

        # First run will not have a target atom
        if self.train_structure is None:
            self.write_to_output('Calling initial bootstrap DFT run...')
        else:
            self.write_to_output('Calling DFT due to atom {} at position {} '
                                 'with uncertainties {}...'.format(target_atom,
                                                   self.structure.positions[
                                                       target_atom],
                                                   self.structure.stds[
                                                       target_atom]))

        # If not in punchout mode, run QE on the entire structure
        if self.punchout_settings is None:
            self.train_structure = self.structure
            # Train on all atoms
            train_atoms = list(range(self.train_structure.nat))
        else:
            # On first run, pick a random target atom to punch out around
            if self.train_structure is None:
                target_atom = np.random.randint(0, self.structure.nat)
            self.train_structure = punchout(self.structure,
                                            atom=target_atom,
                                            d=self.punchout_settings['d'])
            # Get the index of the central atom
            train_atoms = [self.train_structure.get_index_from_position((
                np.zeros(3)))]

        forces = run_espresso(self.qe_input, self.train_structure,
                              time_log=self.run_stats, log_name='last_dft')
        self.train_structure.forces = forces

        # Assign forces to structure if not in punchout mode
        if not self.punchout_settings:
            self.structure.dft_forces = True
            self.structure.forces = [np.array(force) for force in forces]
            self.structure.stds = [np.zeros(3)] * self.structure.nat

        self.write_to_output('Done.\n')
        self.run_stats['dft_calls'] += 1

        # Write input positions and force results

        self.write_last_dft_run()

        # Update hyperparameters and write results

        self.write_to_output('Updating database hyperparameters...\n')
        self.gp.update_db(self.train_structure, forces,
                          custom_range=train_atoms)

        self.gp.train(time_log=self.run_stats)

        self.write_hyps()

    def update_positions(self):
        """
        Apply a timestep to self.structure based on current structure's forces.
        """
        # Precompute dt squared for efficiency
        dtdt = self.dt ** 2

        for i, pre_pos in enumerate(self.structure.prev_positions):
            temp_pos = np.copy(self.structure.positions[i])
            mass = self.structure.mass_dict[self.structure.species[i]]
            pos = self.structure.positions[i]
            forces = self.structure.forces[i]

            # Verlet step
            self.structure.positions[i] = 2 * pos - pre_pos + dtdt * forces / \
                                          mass

            self.structure.prev_positions[i] = np.copy(temp_pos)

        self.curr_step += 1

    @staticmethod
    def write_to_output(string: str, output_file: str = 'otf_run.out'):
        """
        Write a string or list of strings to the output file.
        :param string: String to print to output
        :type string: str
        :param output_file: File to write to
        :type output_file: str
        """
        with open(output_file, 'a') as f:
            f.write(string)

    def write_header(self):
        # Create blank output file with time header and structure information
        with open('otf_run.out', 'w') as f:
            f.write(str(datetime.datetime.now()) + '\n')

        headerstring = ''
        headerstring += 'Structure Cutoff Radius: {}\n'.format(
            self.structure.cutoff)
        headerstring += 'Kernel: {}\n'.format(
            self.gp.kernel_name)
        headerstring += '# Hyperparameters: {}\n'.format(
            len(self.gp.hyps))
        headerstring += 'Initial Hyperparameter Optimization Algorithm: {}' \
                        '\n'.format(self.gp.algo)
        headerstring += 'Timestep (ps): {}\n'.format(self.dt)
        headerstring += 'Number of Frames: {}\n'.format(self.Nsteps)
        headerstring += 'Number of Atoms: {}\n'.format(self.structure.nat)
        headerstring += 'System Species: {}\n'.format(set(
            self.structure.species))

        self.write_to_output(headerstring)

    def write_md_config(self):
        """
        Write current step to the output file including positions, forces, and
        force variances

        :return:
        """

        string = ' '

        string += "-------------------- \n"

        # Mark if a frame had DFT forces with an asterisk
        string += "-" + ('*' if self.structure.dft_forces else ' ') + \
                  "Frame: " + str(self.curr_step)

        string += " Simulation Time: "
        string += str(np.round(self.dt * self.curr_step, 6)) + '\n'

        # Construct Header line
        string += 'El \t\t\t  Position (A) \t\t\t\t\t '
        if not self.structure.dft_forces:
            string += 'GP Force (ev/A) '
        else:
            string += 'DFT Force (ev/A) '
        string += '\t\t\t\t\t\t Std. Dev (ev/A) \n'

        # Construct atom-by-atom description

        for i in range(len(self.structure.positions)):
            string += self.structure.species[i] + ' '
            for j in range(3):
                string += str("%.8f" % self.structure.positions[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str("%.8f" % self.structure.forces[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str('%.6e' % self.structure.stds[i][j]) + ' '
            string += '\n'

        self.write_to_output(string)

    def write_last_dft_run(self):

        qe_strings = '~ DFT Call : {}   DFT Time: {} s \n'.format(
            self.run_stats['dft_calls'],
            np.round(self.run_stats['last_dft'], 3))
        qe_strings += 'Lattice: [{}, {}, {}] \n '.format(self.structure.vec1,
                                                         self.structure.vec2,
                                                         self.structure.vec3)
        # np.array2string(self.structure.vec1, separator=','),
        # np.array2string(self.structure.vec2, separator=','),
        # np.array2string(self.structure.vec3, separator=','))

        qe_strings += 'El \t\t\t  Position (A) \t\t\t\t\t DFT Force (ev/A) \n'

        for n in range(self.train_structure.nat):
            qe_strings += self.train_structure.species[n] + ': '
            for i in range(3):
                qe_strings += '%.8f  ' % self.train_structure.positions[n][i]
            qe_strings += '\t '
            for i in range(3):
                qe_strings += '%.8f  ' % self.train_structure.forces[n][i]
            qe_strings += '\n'

        self.write_to_output(qe_strings)
        self.write_to_output('=' * 20 + '\n')

    def write_hyps(self):

        self.write_to_output('New GP Hyperparameters for DFT call {}: '
                             '\n'.format(self.run_stats['dft_calls']))

        for i, label in enumerate(self.gp.hyp_labels):
            self.write_to_output('Hyp{} : {} = {}\n'.format(i, label,
                                                            self.gp.hyps[i]))



    def conclude_run(self):
        """
        Print summary information about the OTF run into the output
        :return:
        """
        self.end_time = time.time()

        footer = '▬' * 20 + '\n'
        footer += 'Run complete. \n'

        dft_time = self.run_stats.get('run_espresso', 0)
        footer += 'Total DFT Time: {} s \n'.format(np.round(dft_time, 3))
        train_time = self.run_stats.get('train', 0)
        footer += 'Total Train Time: {} s \n'.format(np.round(train_time, 3))
        footer += 'Total Prediction Time: {} s \n'.format(
            np.round(self.run_stats['predict_on_structure'], 3))
        footer += 'Total Time: {} m {} s \n'.format(
            int((self.end_time - self.start_time) / 60),
            np.round(self.end_time - self.start_time, 3))

        self.write_to_output(footer)

    def is_std_in_bound(self) -> (bool, int):
        """
        Return (std is in bound, index of highest uncertainty atom)

        :return: Int, -1 f model error is within acceptable bounds
        """
        stds = self.structure.stds
        max_std = np.nanmax(stds)

        # Negative tolerance is a hard user-defined std cutoff; Positive is a
        # coefficient on the noise variance; 0 means no failure condition

        tol = self.std_tolerance

        if tol == 0:
            return True, -1
        constant_cutoff_trip = tol < 0 and max_std >= np.abs(tol)
        rel_cutoff_trip = tol > 0 and max_std >= tol * np.abs(self.gp.hyps[-1])

        # Check if either condition tripped
        if constant_cutoff_trip or rel_cutoff_trip:
            nat = self.structure.nat
            target_atom = np.argmax([np.max(stds[i]) for i in range(nat)])
            return False, target_atom
        else:
            return True, -1

    # TODO Unit test this
    def augment_db_from_pwscf(self, pwscf_in_file: str, pwscf_out_file: str,
                              cutoff_rad: float = None, train: bool = True,
                              origin_atom_only: bool = False,
                              write_hyps: bool = True):
        """
        Augment training database from a pre-existing pwscf output file
        Useful to start off an OTF run non-bootstrap
        :param pwscf_in_file: Input file to get structure from
        :param pwscf_out_file:  Output file to get forces from
        :param cutoff: Cutoff radius for new structure to use in db
        :param train: Train GP after augmenting DB
        :param origin_atom_only: Only use an atom at (0,0,0)
        :return:
        """

        if not cutoff_rad:
            cutoff = self.structure.cutoff
        else:
            cutoff = float(cutoff_rad)

        pw_structure = qe_input_to_structure(qe_input=pwscf_in_file,
                                             cutoff=cutoff)
        forces = parse_qe_forces(pwscf_out_file)

        if origin_atom_only:
            central_atom = [pw_structure.get_index_from_position([np.zeros(
                3)])]
        else:
            central_atom = []

        self.gp.update_db(pw_structure, forces, central_atom)

        if train:
            self.gp.train(time_log=self.run_stats)
            if write_hyps:
                self.write_hyps()

    def augment_db_from_otf_run(self, otf_run_output: str, train: bool =
        True, origin_atom_only: bool = False, write_hyps: bool = True,
            use_prev_hyps: bool = True):
        """
        Parses OTF run and augments gp database with QE runs
        :param origin_atom_only:
        :param train:
        :param otf_run_output:
        :return:
        """

        header_info = parse_header_information(otf_run_output)

        cutoff = header_info['cutoff']

        lattices_set, species_set, positions_set, forces_set = \
            parse_dft_information(outfile=otf_run_output)

        # Loop through
        for i in range(len(positions_set)):
            cur_spec = species_set[i]
            cur_pos = positions_set[i]
            cur_forces = forces_set[i]
            cur_lattice = lattices_set[i]

            curr_struc = Structure(positions=cur_pos, species=cur_spec,
                                   lattice=cur_lattice, mass_dict={},
                                   cutoff=cutoff)

            if origin_atom_only:
                central_atom = [curr_struc.get_index_from_position([np.zeros(
                    3)])]
            else:
                central_atom = []

            self.gp.update_db(struc=curr_struc, forces=cur_forces,
                              custom_range=central_atom)
        if use_prev_hyps:
            last_hyps = parse_hyperparameter_information(otf_run_output)[-1]
            self.gp.hyps = np.array(last_hyps)

        if train:
            self.gp.train(time_log=self.run_stats)

        if write_hyps:
            self.write_hyps()


if __name__ == '__main__':
    import os

    os.system('cp qe_input_3.in pwscf.in')

    otf = OTF('pwscf.in', .0001, 500, kernel='n_body_sc', bodies=2,
              cutoff=5, std_tolerance_factor=.5,opt_algo='nelder-mead')
    otf.run()
    # otf.run_and_train(target_atom=0)
    # parse_output('otf_run.out')
    pass
