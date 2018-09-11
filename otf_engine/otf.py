#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
OTF engine

Steven Torrisi, Jon Vandermause, Simon Batzner
"""

import numpy as np
import datetime
from struc import Structure
from gp import GaussianProcess
from env import ChemicalEnvironment
from punchout import punchout
from qe_util import run_espresso, parse_qe_input


class OTF(object):
    def __init__(self, qe_input: str, dt: float, number_of_steps: int,
                 kernel: str, cutoff: float, punchout_d: float = None):
        """
        On-the-fly learning engine, containing methods to run OTF calculation

        :param qe_input: str, Location of QE input
        :param dt: float, Timestep size
        :param number_of_steps: int, Number of steps
        :param kernel: Type of kernel for GP regression model
        :param cutoff: Cutoff radius for kernel in angstrom
        :param punchout_d: Box distance around a high-uncertainty atom to
                         punch out
        """

        self.qe_input = qe_input
        self.dt = dt
        self.Nsteps = number_of_steps
        self.gp = GaussianProcess(kernel)
        self.cutoff = cutoff
        self.punchout_d = punchout_d

        positions, species, cell, masses = parse_qe_input(self.qe_input)
        self.structure = Structure(lattice=cell, species=species,
                                   positions=positions, cutoff=cutoff,
                                   mass_dict=masses)

        self.curr_step = 0
        self.train_structure = None

        # Create blank output file with time header and structure information
        with open('otf_run.out', 'w') as f:
            f.write(str(datetime.datetime.now()) + '\n')
            # f.write(str(self.structure.species)+'\n')

    def run(self):
        """
        Performs main loop of OTF engine.
        :return:
        """

        # Bootstrap first training point
        self.run_and_train()

        # Main loop
        while self.curr_step < self.Nsteps:

            self.predict_on_structure()

            if not self.is_std_in_bound():
                self.run_and_train()
                continue

            self.update_positions()
            self.write_config()
            self.curr_step += 1

    def predict_on_structure(self):
        """
        Assign forces to self.structure based on self.gp
        """

        for n in range(self.structure.nat):
            chemenv = ChemicalEnvironment(self.structure, n)
            for i in range(3):
                force, var = self.gp.predict(chemenv, i + 1)
                self.structure.forces[n][i] = float(force)
                self.structure.stds[n][i] = np.sqrt(var)

    def run_and_train(self):
        """
        Runs QE on the current self.structure config and re-trains  self.GP.
        :return:
        """

        # Run espresso and write out results
        self.write_to_output('=' * 20 + '\n' + 'Calling QE... ')

        # If not in punchout mode, run QE on the entire structure
        if self.punchout_d is None:
            self.train_structure = self.structure
        # If first run, pick a random atom to punch out a structure around
        elif self.train_structure is None:
            target_atom = np.random.randint(0, len(self.structure.positions))
            self.train_structure = punchout(self.structure, target_atom,
                                            d=self.punchout_d)

        forces = run_espresso(self.qe_input, self.train_structure)

        self.write_to_output('Done.\n')

        # Write input positions and force results
        qe_strings = 'Resultant Positions and Forces:\n'
        for n in range(self.train_structure.nat):
            qe_strings += self.train_structure.species[n] + ': '
            for i in range(3):
                qe_strings += '%.8f  ' % self.train_structure.positions[n][i]
            qe_strings += '\t '
            for i in range(3):
                qe_strings += '%.8f  ' % forces[n][i]
            qe_strings += '\n'
        self.write_to_output(qe_strings)

        # Update hyperparameters and write results
        self.write_to_output('Updating database hyperparameters...\n')
        self.gp.update_db(self.train_structure, forces)
        self.gp.train()
        self.write_to_output('New GP Hyperparameters:\n' +
                             'Signal std: \t' +
                             str(self.gp.sigma_f) + '\n' +
                             'Length scale: \t\t' +
                             str(self.gp.length_scale) + '\n' +
                             'Noise std: \t' + str(self.gp.sigma_n) +
                             '\n'
                             )

    def update_positions(self):
        """
        Apply a timestep to self.structure based on current structure's forces.
        """

        # Precompute dt squared for efficiency
        dtdt = self.dt ** 2

        for i, pre_pos in enumerate(self.structure.prev_positions):
            temp_pos = np.array(self.structure.positions[i])
            mass = self.structure.mass_dict[self.structure.species[i]]
            pos = self.structure.positions[i]
            forces = self.structure.forces[i]

            self.structure.positions[i] = 2 * pos - pre_pos + dtdt * forces / \
                mass

            self.structure.prev_positions[i] = np.array(temp_pos)

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

    def write_config(self):
        """
        Write current step to the output file including positions, forces, and
        force variances
        """

        string = ''

        string += "-------------------- \n"

        string += "- Frame " + str(self.curr_step)
        string += " Sim. Time "
        string += str(np.round(self.dt * self.curr_step, 3)) + '\n'

        string += 'El \t\t\t Position \t\t\t\t\t Force \t\t\t\t\t\t\t ' \
                  'Std. Dev \n'

        for i in range(len(self.structure.positions)):
            string += self.structure.species[i] + ' '
            for j in range(3):
                string += str("%.8f"%self.structure.positions[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str(np.round(self.structure.forces[i][j], 8)) + ' '
            string += '\t'
            for j in range(3):
                string += str('%.6e' % self.structure.stds[i][j]) + ' '
            string += '\n'

        self.write_to_output(string)

    # TODO change this to use the signal variance
    def is_std_in_bound(self):
        """
        Evaluate if the model error are within predefined criteria

        :return: Bool, If model error is within acceptable bounds
        """

        # Some decision making criteria

        # SKETCH OF EVENTUAL CODE:
        """
        high_error_atom = -1 
        
        if self.punchout_d is not None:
            self.train_structure= punchout(self.structure,atom= 
            high_error_atom, d= self.punchout_d )
            return False
        """

        return True


# TODO Currently won't work: needs to be re-done when we finalize our output
#  formatting
def parse_otf_output(outfile: str):
    """
    Parse the output of a otf run for analysis
    :param outfile: str, Path to file
    :return: dict{int:value,'species':list}, Dict of positions, forces,
    vars indexed by frame and of species
    """

    results = {}
    with open(outfile, 'r') as f:
        lines = f.readlines()

    frame_indices = [lines.index(line) for line in lines if line[0] == '-']
    n_atoms = frame_indices[1] - frame_indices[0] - 2

    for frame_number, index in enumerate(frame_indices):
        positions = []
        forces = []
        stds = []
        species = []

        for at_index in range(n_atoms):
            at_line = lines[index + at_index + 1].strip().split(',')
            species.append(at_line[0])
            positions.append(
                np.fromstring(','.join((at_line[1], at_line[2], at_line[3])),
                              sep=','))
            forces.append(
                np.fromstring(','.join((at_line[4], at_line[5], at_line[6])),
                              sep=','))
            stds.append(
                np.fromstring(','.join((at_line[7], at_line[8], at_line[9])),
                              sep=','))

            results[frame_number] = {'positions': positions,
                                     'forces': forces,
                                     'vars': vars}

        results['species'] = species
        print(results)


if __name__ == '__main__':

    import os
    os.system('cp qe_input_1.in pwscf.in')

    otf = OTF('pwscf.in', .001, 10, kernel='two_body',
              cutoff=10, punchout_d=None)
    otf.run()
    # parse_output('otf_run.out')
    pass
