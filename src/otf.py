#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
OTF engine

Steven Torrisi, Jon Vandermause, Simon Batzner
"""

import numpy as np
import os
import datetime
from struc import Structure
from math import sqrt
from gp import GaussianProcess
from env import ChemicalEnvironment


# todo strike this; replace with QE input


class OTF(object):
    def __init__(self, qe_input: str, dt: float, number_of_steps: int,
                 kernel: str, cutoff: float):
        """
        On-the-fly learning engine, containing methods to run OTF calculation

        :param qe_input: str, Location of QE input
        :param dt: float, Timestep size
        :param number_of_steps: int, Number of steps
        :param kernel: Type of kernel for GP regression model
        :param cutoff: Cutoff radius for kernel
        """

        self.qe_input = qe_input
        self.dt = dt
        self.Nsteps = number_of_steps
        self.gp = GaussianProcess(kernel)  # Fake_GP(kernel)  #

        positions, species, cell, masses = parse_qe_input(self.qe_input)
        self.structure = Structure(lattice=cell, species=species,
                                   positions=positions, cutoff=cutoff,
                                   mass_dict=masses)

        self.curr_step = 0

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
                self.structure.stds[n][i] = sqrt(var)

    def run_and_train(self):
        """
        Runs QE on the current self.structure config and re-trains  self.GP.
        :return:
        """

        # Run espresso and write out results
        self.write_to_output('=' * 20 + '\n' + 'Calling QE... ')
        forces = self.run_espresso()
        self.write_to_output('Done.\n')

        # Write input positions and force results
        qe_strings = 'Resultant Positions and Forces:\n'
        for n in range(self.structure.nat):
            qe_strings += self.structure.species[n] + ': '
            for i in range(3):
                qe_strings += '%.3f  ' % self.structure.positions[n][i]
            qe_strings += '\t '
            for i in range(3):
                qe_strings += '%.3f  ' % forces[n][i]
            qe_strings += '\n'
        self.write_to_output(qe_strings)

        # Update hyperparameters and write results
        self.write_to_output('Updating database hyperparameters...\n')
        self.gp.update_db(self.structure, forces)
        self.gp.train()
        self.write_to_output('New GP Hyperparameters:\n' +
                             'Signal variance: \t' +
                             str(self.gp.sigma_f) + '\n' +
                             'Length scale: \t\t' +
                             str(self.gp.length_scale) + '\n' +
                             'Noise variance: \t' + str(self.gp.sigma_n) +
                             '\n'
                             )

    def update_positions(self):
        """
        Apply a timestep to self.structure based on current structure's forces.
        """

        # Maintain list of elemental masses in amu to calculate acceleration

        # Load in masses using list comprehension
        masses = [self.structure.mass_dict[spec] for spec in
                  self.structure.species]

        for i, prev_pos in enumerate(self.structure.prev_positions):
            temp_pos = self.structure.positions[i]
            self.structure.positions[i] = 2 * self.structure.positions[i] - \
                                          prev_pos + self.dt ** 2 * \
                                          self.structure.forces[i] / masses[i]

            self.structure.prev_positions[i] = np.array(temp_pos)

    def run_espresso(self):
        """
        Calls quantum espresso from input located at self.qe_input

        :return: List [nparray] List of forces
        """

        self.write_to_output('')
        self.write_qe_input_positions()

        pw_loc = os.environ.get('PWSCF_COMMAND', 'pwscf.x')

        qe_command = '{0} < {1} > {2}'.format(pw_loc, self.qe_input,
                                              'pwscf.out')

        os.system(qe_command)

        return parse_qe_forces('pwscf.out')

    def write_qe_input_positions(self):
        """
        Write the current configuration of the OTF structure to the
        qe input file
        """

        with open(self.qe_input, 'r') as f:
            lines = f.readlines()

        file_pos_index = None
        for i, line in enumerate(lines):
            if 'ATOMIC_POSITIONS' in line:
                file_pos_index = int(i + 1)
        assert file_pos_index is not None, 'Failed to find positions in input'

        for pos_index, line_index in enumerate(
                range(file_pos_index, len(lines))):
            pos_string = ' '.join([self.structure.species[pos_index],
                                   str(self.structure.positions[pos_index][0]),
                                   str(self.structure.positions[pos_index][1]),
                                   str(self.structure.positions[pos_index][
                                           2])])
            lines[line_index] = str(pos_string + '\n')

        with open(self.qe_input, 'w') as f:
            for line in lines:
                f.write(line)

    def write_to_output(self, string: str, output_file: str = 'otf_run.out'):
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
                string += str("%.6f" % self.structure.positions[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str(np.round(self.structure.forces[i][j], 6)) + ' '
            string += '\t'
            for j in range(3):
                string += str('%.6e' % self.structure.stds[i][j]) + ' '
            string += '\n'

            self.write_to_output(string)

    def is_std_in_bound(self):
        """
        Evaluate if the model error are within predefined criteria

        :return: Bool, If model error is within acceptable bounds
        """

        # Some decision making criteria
        return True


def parse_qe_input(qe_input: str):
    """
    Reads the positions, species, and cell in from the qe input file

    :param qe_input: str, Path to PWSCF input file
    :return: List[nparray], List[str], nparray, Positions, species,
                    3x3 Bravais cell
    """
    positions = []
    species = []
    cell = []

    with open(qe_input) as f:
        lines = f.readlines()

    # Find the cell and positions in the output file
    cell_index = None
    positions_index = None
    nat = None
    species_index = None

    for i, line in enumerate(lines):
        if 'CELL_PARAMETERS' in line:
            cell_index = int(i + 1)
        if 'ATOMIC_POSITIONS' in line:
            positions_index = int(i + 1)
        if 'nat' in line:
            nat = int(line.split('=')[1])
        if 'ATOMIC_SPECIES' in line:
            species_index = int(i + 1)

    assert cell_index is not None, 'Failed to find cell in input'
    assert positions_index is not None, 'Failed to find positions in input'
    assert nat is not None, 'Failed to find number of atoms in input'
    assert species_index is not None, 'Failed to find atomic species in input'

    # Load cell
    for i in range(cell_index, cell_index + 3):
        cell_line = lines[i].strip()
        cell.append(np.fromstring(cell_line, sep=' '))
    cell = np.array(cell)

    # Check cell IO
    assert cell != [], 'Cell failed to load'
    assert np.shape(cell) == (3, 3), 'Cell failed to load correctly'

    # Load positions
    for i in range(positions_index, positions_index + nat):
        line_string = lines[i].strip().split()
        species.append(line_string[0])

        pos_string = ' '.join(line_string[1:4])

        positions.append(np.fromstring(pos_string, sep=' '))
    # Check position IO
    assert positions != [], "Positions failed to load"

    # Load masses
    masses = {}
    for i in range(species_index, species_index + len(set(species))):
        # Expects lines of format like: H 1.0 H_pseudo_name.ext
        line = lines[i].strip().split()
        masses[line[0]] = float(line[1])

    return positions, species, cell, masses


def parse_qe_forces(outfile: str):
    """
    Get forces from a pwscf file in Ryd/bohr

    :param outfile: str, Path to pwscf output file
    :return: list[nparray] , List of forces acting on atoms
    """
    forces = []
    total_energy = np.nan

    with open(outfile, 'r') as outf:
        for line in outf:
            if line.lower().startswith('!    total energy'):
                total_energy = float(line.split()[-2]) * 13.605698066

            if line.find('force') != -1 and line.find('atom') != -1:
                line = line.split('force =')[-1]
                line = line.strip()
                line = line.split(' ')
                # print("Parsed line",line,"\n")
                line = [x for x in line if x != '']
                temp_forces = []
                for x in line:
                    temp_forces.append(float(x))
                forces.append(np.array(list(temp_forces)))

    assert total_energy != np.nan, "Quantum ESPRESSO parser failed to read " \
                                   "the file {}. Run failed.".format(outfile)

    return forces


# TODO Currently won't work: needs to be re-done in light of new output
# formatting, but could wait until we finalize an output file format ,
# will be useful for data analysis
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
    os.system('cp tests/test_files/qe_input_2.in pwscf.in')

    otf = OTF('pwscf.in', .1, 10, kernel='two_body',
              cutoff=10)
    otf.run()
    # parse_output('otf_run.out')
    pass
