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

from gp import GaussianProcess
from env import ChemicalEnvironment


class Fake_GP(GaussianProcess):
    """
    Fake GP that returns random forces and variances.
    Hope we do better than this!
    """

    def __init__(self, kernel):
        super(GaussianProcess, self).__init__()

        pass

    def train(self):
        """
        Neuters the train method of GaussianProcess
        """
        pass

    def update_db(self, structure, forces):
        """
        Neuters the update_db method of GaussianProcess
        """
        pass

    def predict(self, structure, _):
        """
        Substitutes in the predict method of GaussianProcess
        """
        structure.forces = [np.random.randn(3) for n in range(structure.nat)]
        structure.stds = [np.random.randn(3) for n in range(structure.nat)]

        return structure

    def predict_on_structure(self, structure):
        """
        Substitutes in the predict_on_structure method of GaussianProcess
        """

        structure.forces = [np.random.randn(3) for n in structure.positions]
        structure.stds = [np.random.randn(3) for n in structure.positions]

        return structure


# todo strike this; replace with QE input
mass_dict = {'H': 1.0,
             'Si': 28.0855,
             'C': 12.0107,
             'Au': 196.96657,
             'Ag': 107.8682,
             'Pd': 106.42}


class OTF(object):
    def __init__(self, qe_input, dt, number_of_steps, kernel, cutoff):
        """
        On-the-fly learning engine, containing methods to run OTF calculation

        :param qe_input: str, Location of QE input
        :param dt: float, Timestep size
        :param number_of_steps: int, Number of steps
        :param kernel: GP, Regression model object
        """

        self.qe_input = qe_input
        self.dt = dt
        self.Nsteps = number_of_steps
        self.gp = Fake_GP(kernel)  # GaussianProcess(kernel)

        positions, species, cell, = parse_qe_input(self.qe_input)
        self.structure = Structure(lattice=cell, species=species,
                                   positions=positions, cutoff=cutoff,
                                   mass_dict=mass_dict)

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

            self.structure = self.predict_on_structure(self.structure)

            if not self.is_std_in_bound():
                self.run_and_train()
                continue

            self.update_positions()
            self.write_step()
            self.curr_step += 1

    def predict_on_structure(self, structure):
        """
        :param structure: Structure to have forces predicted upon by GP
        :type structure: Structure
        :return: Structure with forces and stds
        :rtype: Structure
        """

        for n in range(structure.nat):
            chemenv = ChemicalEnvironment(structure, n)
            for i in range(3):
                structure.forces[n][i], structure.stds[n][i] = \
                    self.gp.predict(chemenv, i)

        return structure

    def run_and_train(self):
        """
        Runs QE on the current self.structure config and re-trains  self.GP.
        :return:
        """
        forces = self.run_espresso()
        self.gp.update_db(self.structure, forces)
        self.gp.train()

    def update_positions(self):
        """
        Apply a timestep to self.structure based on current structure's forces.
        """

        # Maintain list of elemental masses in amu to calculate acceleration

        for i, prev_pos in enumerate(self.structure.prev_positions):
            temp_pos = self.structure.positions[i]
            self.structure.positions[i] = 2 * self.structure.positions[i] - \
                                          prev_pos + self.dt ** 2 * \
                                          self.structure.forces[i] / \
                                          mass_dict[self.structure.species[i]]
            self.structure.prev_positions[i] = np.array(temp_pos)

    def run_espresso(self):
        """
        Calls quantum espresso from input located at self.qe_input

        :return: List [nparray] List of forces
        """

        self.write_qe_input_positions()

        pw_loc = os.environ.get('PWSCF_COMMAND', 'pwscf.x')

        qe_command = '{0} < {1} > {2}'.format(pw_loc, self.qe_input,
                                              'pwscf.out')

        os.system(qe_command)

        return parse_qe_forces('pwscf.out')

    def write_qe_input_positions(self):
        """
        Write the current configuration of the OTF structure to the qe input file
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

    def write_step(self):
        """
        Write current step to the output file including positions, forces, and force variances
        """

        with open('otf_run.out', 'a') as f:

            f.write("====================== \n")
            f.write(
                ' '.join(["-", "Frame", str(self.curr_step), 'Time',
                          str(np.round(self.dt * self.curr_step, 3)), '\n']))

            for i in range(len(self.structure.positions)):
                to_write = [self.structure.species[i]]
                for j in range(3):
                    to_write.append(
                        str(np.round(self.structure.positions[i][j], 3)))
                for j in range(3):
                    to_write.append(
                        str(np.round(self.structure.forces[i][j], 3)))
                for j in range(3):
                    to_write.append(
                        str(np.round(self.structure.stds[i][j], 3)))
                curr_line = ','.join(to_write)
                curr_line += '\n'

                f.write(curr_line)

    def is_std_in_bound(self):
        """
        Evaluate if the model error are within predefined criteria

        :return: Bool, If model error is within acceptable bounds
        """

        # Some decision making criteria
        return True


def parse_qe_input(qe_input):
    """
    Reads the positions, species, and cell in from the

    :param qe_input: str, Path to PWSCF input file
    :return: List[nparray], List[str], nparray, Positions, species, 3x3 Bravais cell
    """
    positions = []
    species = []
    cell = []

    with open(qe_input) as f:
        lines = f.readlines()

    # Find the cell and positions in the output file
    cell_index = None
    positions_index = None
    for i, line in enumerate(lines):
        if 'CELL_PARAMETERS' in line:
            cell_index = int(i + 1)
        if 'ATOMIC_POSITIONS' in line:
            positions_index = int(i + 1)
    assert cell_index is not None, 'Failed to find cell in output'
    assert positions_index is not None, 'Failed to find positions in output'

    # Load cell
    for i in range(cell_index, cell_index + 3):
        cell_line = lines[i].strip()
        cell.append(np.fromstring(cell_line, sep=' '))
    cell = np.array(cell)

    # Check cell IO
    assert cell != [], 'Cell failed to load'
    assert np.shape(cell) == (3, 3), 'Cell failed to load correctly'

    # Load positions
    for i in range(positions_index, len(lines)):
        line_string = lines[i].strip().split()
        species.append(line_string[0])

        pos_string = ' '.join(line_string[1:4])

        positions.append(np.fromstring(pos_string, sep=' '))
    # Check position IO
    assert positions != [], "Positions failed to load"

    return positions, species, cell


# TODO get masses
def parse_qe_forces(outfile):
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

    assert total_energy != np.nan, "Quantum ESPRESSO parser failed to read the file {}. Run failed.".format(
        outfile)

    return forces


def parse_output(outfile):
    """
    Parse the output of a otf run for analysis
    :param outfile: str, Path to file
    :return: dict{int:value,'species':list}, Dict of positions, forces, vars indexed by frame and of species
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
    os.system('cp tests/test_files/qe_input_1.in pwscf.in')

    otf = OTF('pwscf.in', .1, 10, kernel='two_body',
              cutoff=10)
    otf.run()
    # parse_output('otf_run.out')
    pass
