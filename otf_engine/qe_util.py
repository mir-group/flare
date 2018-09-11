#!/usr/bin/env python3
# pylint: disable=redefined-outer-name
""""
Assembly of helper functions which call pw.x in the Quantum ESPRESSO suite

Assumes the environment variable PWSCF_COMMAND is set as a path to the pw.x
binary

Steven Torrisi
"""

import os
import numpy as np
from struc import Structure
from typing import List


def run_espresso(qe_input: str, structure: Structure, temp: bool = False) -> \
        List[np.array]:
    """
    Calls quantum espresso from input located at self.qe_input

    :param qe_input: Path to pwscf.in file
    :type qe_input: str
    :param structure: Structure object for the edited
    :type structure: Structure
    :param temp: Run pwscf off of an edited QE input instead of the original
    :type temp: Bool

    :return: List [nparray] List of forces
    """

    # If temp, copy the extant file into a new run directory
    run_qe_path = qe_input
    run_qe_path += '_run ' if temp else ''

    os.system(' '.join(['cp', qe_input, run_qe_path]))
    edit_qe_input_positions(run_qe_path, structure)

    pw_loc = os.environ.get('PWSCF_COMMAND', 'pwscf.x')

    qe_command = '{0} < {1} > {2}'.format(pw_loc, run_qe_path,
                                          'pwscf.out')

    os.system(qe_command)

    return parse_qe_forces('pwscf.out')


def parse_qe_input(qe_input: str):
    """
    Reads the positions, species, cell, and masses in from the qe input file

    :param qe_input: Path to PWSCF input file
    :type qe_input: str
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


def edit_qe_input_positions(qe_input: str, structure: Structure):
    """
    Write the current configuration of the OTF structure to the
    qe input file
    """

    with open(qe_input, 'r') as f:
        lines = f.readlines()

    file_pos_index = None
    cell_index = None
    nat = None
    for i, line in enumerate(lines):
        if 'ATOMIC_POSITIONS' in line:
            file_pos_index = int(i + 1)
        if 'CELL_PARAMETERS' in line:
            cell_index = int(i + 1)
        # Load nat into variable then overwrite it with new nat
        if 'nat' in line:
            nat = int(line.split('=')[1])
            nat_index = int(i)
            lines[nat_index] = 'nat = ' + str(structure.nat) + '\n'

    assert file_pos_index is not None, 'Failed to find positions in input'
    assert cell_index is not None, 'Failed to find cell in input'
    assert nat is not None, 'Failed to find nat in input'

    for pos_index, line_index in enumerate(
            range(file_pos_index, file_pos_index + structure.nat)):
        pos_string = ' '.join([structure.species[pos_index],
                               str(structure.positions[pos_index][
                                       0]),
                               str(structure.positions[pos_index][
                                       1]),
                               str(structure.positions[pos_index][
                                       2])])
        lines[line_index] = str(pos_string + '\n')

    # TODO current assumption: if there is a new structure, then the new
    # structure has fewer atoms than the  previous one. If we are always
    # 'editing' a version of the larger structure than this will be okay with
    # the punchout method.
    for line_index in range(file_pos_index + structure.nat,
                            file_pos_index + nat):
        lines[line_index] = ''

    lines[cell_index] = ' '.join([str(x) for x in structure.vec1]) + '\n'
    lines[cell_index + 1] = ' '.join([str(x) for x in structure.vec2]) \
                            + '\n'
    lines[cell_index + 2] = ' '.join([str(x) for x in structure.vec3]) \
                            + '\n'

    with open(qe_input, 'w') as f:
        for line in lines:
            f.write(line)


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
                line = [x for x in line if x != '']
                temp_forces = []
                for x in line:
                    temp_forces.append(float(x))
                forces.append(np.array(list(temp_forces)))

    assert total_energy != np.nan, "Quantum ESPRESSO parser failed to read " \
                                   "the file {}. Run failed.".format(outfile)

    return forces
