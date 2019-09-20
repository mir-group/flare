import os
from subprocess import call
import time
import numpy as np
from flare import struc
from typing import List


def run_espresso(qe_input, structure, pw_loc):
    run_qe_path = qe_input
    edit_qe_input_positions(run_qe_path, structure)
    qe_command = '{0} < {1} > {2}'.format(pw_loc, run_qe_path,
                                                 'pwscf.out')
    # os.system(qe_command)
    call(qe_command, shell=True)

    return parse_qe_forces('pwscf.out')


def run_espresso_par(qe_input, structure, pw_loc, no_cpus):
    run_qe_path = qe_input
    edit_qe_input_positions(run_qe_path, structure)
    qe_command = \
        'mpirun -np {0} {1} < {2} > {3}'.format(no_cpus, pw_loc, run_qe_path,
                                                'pwscf.out')

    with open("alog", "a+") as fout:
        print(qe_command, file=fout)
    # os.system(qe_command)
    call(qe_command, shell=True)

    return parse_qe_forces('pwscf.out')


def run_espresso_en_par(qe_input, structure, pw_loc, no_cpus):
    run_qe_path = qe_input
    edit_qe_input_positions(run_qe_path, structure)
    qe_command = \
        'mpirun -np {0} {1} < {2} > {3}'.format(no_cpus, pw_loc, run_qe_path,
                                                'pwscf.out')
    # os.system(qe_command)
    call(qe_command, shell=True)

    forces, energy = parse_qe_forces_and_energy('pwscf.out')

    return forces, energy


def run_espresso_npool(qe_input, qe_output, structure, pw_loc, npool):
    run_qe_path = qe_input
    edit_qe_input_positions(run_qe_path, structure)
    qe_command = \
        'mpirun {0} -npool {1} < {2} > {3}'.format(pw_loc, npool, run_qe_path,
                                                   qe_output)
    # os.system(qe_command)
    call(qe_command, shell=True)

    return parse_qe_forces(qe_output)


def run_espresso_command(qe_input, qe_output, pw_loc, npool):
    qe_command = \
        'mpirun {0} -npool {1} < {2} > {3}'.format(pw_loc, npool, qe_input,
                                                   qe_output)
    # os.system(qe_command)
    call(qe_command, shell=True)

    return parse_qe_forces(qe_output)


def parse_qe_input(qe_input: str):
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
    assert len(cell) != 0, 'Cell failed to load'
    assert np.shape(cell) == (3, 3), 'Cell failed to load correctly'

    # Load positions
    for i in range(positions_index, positions_index + nat):
        line_string = lines[i].strip().split()
        species.append(line_string[0])

        pos_string = ' '.join(line_string[1:4])

        positions.append(np.fromstring(pos_string, sep=' '))
    # Check position IO
    assert positions != [], "Positions failed to load"
    positions = np.array(positions)

    # see conversions.nb for conversion from amu to md units
    massconvert = 0.000103642695727
    masses = {}
    for i in range(species_index, species_index + len(set(species))):
        # Expects lines of format like: H 1.0 H_pseudo_name.ext
        line = lines[i].strip().split()
        masses[line[0]] = float(line[1]) * massconvert

    return positions, species, cell, masses


def qe_input_to_structure(qe_input: str):
    """
    Parses a qe input and returns the atoms in the file as a Structure object
    :param qe_input: QE Input file to parse
    :return:
    """
    positions, species, cell, masses = parse_qe_input(qe_input)
    _, coded_species = struc.get_unique_species(species)
    return struc.Structure(positions=positions, species=coded_species,
                           cell=cell, mass_dict=masses, species_labels=species)


def edit_qe_input_positions(qe_input: str, structure):
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

    # TODO Catch case where the punchout structure has more atoms than the
    # original structure

    for pos_index, line_index in enumerate(
            range(file_pos_index, file_pos_index + structure.nat)):
        pos_string = ' '.join([structure.species_labels[pos_index],
                               str(structure.positions[pos_index][
                                       0]),
                               str(structure.positions[pos_index][
                                       1]),
                               str(structure.positions[pos_index][
                                       2])])
        if line_index < len(lines):
            lines[line_index] = str(pos_string + '\n')
        else:
            lines.append(str(pos_string + '\n'))

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
    Get forces from a pwscf file in eV/A

    :param outfile: str, Path to pwscf output file
    :return: list[nparray] , List of forces acting on atoms
    """
    forces = []
    total_energy = np.nan

    with open(outfile, 'r') as outf:
        for line in outf:
            if line.lower().startswith('!    total energy'):
                total_energy = float(line.split()[-2])

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

    # Convert from ry/au to ev/angstrom
    conversion_factor = 25.71104309541616

    forces = [conversion_factor * force for force in forces]
    forces = np.array(forces)

    return forces


def parse_qe_forces_and_energy(outfile: str):
    """
    Get forces from a pwscf file in eV/A

    :param outfile: str, Path to pwscf output file
    :return: list[nparray] , List of forces acting on atoms
    """
    forces = []
    total_energy = np.nan

    with open(outfile, 'r') as outf:
        for line in outf:
            if line.lower().startswith('!    total energy'):
                total_energy = float(line.split()[-2])

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

    # Convert from ry/au to ev/angstrom
    conversion_factor = 25.71104309541616

    forces = [conversion_factor * force for force in forces]
    forces = np.array(forces)

    return forces, total_energy
