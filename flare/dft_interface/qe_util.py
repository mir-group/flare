"""
This module is used to call Quantum Espresso simulation and parse its output
The user need to supply a complete input script with single-point scf
calculation, CELL_PARAMETERS, ATOMIC_POSITIONS, nat, ATOMIC_SPECIES
arguments.  It is case sensitive. and the nat line should be the first
argument of the line it appears. The user can also opt to the ASE interface instead.

This module will copy the input template to a new file with "_run" suffix,
edit the atomic coordination in the ATOMIC_POSITIONS block and run the similation with the parallel set up given.
"""

import os
from subprocess import call
import time
import numpy as np
from flare import struc
from typing import List

name = "QE"

def run_dft_par(dft_input, structure, dft_loc, n_cpus=1, dft_out='pwscf.out',
                npool=None, mpi="mpi", **dft_kwargs):
    """run DFT calculation with given input template
    and atomic configurations. if n_cpus == 1, it executes serial run.

    :param dft_input: input template file name
    :param structure: atomic configuration
    :param dft_loc:   relative/absolute executable of the DFT code
    :param n_cpus:   # of CPU for mpi
    :param dft_out:   output file name
    :param npool:     not used
    :param mpi:       not used
    :param **dft_wargs: not used
    :return: forces
    """

    newfilename = edit_dft_input_positions(dft_input, structure)

    if npool is None:
        dft_command = \
            f'{dft_loc} -i {newfilename}'
    else:
        dft_command = \
            f'{dft_loc} -nk {npool} -i {newfilename}'

    if (n_cpus > 1):
        if (mpi == "mpi"):
            dft_command = f'mpirun -np {n_cpus} {dft_command}'
        else:
            dft_command = f'srun -n {n_cpus} --mpi=pmi2 {dft_command}'

    with open(dft_out, "w+") as fout:
        call(dft_command.split(), stdout=fout)

    os.remove(newfilename)

    return parse_dft_forces(dft_out)


def run_dft_en_par(dft_input, structure, dft_loc, n_cpus):
    """run DFT calculation with given input template
    and atomic configurations. This function is not used atm

    if n_cpus == 1, it executes serial run.

    :param dft_input: input template file name
    :param structure: atomic configuration
    :param dft_loc:   relative/absolute executable of the DFT code
    :param n_cpus:   # of CPU for mpi
    :param dft_out:   output file name
    :param npool:     not used
    :param mpi:       not used
    :param **dft_wargs: not used
    :return: forces, energy
    """

    run_qe_path = dft_input
    edit_dft_input_positions(run_qe_path, structure)
    qe_command = \
        'mpirun -np {n_cpus} {dft_loc} -i {run_qe_path}'

    with open('pwscf.out', "w+") as fout:
        call(qe_command.split(), stdout=fout)

    forces, energy = parse_dft_forces_and_energy('pwscf.out')

    return forces, energy


def run_dft_en_npool(qe_input, structure, dft_loc, npool):
    run_qe_path = qe_input
    edit_dft_input_positions(run_qe_path, structure)
    qe_command = \
        'mpirun {0} -npool {1} < {2} > {3}'.format(dft_loc, npool, run_qe_path,
                                                   'pwscf.out')
    call(qe_command, shell=True)
    forces, energy = parse_dft_forces_and_energy('pwscf.out')

    return forces, energy


def parse_dft_input(dft_input: str):
    """ parse the input to get information of atomic configuration

    :param dft_input: input file name
    :return: positions, species, cell, masses
    """

    positions = []
    species = []
    cell = []

    with open(dft_input) as f:
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


def dft_input_to_structure(dft_input: str):
    """ Parses a qe input and returns the atoms in the
    file as a Structure object

    :param dft_input: QE Input file to parse
    :return: class Structure
    """
    positions, species, cell, masses = parse_dft_input(dft_input)
    _, coded_species = struc.get_unique_species(species)
    return struc.Structure(positions=positions, species=coded_species,
                           cell=cell, mass_dict=masses, species_labels=species)


def edit_dft_input_positions(dft_input: str, structure):
    """
    Write the current configuration of the OTF structure to the
    qe input file

    :param dft_input: dft input file name
    :param structure: atomic structure to compute
    :return: the name of the edited file
    """

    with open(dft_input, 'r') as f:
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

    newfilename = dft_input + "_run"

    with open(newfilename, 'w') as f:
        for line in lines:
            f.write(line)

    return newfilename


def parse_dft_forces(outfile: str):
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


def parse_dft_forces_and_energy(outfile: str):
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
