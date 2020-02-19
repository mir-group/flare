"""
This module is used to call CP2K simulation and parse its output
The user need to supply a complete input script with ENERGY_FORCE or ENERGY runtype, and CELL, COORD blocks. Example scripts can be found in tests/test_files/cp2k_input...

The module will copy the input template to a new file with "_run" suffix,
edit the atomic coordination in the COORD blocks and run the similation with
the parallel set up given.

We note that, if the CP2K executable is only for serial run, using it along with MPI setting can lead to repeating output in the output file, wrong number of forces and error in the other modules.
"""

import os
from subprocess import call
import time
import numpy as np
from flare import output
from flare import struc
from typing import List

name = "CP2K"

def run_dft_par(dft_input, structure, dft_loc, ncpus=1, dft_out="dft.out",
                npool=None, mpi="mpi", **dft_kwargs):
    """run DFT calculation with given input template
    and atomic configurations. if ncpus == 1, it executes serial run.

    :param dft_input: input template file name
    :param structure: atomic configuration
    :param dft_loc:   relative/absolute executable of the DFT code
    :param ncpus:   # of CPU for mpi
    :param dft_out:   output file name
    :param npool:     not used
    :param mpi:       not used
    :param **dft_wargs: not used
    :return: forces
    """

    newfilename = edit_dft_input_positions(dft_input, structure)
    dft_command = \
        f'{dft_loc} -i {newfilename}'
    if (ncpus > 1):
        if (mpi == "mpi"):
            dft_command = f'mpirun -np {ncpus} {dft_command}'
        else:
            dft_command = f'srun -n {ncpus} {dft_command}'

    # output.write_to_output(dft_command+'\n')
    with open(dft_out, "w+") as fout:
        call(dft_command.split(), stdout=fout)

    os.remove(newfilename)

    return parse_dft_forces(dft_out)


def run_dft_en_par(dft_input:str, structure,
        dft_loc:str, ncpus:int, dft_out:str ="dft.out",
        npool:int =None, mpi:str ="mpi", **dft_kwargs):
    """run DFT calculation with given input template
    and atomic configurations. This function is not used atm.

    :param dft_input: input template file name
    :param structure: atomic configuration
    :param dft_loc:   relative/absolute executable of the DFT code
    :param ncpus:   # of CPU for mpi
    :param dft_out:   output file name
    :param npool:     not used
    :param mpi:       not used
    :param **dft_wargs: not used
    :return: forces, energy
    """

    newfilename = edit_dft_input_positions(dft_input, structure)
    dft_command = \
        f'{dft_loc} -i {newfilename} > {dft_out}'
    if (ncpus > 1):
        dft_command = f'mpirun -np {ncpus} {dft_command}'
    # output.write_to_output(dft_command+'\n')
    call(dft_command, shell=True)
    os.remove(newfilename)

    forces, energy = parse_dft_forces_and_energy(dft_out)

    return forces, energy


def parse_dft_input(dft_input: str):
    """ Parse CP2K input file prepared by the user
    the parser is very limited. The user have to define things
    in a good format.
    It requires the "CELL", "COORD" blocks

    :param dft_input: file name
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
    # species_index = None

    for i, line in enumerate(lines):
        if '&CELL' in line:
            cell_index = int(i + 1)
        elif 'COORD' in line and 'END' not in line:
            positions_index = int(i + 1)
        elif ('&END' in line and (positions_index is not None) and (nat is None)):
            nat = i - positions_index
    #     if 'ATOMIC_SPECIES' in line:
    #         species_index = int(i + 1)

    assert cell_index is not None, 'Failed to find cell in input'
    assert positions_index is not None, 'Failed to find positions in input'
    assert nat is not None, 'Failed to find number of atoms in input'

    # Load cell
    # TO DO: allow to mess up the order of A, B, and C
    for i in range(cell_index, cell_index + 3):
        cell_line = list(map(float,lines[i].split()[1:]))
        cell.append(cell_line) #np.fromstring(cell_line[1:], sep=' '))
    cell = np.array(cell)

    # Check cell IO
    assert len(cell) != 0, 'Cell failed to load'
    assert np.shape(cell) == (3, 3), 'Cell failed to load correctly'

    # Load positions
    for i in range(positions_index, positions_index + nat):
        pos_line = lines[i].split()
        species.append(pos_line[0])
        # positions.append(np.fromstring(pos_string, sep=' '))
        positions.append(list(map(float, pos_line[1:])))

    # Check position IO
    assert positions != [], "Positions failed to load"
    positions = np.array(positions)

    # see conversions.nb for conversion from amu to md units
    ele_mass = {"H":1.007900, "He":4.002600, "Li":6.941000, "Be":9.012200, "B":10.811000,
                "C":12.010700, "N":14.006700, "O":15.999400, "F":18.998400, "Ne":20.179700,
                "Na":22.989700, "Mg":24.305000, "Al":26.981500, "Si":28.085500, "P":30.973800,
                "S":32.065000, "Cl":35.453000, "K":39.098300, "Ar":39.948000, "Ca":40.078000,
                "Sc":44.955900, "Ti":47.867000, "V":50.941500, "Cr":51.996100, "Mn":54.938000,
                "Fe":55.845000, "Ni":58.693400, "Co":58.933200, "Cu":63.546000, "Zn":65.390000,
                "Ga":69.723000, "Ge":72.640000, "As":74.921600, "Se":78.960000, "Br":79.904000,
                "Kr":83.800000, "Rb":85.467800, "Sr":87.620000, "Y":88.905900, "Zr":91.224000,
                "Nb":92.906400, "Mo":95.940000, "Tc":98.000000, "Ru":101.070000, "Rh":102.905500,
                "Pd":106.420000, "Ag":107.868200, "Cd":112.411000, "In":114.818000, "Sn":118.710000,
                "Sb":121.760000, "I":126.904500, "Te":127.600000, "Xe":131.293000, "Cs":132.905500,
                "Ba":137.327000, "La":138.905500, "Ce":140.116000, "Pr":140.907700, "Nd":144.240000,
                "Pm":145.000000, "Sm":150.360000, "Eu":151.964000, "Gd":157.250000, "Tb":158.925300,
                "Dy":162.500000, "Ho":164.930300, "Er":167.259000, "Tm":168.934200, "Yb":173.040000,
                "Lu":174.967000, "Hf":178.490000, "Ta":180.947900, "W":183.840000, "Re":186.207000,
                "Os":190.230000, "Ir":192.217000, "Pt":195.078000, "Au":196.966500, "Hg":200.590000,
                "Tl":204.383300, "Pb":207.200000, "Bi":208.980400, "Po":209.000000, "At":210.000000,
                "Rn":222.000000, "Fr":223.000000, "Ra":226.000000, "Ac":227.000000, "Pa":231.035900,
                "Th":232.038100, "Np":237.000000, "U":238.028900, "Am":243.000000, "Pu":244.000000,
                "Cm":247.000000, "Bk":247.000000, "Cf":251.000000, "Es":252.000000, "Fm":257.000000,
                "Md":258.000000, "No":259.000000, "Rf":261.000000, "Lr":262.000000, "Db":262.000000,
                "Bh":264.000000, "Sg":266.000000, "Mt":268.000000, "Rg":272.000000, "Hs":277.000000}

    # TO DO: allow customize mass
    massconvert = 0.000103642695727
    masses = {}
    for ele in ele_mass.keys():
        # Expects lines of format like: H 1.0 H_pseudo_name.ext
        masses[ele] = ele_mass[ele] * massconvert

    return positions, species, cell, masses


def dft_input_to_structure(dft_input: str):
    """
    Parses a qe input and returns the atoms in the file as a Structure object
    :param dft_input: input file to parse
    :return: atomic structure
    """
    positions, species, cell, masses = parse_dft_input(dft_input)
    _, coded_species = struc.get_unique_species(species)
    return struc.Structure(positions=positions, species=coded_species,
                           cell=cell, mass_dict=masses, species_labels=species)


def edit_dft_input_positions(dft_input: str, structure):
    """ Write the current configuration of the OTF structure to the
    qe input file

    :param dft_input: intput file name
    :param structure: structure to print
    :type structure: class Structure
    :return newfilename: the name of the edited intput file.
                         with "_run" suffix
    """

    with open(dft_input, 'r') as f:
        lines = f.readlines()

    file_pos_index = None
    cell_index = None
    nat = None

    for i, line in enumerate(lines):
        if '&CELL' in line:
            cell_index = int(i + 1)
        if '&COORD' in line:
            file_pos_index = int(i + 1)
        if ('&END' in line and (file_pos_index is not None)):
            nat = i - file_pos_index
    #     if 'ATOMIC_SPECIES' in line:
    #         species_index = int(i + 1)

    assert file_pos_index is not None, 'Failed to find positions in input'
    assert cell_index is not None, 'Failed to find cell in input'
    assert nat is not None, 'Failed to find nat in input'

    for pos_index, line_index in enumerate(
            range(file_pos_index, file_pos_index + structure.nat)):
        pos =  structure.positions[pos_index]
        specs = structure.species_labels[pos_index]
        pos_string = f'{specs} {pos[0]} {pos[1]} {pos[2]}\n'
        if line_index < len(lines):
            lines[line_index] = pos_string
        else:
            lines.append(pos_string)

    # # TODO current assumption: if there is a new structure, then the new
    # # structure has fewer atoms than the  previous one. If we are always
    # # 'editing' a version of the larger structure than this will be okay with
    # # the punchout method.
    # for line_index in range(file_pos_index + structure.nat,
    #                         file_pos_index + nat):
    #     lines[line_index] = ''

    lines[cell_index] = 'A '+' '.join([str(x) for x in structure.vec1]) + '\n'
    lines[cell_index + 1] = 'B '+' '.join([str(x) for x in structure.vec2]) \
                            + '\n'
    lines[cell_index + 2] = 'C '+' '.join([str(x) for x in structure.vec3]) \
                            + '\n'

    newfilename = dft_input+"_run"

    with open(newfilename, 'w') as f:
        for line in lines:
            f.write(line)

    return newfilename


def parse_dft_forces_and_energy(outfile: str):
    """ Get forces from a pwscf file in eV/A
    the input run type to be ENERGY_FORCE

    :param outfile: str, Path to dft.output file
    :return: list[nparray] , List of forces acting on atoms
    :return: float, total potential energy
    """
    forces = []
    total_energy = np.nan

    startforce = -1
    with open(outfile, 'r') as outf:
        for line in outf:
            if line.find('FORCE_EVAL') != -1:
                total_energy = float(line.split()[8])

            if (startforce >= 2):
                if (line.find('SUM')!=-1):
                    startforce = -1
                else:
                    line = line.split()[3:]
                    forces.append(list(map(float, line)))
                    startforce += 1
            elif (startforce >=0):
                startforce += 1
            elif line.find('FORCES') != -1 and line.find('in') != -1:
                startforce = 0


    assert total_energy != np.nan, "dft parser failed to read " \
                                   "the file {}. Run failed."+outfile

    # Convert from ry/au to ev/angstrom
    conversion_factor = 25.71104309541616*2.0

    forces = np.array(forces)*conversion_factor
    total_energy *= 27.2114

    return forces, total_energy


def parse_dft_forces(outfile: str):
    """ Get forces from a pwscf file in eV/A

    :param outfile: str, Path to dft.output file
    :return: list[nparray] , List of forces acting on atoms
    """

    f, e = parse_dft_forces_and_energy(outfile)
    return f
