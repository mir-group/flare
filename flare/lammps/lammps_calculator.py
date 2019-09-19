"""LAMMPS calculator for preparing and parsing single-point LAMMPS \
calculations."""
import os
import numpy as np

# TODO: split LAMMPS input and data files into separate classes


def run_lammps(lammps_executable, input_file, output_file):
    """Runs a single point LAMMPS calculation.

    :param lammps_executable: LAMMPS executable file.
    :type lammps_executable: str
    :param input_file: LAMMPS input file.
    :type input_file: str
    :param output_file: Desired LAMMPS output file.
    :type output_file: str
    """
    # run lammps
    lammps_command = '%s < %s > %s' % (lammps_executable,
                                       input_file,
                                       output_file)
    os.system(lammps_command)


def lammps_parser(dump_file):
    """Parses LAMMPS dump file. Assumes the forces are the final quantities \
to get dumped.

    :param dump_file: Dump file to be parsed.
    :type dump_file: str
    :return: Numpy array of forces on atoms.
    :rtype: np.ndarray
    """
    forces = []

    with open(dump_file, 'r') as outf:
        lines = outf.readlines()

    for count, line in enumerate(lines):
        if line.startswith('ITEM: ATOMS'):
            force_start = count

    for line in lines[force_start+1:]:
        fline = line.split()
        forces.append([float(fline[-3]),
                       float(fline[-2]),
                       float(fline[-1])])

    return np.array(forces)


# -----------------------------------------------------------------------------
#                           data functions
# -----------------------------------------------------------------------------

def lammps_dat(structure, atom_types, atom_masses, species):
    """Create LAMMPS data file for an uncharged material.

    :param structure: Structure object containing coordinates and cell.
    :type structure: struc.Structure
    :param atom_types: Atom types ranging from 1 to N.
    :type atom_types: List[int]
    :param atom_masses: Atomic masses of the atom types.
    :type atom_masses: List[int]
    :param species: Type of each atom.
    :type species: List[int]
    """

    dat_text = """Header of the LAMMPS data file

%i atoms
%i atom types
""" % (structure.nat, len(atom_types))

    dat_text += lammps_cell_text(structure)
    dat_text += """
Masses

"""
    mass_text = ''
    for atom_type, atom_mass in zip(atom_types, atom_masses):
        mass_text += '%i %i\n' % (atom_type, atom_mass)
    dat_text += mass_text
    dat_text += """
Atoms
"""
    dat_text += lammps_pos_text(structure, species)

    return dat_text


def lammps_dat_charged(structure, atom_types, atom_charges, atom_masses,
                       species):
    """Create LAMMPS data file for a charged material.

    :param structure: Structure object containing coordinates and cell.
    :type structure: struc.Structure
    :param atom_types: List of atom types.
    :type atom_types: List[int]
    :param atom_charges: Charge of each atom.
    :type atom_charges: List[float]
    :param atom_masses: Mass of each atom type.
    :type atom_masses: List[float]
    :param species: Type of each atom.
    :type species: List[int]
    """

    dat_text = """Header of the LAMMPS data file

%i atoms
%i atom types
""" % (structure.nat, len(atom_types))

    dat_text += lammps_cell_text(structure)
    dat_text += """
Masses

"""
    mass_text = ''
    for atom_type, atom_mass in zip(atom_types,
                                    atom_masses):
        mass_text += '%i %i\n' % (atom_type, atom_mass)
    dat_text += mass_text
    dat_text += """
Atoms
"""
    dat_text += lammps_pos_text_charged(structure, atom_charges, species)

    return dat_text


def lammps_cell_text(structure):
    """ Write cell from structure object."""

    cell_text = """
0.0 %f  xlo xhi
0.0 %f  ylo yhi
0.0 %f  zlo zhi
%f %f %f  xy xz yz
""" % (structure.cell[0, 0],
       structure.cell[1, 1],
       structure.cell[2, 2],
       structure.cell[1, 0],
       structure.cell[2, 0],
       structure.cell[2, 1])

    return cell_text


def lammps_pos_text(structure, species):
    """Create LAMMPS position text for a system of uncharged particles."""

    pos_text = '\n'
    for count, (pos, spec) in enumerate(zip(structure.positions, species)):
        pos_text += '%i %i %f %f %f \n' % \
            (count+1, spec, pos[0], pos[1], pos[2])
    return pos_text


def lammps_pos_text_charged(structure, charges, species):
    """Create LAMMPS position text for a system of charged particles."""

    pos_text = '\n'
    for count, (pos, chrg, spec) in enumerate(zip(structure.positions, charges,
                                                  species)):
        pos_text += '%i %i %f %f %f %f \n' % \
            (count+1, spec, chrg, pos[0], pos[1], pos[2])
    return pos_text


def write_text(file, text):
    """Write text to file."""

    with open(file, 'w') as fin:
        fin.write(text)

# -----------------------------------------------------------------------------
#                           input functions
# -----------------------------------------------------------------------------


def generic_lammps_input(dat_file, style_string, coeff_string, dump_file):
    """Create text for generic LAMMPS input file."""

    input_text = """# generic lammps input file
units metal
atom_style atomic
dimension  3
boundary   p p p
newton off
read_data %s

pair_style %s
pair_coeff %s

thermo_style one
dump 1 all custom 1 %s id type x y z fx fy fz
dump_modify 1 sort id
run 0
""" % (dat_file, style_string, coeff_string, dump_file)

    return input_text


def ewald_input(dat_file, short_cut, kspace_accuracy, dump_file):
    """Create text for Ewald input file."""

    input_text = """# Ewald input file
units metal
atom_style charge
dimension  3
boundary   p p p
read_data %s

pair_style coul/long %s
pair_coeff * *
kspace_style ewald %s

thermo_style one
dump 1 all custom 1 %s id type x y z fx fy fz
dump_modify 1 sort id
run 0
""" % (dat_file, short_cut, kspace_accuracy, dump_file)

    return input_text
