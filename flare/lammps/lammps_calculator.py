"""LAMMPS calculator for preparing and parsing single-point LAMMPS \
calculations."""
import subprocess
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
    lammps_command = f'{lammps_executable} -in {input_file} '
    print("run command:", lammps_command)
    with open("tmp2False.out", "w+") as fout:
        subprocess.call(lammps_command.split(), stdout=fout)


def lammps_parser(dump_file, std=False):
    """Parses LAMMPS dump file. Assumes the forces are the final quantities \
to get dumped.

    :param dump_file: Dump file to be parsed.
    :type dump_file: str
    :return: Numpy array of forces on atoms.
    :rtype: np.ndarray
    """
    forces = []
    stds = []

    with open(dump_file, 'r') as outf:
        lines = outf.readlines()

    for count, line in enumerate(lines):
        if line.startswith('ITEM: ATOMS'):
            force_start = count

    for line in lines[force_start+1:]:
        fline = line.split()
        if std:
            forces.append([float(fline[-4]),
                           float(fline[-3]),
                           float(fline[-2])])
            stds.append(float(fline[-1]))
        else:
            forces.append([float(fline[-3]),
                           float(fline[-2]),
                           float(fline[-1])])

    return np.array(forces), np.array(stds)


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

    dat_text = f"""Header of the LAMMPS data file

{structure.nat} atoms
{len(atom_types)} atom types
"""

    dat_text += lammps_cell_text(structure)
    dat_text += """
Masses

"""
    mass_text = ''
    for atom_type, atom_mass in zip(atom_types, atom_masses):
        mass_text += f'{atom_type} {atom_mass}\n'
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

    dat_text = f"""Header of the LAMMPS data file

{structure.nat} atoms
{len(atom_types)} atom types
"""

    dat_text += lammps_cell_text(structure)
    dat_text += """
Masses

"""
    mass_text = ''
    for atom_type, atom_mass in zip(atom_types,
                                    atom_masses):
        mass_text += f'{atom_type} {atom_mass}\n'
    dat_text += mass_text
    dat_text += """
Atoms
"""
    dat_text += lammps_pos_text_charged(structure, atom_charges, species)

    return dat_text


def lammps_cell_text(structure):
    """ Write cell from structure object."""

    cell_text = f"""
0.0 {structure.cell[0, 0]}  xlo xhi
0.0 {structure.cell[1, 1]}  ylo yhi
0.0 {structure.cell[2, 2]}  zlo zhi
{structure.cell[1, 0]} {structure.cell[2, 0]} {structure.cell[2, 1]}  xy xz yz
"""

    return cell_text


def lammps_pos_text(structure, species):
    """Create LAMMPS position text for a system of uncharged particles."""

    pos_text = '\n'
    for count, (pos, spec) in enumerate(zip(structure.positions, species)):
        pos_text += f'{count+1} {spec} {pos[0]} {pos[1]} {pos[2]}\n'
    return pos_text


def lammps_pos_text_charged(structure, charges, species):
    """Create LAMMPS position text for a system of charged particles."""

    pos_text = '\n'
    for count, (pos, chrg, spec) in enumerate(zip(structure.positions, charges,
                                                  species)):
        pos_text += f'{count+1} {spec} {chrg} {pos[0]} {pos[1]} {pos[2]}\n'
    return pos_text


def write_text(file, text):
    """Write text to file."""

    with open(file, 'w') as fin:
        fin.write(text)

# -----------------------------------------------------------------------------
#                           input functions
# -----------------------------------------------------------------------------


def generic_lammps_input(dat_file, style_string, coeff_string, dump_file, 
        newton=False, std_string=''):
    """Create text for generic LAMMPS input file."""

    if newton:
        ntn = 'on'
    else:
        ntn = 'off'

    if std_string != '':
        compute_cmd = f"compute std all uncertainty/atom {std_string}"
        c_std = "c_std"
    else:
        compute_cmd = ""
        c_std = ""

    input_text = f"""# generic lammps input file
units metal
atom_style atomic
dimension  3
boundary   p p p
newton {ntn}
read_data {dat_file}

pair_style {style_string}
pair_coeff {coeff_string}

thermo_style one
{compute_cmd}
dump 1 all custom 1 {dump_file} id type x y z fx fy fz {c_std}
dump_modify 1 sort id
run 0
"""

    return input_text


def ewald_input(dat_file, short_cut, kspace_accuracy, dump_file, newton=True):
    """Create text for Ewald input file."""
    if newton is True:
        ntn = 'on'
    else:
        ntn = 'off'

    input_text = f"""# Ewald input file
newton {ntn}
units metal
atom_style charge
dimension  3
boundary   p p p
read_data {dat_file}

pair_style coul/long {short_cut}
pair_coeff * *
kspace_style ewald {kspace_accuracy}

thermo_style one
dump 1 all custom 1 {dump_file} id type x y z fx fy fz
dump_modify 1 sort id
run 0
"""

    return input_text
