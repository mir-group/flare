import numpy as np
import subprocess
import os


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
    """Parses LAMMPS dump file. Assumes the forces are the final \
quantities to get dumped.

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


class Lammps_Calculator:
    def __init__(self, struc, style_string, coeff_string, lammps_folder,
                 lammps_executable, atom_types, atom_masses, species,
                 charges=None):
        self.struc = struc
        self.style_string = style_string
        self.coeff_string = coeff_string
        self.lammps_folder = lammps_folder
        self.lammps_executable = lammps_executable
        self.atom_types = atom_types
        self.atom_masses = atom_masses
        self.species = species
        self.charges = charges

        self.input_file = lammps_folder + '/tmp.in'
        self.output_file = lammps_folder + '/tmp.out'
        self.dat_file = lammps_folder + '/tmp.data'
        self.dump_file = lammps_folder + '/tmp.dump'

        self.input_text = self.lammps_input()
        self.dat_text = self.lammps_dat()

    def get_forces(self):
        self.lammps_generator()
        self.run_ewald()
        forces = self.lammps_parser()
        return forces

    def run_ewald(self):
        # create input and data files
        self.lammps_generator()

        # run lammps
        lammps_command = '%s < %s > %s' % (self.lammps_executable,
                                           self.input_file,
                                           self.output_file)
        os.system(lammps_command)

    def lammps_generator(self):
        self.write_file(self.input_file, self.input_text)
        self.write_file(self.dat_file, self.dat_text)

    def lammps_input(self):
        input_text = """# lammps input file created with eam.py.
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
""" % (self.dat_file, self.style_string, self.coeff_string, self.dump_file)

        return input_text

    def lammps_dat(self):
        dat_text = """Header of the LAMMPS data file

%i atoms
%i atom types
""" % (self.struc.nat, len(self.atom_types))

        dat_text += self.lammps_cell_text()
        dat_text += """
Masses

"""
        mass_text = ''
        for atom_type, atom_mass in zip(self.atom_types, self.atom_masses):
            mass_text += '%i %i\n' % (atom_type, atom_mass)
        dat_text += mass_text
        dat_text += """
Atoms
"""
        dat_text += self.lammps_pos_text()

        return dat_text

    def lammps_dat_charged(self):
        dat_text = """Header of the LAMMPS data file

%i atoms
%i atom types
""" % (self.struc.nat, len(self.atom_types))

        dat_text += self.lammps_cell_text()
        dat_text += """
Masses

"""
        mass_text = ''
        for atom_type, atom_mass in zip(self.atom_types,
                                        self.atom_masses):
            mass_text += '%i %i\n' % (atom_type, atom_mass)
        dat_text += mass_text
        dat_text += """
Atoms
"""
        dat_text += self.lammps_pos_text_charged()

        return dat_text

    def lammps_cell_text(self):
        """ Write cell from structure object. Assumes orthorombic periodic
        cell."""

        cell_text = """
0.0 %f  xlo xhi
0.0 %f  ylo yhi
0.0 %f  zlo zhi
%f %f %f  xy xz yz
""" % (self.struc.cell[0, 0],
            self.struc.cell[1, 1],
            self.struc.cell[2, 2],
            self.struc.cell[1, 0],
            self.struc.cell[2, 0],
            self.struc.cell[2, 1])

        return cell_text

    def lammps_pos_text(self):
        pos_text = '\n'
        for count, (pos, spec) in enumerate(zip(self.struc.positions,
                                                self.species)):
            pos_text += '%i %i %f %f %f \n' % \
                (count+1, spec, pos[0], pos[1], pos[2])
        return pos_text

    def lammps_pos_text_charged(self):
        pos_text = '\n'
        for count, (pos, chrg, spec) in enumerate(zip(self.struc.positions,
                                                      self.charges,
                                                      self.species)):
            pos_text += '%i %i %f %f %f %f \n' % \
                (count+1, spec, chrg, pos[0], pos[1], pos[2])
        return pos_text

    @staticmethod
    def write_file(file, text):
        with open(file, 'w') as fin:
            fin.write(text)
