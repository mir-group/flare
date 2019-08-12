import numpy as np
import subprocess
import os


class EwaldCalculator:
    def __init__(self, struc, charges, cutoff, accuracy, lammps_folder,
                 lammps_executable):
        self.struc = struc
        self.charges = charges
        self.cutoff = cutoff
        self.accuracy = accuracy
        self.lammps_folder = lammps_folder
        self.lammps_executable = lammps_executable

        self.input_file = lammps_folder + '/tmp.in'
        self.output_file = lammps_folder + '/tmp.out'
        self.dat_file = lammps_folder + '/tmp.data'
        self.dump_file = lammps_folder + '/tmp.dump'

        self.input_text = self.lammps_ewald_input()
        self.dat_text = self.lammps_ewald_dat()

    def get_ewald_forces(self):
        self.lammps_ewald_generator()
        self.run_ewald()
        forces = self.lammps_ewald_parser()
        return forces

    def run_ewald(self):
        # create input and data files
        self.lammps_ewald_generator()

        # run lammps
        lammps_command = '%s < %s > %s' % (self.lammps_executable,
                                           self.input_file,
                                           self.output_file)
        os.system(lammps_command)

    def lammps_ewald_generator(self):
        self.write_file(self.input_file, self.input_text)
        self.write_file(self.dat_file, self.dat_text)

    def lammps_ewald_parser(self):
        forces = []

        with open(self.dump_file, 'r') as outf:
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

    def lammps_ewald_input(self):
        input_text = """# long range coulomb calculation
units metal
atom_style charge
dimension  3
boundary   p p p
read_data %s

pair_style coul/long %f
pair_coeff * *
kspace_style ewald %e

thermo_style one
dump 1 all custom 1 %s id type x y z q fx fy fz
dump_modify 1 sort id
run 0
""" % (self.dat_file, self.cutoff, self.accuracy, self.dump_file)

        return input_text

    def lammps_ewald_dat(self):
        dat_text = """Header of the LAMMPS data file

%i atoms
1 atom types
""" % (self.struc.nat)

        dat_text += self.lammps_cell_text()

        dat_text += """
Masses

1  1

Atoms
"""

        dat_text += self.lammps_pos_text()

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
        for count, pos in enumerate(self.struc.positions):
            pos_text += '%i 1 %f %f %f %f \n' % \
                (count+1, self.charges[self.struc.coded_species[count]],
                    pos[0], pos[1], pos[2])
        return pos_text

    @staticmethod
    def write_file(file, text):
        with open(file, 'w') as fin:
            fin.write(text)
