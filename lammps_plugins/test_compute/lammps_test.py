from flare import struc, env, otf_parser
from flare.lammps import lammps_calculator
import datetime
import numpy as np

# define cell, positions, and species
cell = 100 * np.eye(3)
positions = np.array([[0.0265474, 0.683021, -0.0641653],
                      [0.182227,  -0.461365,  -0.425575],
                      [0.691963,  -0.169211,  -0.643345],
                      [-0.175838,   0.074608,   -0.69256]])
atom_types = [1, 2]
atom_masses = [108, 108]
atom_species = [1, 2, 1, 2]
structure = struc.Structure(cell, atom_species, positions)

# set up input and data files
data_file_name = 'tmp.data'
lammps_location = 'beta.txt'
# lammps_location = 'Ag_u3.eam'
style_string = 'flare'
coeff_string = '* * {}'.format(lammps_location)
lammps_executable = '/Users/jonpvandermause/Codes/lammps/src/lmp_mpi'
dump_file_name = 'tmp.dump'
input_file_name = 'tmp.in'
output_file_name = 'tmp.out'

# write data file
data_text = lammps_calculator.lammps_dat(structure, atom_types,
                                         atom_masses, atom_species)
lammps_calculator.write_text(data_file_name, data_text)

# write input file
input_text = \
    lammps_calculator.generic_lammps_input(data_file_name, style_string,
                                           coeff_string, dump_file_name)
lammps_calculator.write_text(input_file_name, input_text)

# run lammps
lammps_calculator.run_lammps(lammps_executable, input_file_name,
                             output_file_name)
