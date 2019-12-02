import numpy as np
from flare import otf_parser

file_name = 'test_files/AgI_snippet.out'
hyp_no = 2 # use the hyperparameters from the 2nd training step
otf_object = otf_parser.OtfAnalysis(file_name)

# -------------------------------------------------------------------------
#                  reconstruct gp model from otf snippet
# -------------------------------------------------------------------------
from flare import mc_simple

kernel = mc_simple.two_plus_three_body_mc
kernel_grad = mc_simple.two_plus_three_body_mc_grad
gp_model = otf_object.make_gp(kernel=kernel, kernel_grad=kernel_grad,
                              hyp_no=hyp_no)
gp_model.par = True
gp_model.hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']

# write model to a binary file
gp_model.write_model('AgI.gp', format='pickle')
# -------------------------------------------------------------------------
#                              map the potential
# -------------------------------------------------------------------------
from flare.mgp.mgp import MappedGaussianProcess

grid_num_2 = 64
grid_num_3 = 20
lower_cut = 2.5
two_cut = 7.
three_cut = 5.
lammps_location = 'AgI_Molten_15.txt'

# set struc params. cell and masses arbitrary?
mapped_cell = np.eye(3) * 100 # just use a sufficiently large one
struc_params = {'species': [47, 53],
                'cube_lat': mapped_cell}

# grid parameters
grid_params = {'bounds_2': [[lower_cut], [two_cut]],
               'bounds_3': [[lower_cut, lower_cut, -1],
                            [three_cut, three_cut,  1]],
               'grid_num_2': grid_num_2,
               'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
               'svd_rank_2': 64,
               'svd_rank_3': 90,
               'bodies': [2, 3],
               'load_grid': None,
               'update': True}

mgp_model = MappedGaussianProcess(gp_model.hyps, gp_model.cutoffs,
        grid_params, struc_params, mean_only=True, container_only=False,
        GP=gp_model, lmp_file_name=lammps_location)
# -------------------------------------------------------------------------
#                          test the mapped potential
# -------------------------------------------------------------------------
gp_pred_x = gp_model.predict(environ, 1)
mgp_pred = mgp_model.predict(environ, mean_only=True)

# check mgp is within 1 meV/A of the gp
assert(np.abs(mgp_pred[0][0] - gp_pred_x[0]) < 1e-3)

# -------------------------------------------------------------------------
#                           check lammps potential
# -------------------------------------------------------------------------
from flare import struc, env
from flare.lammps import lammps_calculator

# lmp coef file is automatically written now every time MGP is constructed

# create test structure
species = otf_object.gp_species_list[-1]
positions = otf_object.position_list[-1]
forces = otf_object.force_list[-1]
otf_cell = otf_object.header['cell']
structure = struc.Structure(otf_cell, species, positions)

atom_types = [1, 2]
atom_masses = [108, 127]
atom_species = [1, 2] * 27

# create data file
data_file_name = 'tmp.data'
data_text = lammps_calculator.lammps_dat(structure, atom_types,
                                         atom_masses, atom_species)
lammps_calculator.write_text(data_file_name, data_text)

# create lammps input
style_string = 'mgp' 
coeff_string = '* * {} Ag I yes yes'.format(lammps_location)
lammps_executable = '$lmp'
dump_file_name = 'tmp.dump'
input_file_name = 'tmp.in'
output_file_name = 'tmp.out'
input_text = \
    lammps_calculator.generic_lammps_input(data_file_name, style_string,
                                           coeff_string, dump_file_name)
lammps_calculator.write_text(input_file_name, input_text)

lammps_calculator.run_lammps(lammps_executable, input_file_name,
                             output_file_name)

lammps_forces = lammps_calculator.lammps_parser(dump_file_name)

# check that lammps agrees with gp to within 1 meV/A
assert(np.abs(lammps_forces[0, 1] - forces[0, 1]) < 1e-3)

