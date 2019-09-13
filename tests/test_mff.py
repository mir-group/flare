import numpy as np
import time
import pytest
from flare import struc, env
from flare import otf_parser
from flare.mff.mff_mc import MappedForceField
from flare import mc_simple
from flare.lammps import lammps_calculator
import pickle
import os


# ASSUMPTION: You have a Lammps executable with the mff pair style with $lmp
# as the corresponding environment variable.
@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
def test_parse_header():
    # -------------------------------------------------------------------------
    #                  reconstruct gp model from otf snippet
    # -------------------------------------------------------------------------

    file_name = 'test_files/AgI_snippet.out'
    hyp_no = 2

    # parse otf output
    otf_object = otf_parser.OtfAnalysis(file_name)
    otf_cell = otf_object.header['cell']

    # reconstruct gp model
    kernel = mc_simple.two_plus_three_body_mc
    kernel_grad = mc_simple.two_plus_three_body_mc_grad
    gp_model = otf_object.make_gp(kernel=kernel, kernel_grad=kernel_grad,
                                  hyp_no=hyp_no)
    gp_model.par = True
    gp_model.hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']

    # -------------------------------------------------------------------------
    #                          check gp reconstruction
    # -------------------------------------------------------------------------

    # create test structure
    species = np.array([47, 53] * 27)
    positions = otf_object.position_list[-1]
    forces = otf_object.force_list[-1]
    structure = struc.Structure(otf_cell, species, positions)
    atom = 0
    environ = env.AtomicEnvironment(structure, atom, gp_model.cutoffs)
    force_comp = 2
    pred, _ = gp_model.predict(environ, force_comp)

    assert(np.isclose(pred, forces[0][1]))

    # -------------------------------------------------------------------------
    #                              map the potential
    # -------------------------------------------------------------------------

    file_name = 'AgI.gp'
    grid_num_2 = 64
    grid_num_3 = 15
    lower_cut = 2.
    two_cut = 7.
    three_cut = 5.
    lammps_location = 'AgI_Molten_15.txt'

    # set struc params. cell and masses arbitrary?
    mapped_cell = np.eye(3) * 100
    struc_params = {'species': [47, 53],
                    'cube_lat': mapped_cell,
                    'mass_dict': {'0': 27, '1': 16}}

    # grid parameters
    grid_params = {'bounds_2': [[lower_cut], [two_cut]],
                   'bounds_3': [[lower_cut, lower_cut, 0],
                                [three_cut, three_cut, np.pi]],
                   'grid_num_2': grid_num_2,
                   'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
                   'svd_rank_2': 64,
                   'svd_rank_3': 90,
                   'bodies': [2, 3],
                   'load_grid': None,
                   'load_svd': None}

    mff_model = MappedForceField(gp_model, grid_params, struc_params,
                                 mean_only=True)

    # -------------------------------------------------------------------------
    #                          test the mapped potential
    # -------------------------------------------------------------------------

    gp_pred_x = gp_model.predict(environ, 1)
    mff_pred = mff_model.predict(environ, mean_only=True)

    # check mff is within 1 meV/A of the gp
    assert(np.abs(mff_pred[0][0] - gp_pred_x[0]) < 1e-3)

    # -------------------------------------------------------------------------
    #                           check lammps potential
    # -------------------------------------------------------------------------

    mff_model.write_two_plus_three(lammps_location)

    # create test structure
    species = otf_object.gp_species_list[-1]
    positions = otf_object.position_list[-1]
    forces = otf_object.force_list[-1]
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
    style_string = 'mff'
    coeff_string = '* * {} 47 53 yes yes'.format(lammps_location)
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

    os.system('rm tmp.in tmp.out tmp.dump tmp.data AgI_Molten_15.txt'
              ' log.lammps')
