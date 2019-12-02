import numpy as np
import time
import pytest
from flare import struc, env
from flare import otf_parser
from flare.mgp.mgp import MappedGaussianProcess
from flare import mc_simple
from flare.lammps import lammps_calculator
import pickle
import os

@pytest.fixture(scope='module')
def otf_object():
    file_name = 'test_files/AgI_snippet.out'

    # parse otf output
    otf_traj = otf_parser.OtfAnalysis(file_name)

    yield otf_traj
    del otf_traj


@pytest.fixture(scope='module')
def structure(otf_object): 
    # create test structure
    otf_cell = otf_object.header['cell']
    species = np.array([47, 53] * 27)
    positions = otf_object.position_list[-1]
    test_struc = struc.Structure(otf_cell, species, positions)

    yield test_struc
    del test_struc


@pytest.fixture(scope='module')
def params():
    grid_num_2 = 64
    grid_num_3 = 25
    lower_cut = 2.5
    two_cut = 7.
    three_cut = 5.

    # set struc params. cell and masses arbitrary?
    mapped_cell = np.eye(3) * 100
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
                   'update': False}

    map_params = {'grid': grid_params, 
                  'struc': struc_params}

    yield map_params
    del map_params

def test_2_body(otf_object, structure, params):
    # reconstruct gp model
    hyp_no = 2
    otf_cell = otf_object.header['cell']
    sig2 = otf_object.gp_hyp_list[hyp_no-1][-1][0]
    ls2 = otf_object.gp_hyp_list[hyp_no-1][-1][1]
    noise = otf_object.gp_hyp_list[hyp_no-1][-1][-1]
    hyps = np.array([sig2, ls2, noise])

    kernel = mc_simple.two_body_mc
    kernel_grad = mc_simple.two_body_mc_grad
    gp_model = otf_object.make_gp(kernel=kernel, kernel_grad=kernel_grad,
                                  hyps=hyps, hyp_no=hyp_no)
    gp_model.par = True
    gp_model.hyp_labels = ['sig2', 'ls2', 'noise']
 
    # create MGP
    grid_params = params['grid']
    grid_params['bodies'] = [2]
    struc_params = params['struc']
    mgp_model = MappedGaussianProcess(gp_model.hyps, gp_model.cutoffs,
            grid_params, struc_params, mean_only=True, container_only=False,
            GP=gp_model, lmp_file_name='AgI_Molten_15.txt')

    # test if MGP prediction matches GP
    atom = 0
    environ = env.AtomicEnvironment(structure, atom, gp_model.cutoffs)

    gp_pred_x = gp_model.predict(environ, 1)
    mgp_pred = mgp_model.predict(environ, mean_only=True)

    # check mgp is within 1 meV/A of the gp
    assert(np.abs(mgp_pred[0][0] - gp_pred_x[0]) < 1e-3)


def test_3_body(otf_object, structure, params):
    # reconstruct gp model
    hyp_no = 2
    otf_cell = otf_object.header['cell']
    sig = otf_object.gp_hyp_list[hyp_no-1][-1][2]
    ls = otf_object.gp_hyp_list[hyp_no-1][-1][3]
    noise = otf_object.gp_hyp_list[hyp_no-1][-1][-1]
    hyps = np.array([sig, ls, noise])

    kernel = mc_simple.three_body_mc
    kernel_grad = mc_simple.three_body_mc_grad
    gp_model = otf_object.make_gp(kernel=kernel, kernel_grad=kernel_grad,
                                  hyps=hyps, hyp_no=hyp_no)
    gp_model.par = True
    gp_model.hyp_labels = ['sig3', 'ls3', 'noise']
 
    # create MGP
    grid_params = params['grid']
    grid_params['bodies'] = [3]
    struc_params = params['struc']
    mgp_model = MappedGaussianProcess(gp_model.hyps, gp_model.cutoffs,
            grid_params, struc_params, mean_only=True, container_only=False,
            GP=gp_model, lmp_file_name='AgI_Molten_15.txt')

    # test if MGP prediction matches GP
    atom = 0
    environ = env.AtomicEnvironment(structure, atom, gp_model.cutoffs)

    gp_pred_x = gp_model.predict(environ, 1)
    mgp_pred = mgp_model.predict(environ, mean_only=True)

    # check mgp is within 1 meV/A of the gp
    assert(np.abs(mgp_pred[0][0] - gp_pred_x[0]) < 1e-3)


def test_2_plus_3_body(otf_object, structure, params):
    # reconstruct gp model
    kernel = mc_simple.two_plus_three_body_mc
    kernel_grad = mc_simple.two_plus_three_body_mc_grad
    gp_model = otf_object.make_gp(kernel=kernel, kernel_grad=kernel_grad,
                                  hyp_no=2)
    gp_model.par = True
    gp_model.hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
 
    # create MGP
    grid_params = params['grid']
    grid_params['bodies'] = [2, 3]
    struc_params = params['struc']
    mgp_model = MappedGaussianProcess(gp_model.hyps, gp_model.cutoffs,
            grid_params, struc_params, mean_only=True, container_only=False,
            GP=gp_model, lmp_file_name='AgI_Molten_15.txt')

    # test if MGP prediction matches GP
    atom = 0
    environ = env.AtomicEnvironment(structure, atom, gp_model.cutoffs)

    gp_pred_x = gp_model.predict(environ, 1)
    mgp_pred = mgp_model.predict(environ, mean_only=True)

    # check mgp is within 1 meV/A of the gp
    assert(np.abs(mgp_pred[0][0] - gp_pred_x[0]) < 1e-3)



# ASSUMPTION: You have a Lammps executable with the mgp pair style with $lmp
# as the corresponding environment variable.
@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
def test_lammps(otf_object, structure):
    atom_types = [1, 2]
    atom_masses = [108, 127]
    atom_species = [1, 2] * 27

    # create data file
    lammps_location = 'AgI_Molten_15.txt'
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
    forces = otf_object.force_list[-1]
    assert(np.abs(lammps_forces[0, 1] - forces[0, 1]) < 1e-3)

    os.system('rm tmp.in tmp.out tmp.dump tmp.data'
              ' log.lammps')
    os.system('rm '+lammps_location)
    os.system('rm grid3*.npy')
    os.system('rm -r kv3')

