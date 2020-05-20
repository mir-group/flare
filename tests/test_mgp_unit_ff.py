import numpy as np
import time
import pytest
import os, pickle, re

from flare import struc, env, gp
from flare import otf_parser
from flare.mgp.mgp import MappedGaussianProcess
from flare.kernels.utils import str_to_kernel_set
from flare.lammps import lammps_calculator

from .fake_gp import get_force_gp, get_random_structure

body_list = ['2', '3']

def clean():
    for f in os.listdir("./"):
        if re.search(r"grid.*npy", f):
            os.remove(f)
        if re.search("kv3", f):
            os.rmdir(f)


# ASSUMPTION: You have a Lammps executable with the mgp pair style with $lmp
# as the corresponding environment variable.
@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')

@pytest.fixture(scope='module')
def all_gp():

    allgp_dict = {}
    np.random.seed(0)
    for bodies in ['2', '3', '2+3']:
        gp_model = get_force_gp(bodies, 'mc', False)
        gp_model.parallel = True
        gp_model.n_cpus = 2
        gp_model.set_L_alpha()
        allgp_dict[f'{bodies}'] = gp_model

    yield allgp_dict
    del allgp_dict

@pytest.fixture(scope='module')
def all_mgp():

    allmgp_dict = {}
    for bodies in ['2', '3', '2+3']:
        allmgp_dict[f'{bodies}'] = None

    yield allmgp_dict
    del allmgp_dict

@pytest.mark.parametrize('bodies', body_list)
def test_init(bodies, all_gp, all_mgp):
    """
    test the init function
    """

    gp_model = all_gp[f'{bodies}']

    grid_num_2 = 64
    grid_num_3 = 20
    lower_cut = 0.01
    two_cut = gp_model.cutoffs[0]
    three_cut = gp_model.cutoffs[1]
    lammps_location = f'{bodies}_mgp_ff.txt'

    # set struc params. cell and masses arbitrary?
    mapped_cell = np.eye(3) * 2
    struc_params = {'species': [1, 2],
                    'cube_lat': mapped_cell,
                    'mass_dict': {'0': 27, '1': 16}}

    # grid parameters
    # grid parameters
    blist = []
    if ('2' in bodies):
        blist+= [2]
    if ('3' in bodies):
        blist+= [3]
    grid_params = {'bounds_2': [[lower_cut], [two_cut]],
                   'bounds_3': [[lower_cut, lower_cut, -1],
                                [three_cut, three_cut,  1]],
                   'grid_num_2': grid_num_2,
                   'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
                   'svd_rank_2': 64,
                   'svd_rank_3': 90,
                   'bodies': blist,
                   'cutoffs': [two_cut, three_cut],
                   'load_grid': None,
                   'update': False}

    struc_params = {'species': [1, 2],
                    'cube_lat': mapped_cell,
                    'mass_dict': {'0': 27, '1': 16}}

    mgp_model = MappedGaussianProcess(gp_model.hyps, gp_model.cutoffs,
            grid_params, struc_params, mean_only=True, container_only=True,
            GP=gp_model, lmp_file_name=lammps_location)
    all_mgp[f'{bodies}'] = mgp_model


@pytest.mark.parametrize('bodies', body_list)
def test_build_map(all_gp, all_mgp, bodies):
    """
    test the mapping for mc_simple kernel
    """

    gp_model = all_gp[f'{bodies}']
    mgp_model = all_mgp[f'{bodies}']

    mgp_model.build_map(gp_model)

    clean()


@pytest.mark.parametrize('bodies', body_list)
def test_predict(all_gp, all_mgp, bodies):
    """
    test the predict for mc_simple kernel
    """

    gp_model = all_gp[f'{bodies}']
    mgp_model = all_mgp[f'{bodies}']
    atom_id = 1

    nenv=10
    cell = np.eye(3)
    cutoffs = gp_model.cutoffs
    unique_species = gp_model.training_data[0].species
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    test_envi = env.AtomicEnvironment(struc_test, atom_id, cutoffs)

    mgp_pred = mgp_model.predict(test_envi, mean_only=True)

    # check mgp is within 1 meV/A of the gp
    for s in range(3):
        gp_pred_x = gp_model.predict(test_envi, s+1)
        print(mgp_pred, gp_pred_x)
        assert(np.abs(mgp_pred[0][s] - gp_pred_x[0]) < 1e-3), \
                f"{bodies} body mapping is wrong"

    clean()


@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
@pytest.mark.parametrize('bodies', body_list)
def test_lmp_predict(all_gp, all_mgp, bodies):
    """
    test the lammps implementation
    """

    mgp_model = all_mgp[f'{bodies}']
    gp_model = all_gp[f'{bodies}']

    atom_id = 1

    # lmp file is automatically written now every time MGP is constructed
    lammps_location = mgp_model.lmp_file_name
    mgp_model.write_lmp_file(lammps_location)

    # create test structure
    cell = np.eye(3)
    unique_species = gp_model.training_data[0].species
    nenv = 10
    cutoffs = gp_model.cutoffs
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    test_envi = env.AtomicEnvironment(struc_test, atom_id, cutoffs)
    atom_types = [1, 2]
    atom_masses = [108, 127]
    atom_species = struc_test.coded_species

    # create data file
    data_file_name = f'tmp{bodies}.data'
    data_text = lammps_calculator.lammps_dat(struc_test, atom_types,
                                             atom_masses, atom_species)
    lammps_calculator.write_text(data_file_name, data_text)

    # create lammps input
    by = 'no'
    ty = 'no'
    if '2' in bodies:
        by = 'yes'
    if '3' in bodies:
        ty = 'yes'
    style_string = 'mgpf'
    coeff_string = f'* * {lammps_location} H He {by} {ty}'
    lammps_executable = os.environ.get('lmp')
    dump_file_name = f'tmp{bodies}.dump'
    input_file_name = f'tmp{bodies}.in'
    output_file_name = f'tmp{bodies}.out'
    input_text = \
        lammps_calculator.generic_lammps_input(data_file_name, style_string,
                                               coeff_string, dump_file_name,
                                               newton=False)
    lammps_calculator.write_text(input_file_name, input_text)

    lammps_calculator.run_lammps(lammps_executable, input_file_name,
                                 output_file_name)

    lammps_forces = lammps_calculator.lammps_parser(dump_file_name)
    mgp_forces = mgp_model.predict(test_envi, mean_only=True)

    # check that lammps agrees with gp to within 1 meV/A
    for s in range(3):
        assert (np.abs(lammps_forces[atom_id, s] - mgp_forces[0][s]) < 1e-3)

    for f in os.listdir("./"):
        if f in [f'tmp{bodies}.in', f'tmp{bodies}.out', f'tmp{bodies}.dump',
              f'tmp{bodies}.data', 'log.lammps', lammps_location]:
            os.remove(f)

    clean()
