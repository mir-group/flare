import numpy as np
import time
import pytest
import os, pickle, re

from flare import struc, env, gp
from flare import otf_parser
from flare.mgp.mgp_en import MappedGaussianProcess
from flare.kernels.utils import str_to_kernel_set
from flare.lammps import lammps_calculator

from .fake_gp import get_gp, get_random_structure

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
    for bodies in [2, 3]:
        for multihyps in [False, True]:
            gp_model = get_gp(bodies, 'mc', multihyps)
            gp_model.par = True
            gp_model.n_cpus = 2
            gp_model.set_L_alpha()
            allgp_dict[f'{bodies}{multihyps}'] = gp_model

    yield allgp_dict
    del allgp_dict

@pytest.fixture(scope='module')
def all_mgp():

    allmgp_dict = {}
    for bodies in [2, 3]:
        for multihyps in [False, True]:
            allmgp_dict[f'{bodies}{multihyps}'] = None

    yield allmgp_dict
    del allmgp_dict

@pytest.mark.parametrize('bodies', [ 2, 3])
@pytest.mark.parametrize('multihyps', [False, True])
def test_init(bodies, multihyps, all_mgp, all_gp):
    """
    test the init function
    """

    gp_model = all_gp[f'{bodies}{multihyps}']

    grid_num_2 = 128
    grid_num_3 = 40
    lower_cut = 0.01
    two_cut = gp_model.cutoffs[0]
    three_cut = gp_model.cutoffs[1]
    lammps_location = f'{bodies}{multihyps}.mgp'

    # set struc params. cell and masses arbitrary?
    mapped_cell = np.eye(3) * 2
    struc_params = {'species': [1, 2],
                    'cube_lat': mapped_cell,
                    'mass_dict': {'0': 27, '1': 16}}

    # grid parameters
    train_size = len(gp_model.training_data)
    grid_params = {'bodies': [bodies],
                   'cutoffs':gp_model.cutoffs,
                   'bounds_2': [[lower_cut], [two_cut]],
                   'bounds_3': [[lower_cut, lower_cut, lower_cut],
                                [three_cut, three_cut, three_cut]],
                   'grid_num_2': grid_num_2,
                   'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
                   'svd_rank_2': np.min((grid_num_2, 3*train_size)),
                   'svd_rank_3': np.min((grid_num_3**3, 3*train_size)),
                   'load_grid': None,
                   'update': False}

    struc_params = {'species': [1, 2],
                    'cube_lat': np.eye(3)*2,
                    'mass_dict': {'0': 27, '1': 16}}
    mgp_model = MappedGaussianProcess(grid_params, struc_params, n_cpus=4)
    all_mgp[f'{bodies}{multihyps}'] = mgp_model


@pytest.mark.parametrize('bodies', [ 2, 3])
@pytest.mark.parametrize('multihyps', [False, True])
def test_build_map(all_gp, all_mgp, bodies, multihyps):
    """
    test the mapping for mc_simple kernel
    """

    # multihyps = False
    gp_model = all_gp[f'{bodies}{multihyps}']
    mgp_model = all_mgp[f'{bodies}{multihyps}']

    mgp_model.build_map(gp_model)

    for f in os.listdir("./"):
        if re.search("grid3*.npy", f):
            os.remove(f)
        if re.search("kv3*", f):
            os.rmdir(f)

@pytest.mark.parametrize('bodies', [ 2, 3])
@pytest.mark.parametrize('multihyps', [False, True])
def test_predict(all_gp, all_mgp, bodies, multihyps):
    """
    test the predict for mc_simple kernel
    """

    # multihyps = False
    gp_model = all_gp[f'{bodies}{multihyps}']
    mgp_model = all_mgp[f'{bodies}{multihyps}']

    nenv=10
    cell = np.eye(3)
    cutoffs = gp_model.cutoffs
    unique_species = gp_model.training_data[0].species
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    test_envi = env.AtomicEnvironment(struc_test, 1, cutoffs)

    gp_pred_en = gp_model.predict_local_energy(test_envi)
    gp_pred_x = gp_model.predict(test_envi, 1)
    mgp_pred = mgp_model.predict(test_envi, mean_only=True)

    # check mgp is within 1 meV/A of the gp
    assert(np.abs(mgp_pred[3] - gp_pred_en) < 1e-3), \
            f"{bodies} body energy mapping is wrong"
    assert(np.abs(mgp_pred[0][0] - gp_pred_x[0]) < 1e-3), \
            f"{bodies} body mapping is wrong"

    for f in os.listdir("./"):
        if re.search("grid3*.npy", f):
            os.remove(f)
        if re.search("kv3*", f):
            os.rmdir(f)


@pytest.mark.parametrize('bodies', [ 2, 3])
@pytest.mark.parametrize('multihyps', [False, True])
def test_lmp_predict(all_gp, all_mgp, bodies, multihyps):
    """
    test the lammps implementation
    """

    mgp_model = all_mgp[f'{bodies}{multihyps}']
    gp_model = all_gp[f'{bodies}{multihyps}']
    lammps_location = mgp_model.lmp_file_name

    # lmp file is automatically written now every time MGP is constructed
    mgp_model.write_lmp_file(lammps_location)

    # create test structure
    cell = np.eye(3)
    nenv = 10
    unique_species = gp_model.training_data[0].species
    cutoffs = gp_model.cutoffs
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    test_envi = env.AtomicEnvironment(struc_test, 1, cutoffs)
    atom_types = [1, 2]
    atom_masses = [108, 127]
    atom_species = struc_test.coded_species

    # create data file
    data_file_name = f'tmp{bodies}{multihyps}.data'
    data_text = lammps_calculator.lammps_dat(struc_test, atom_types,
                                             atom_masses, atom_species)
    lammps_calculator.write_text(data_file_name, data_text)

    # create lammps input
    if bodies == 2:
        by = 'yes'
        ty = 'no'
    else:
        by = 'no'
        ty = 'yes'
    style_string = 'mgp'
    coeff_string = f'* * {lammps_location} 1 2 {by} {ty}'
    lammps_executable = os.environ.get('lmp')
    dump_file_name = f'tmp{bodies}{multihyps}.dump'
    input_file_name = f'tmp{bodies}{multihyps}.in'
    output_file_name = f'tmp{bodies}{multihyps}.out'
    input_text = \
        lammps_calculator.generic_lammps_input(data_file_name, style_string,
                                               coeff_string, dump_file_name)
    lammps_calculator.write_text(input_file_name, input_text)

    lammps_calculator.run_lammps(lammps_executable, input_file_name,
                                 output_file_name)

    lammps_forces = lammps_calculator.lammps_parser(dump_file_name)
    mgp_forces = mgp_model.predict(test_envi, mean_only=True)

    # check that lammps agrees with gp to within 1 meV/A
    assert (np.abs(lammps_forces[0, 1] - mgp_forces[0][1]) < 1e-3)

    for f in os.listdir("./"):
        if f in [f'tmp{bodies}{multihyps}in', f'tmp{bodies}{multihyps}out', f'tmp{bodies}{multihyps}dump',
              f'tmp{bodies}{multihyps}data', 'log.lammps', lammps_location]:
            os.remove(f)
        if re.search("grid3*.npy", f):
            os.remove(f)
        if re.search("kv3*", f):
            os.rmdir(f)
