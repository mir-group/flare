import numpy as np
from numpy import allclose, isclose
import time
import pytest
import os, pickle, re

from flare import struc, env, gp
from flare import otf_parser
from flare.mgp.mgp import MappedGaussianProcess
from flare.lammps import lammps_calculator

from .fake_gp import get_gp, get_random_structure

body_list = ['2', '3']
multi_list = [False, True]
map_force_list = [False, True]

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
        for multihyps in [False, True]:
            gp_model = get_gp(bodies, 'mc', multihyps, cellabc=[100, 100, 100])
            gp_model.parallel = True
            gp_model.n_cpus = 2
            allgp_dict[f'{bodies}{multihyps}'] = gp_model

    yield allgp_dict
    del allgp_dict

@pytest.fixture(scope='module')
def all_mgp():

    allmgp_dict = {}
    for bodies in ['2', '3', '2+3']:
        for multihyps in [False, True]:
            allmgp_dict[f'{bodies}{multihyps}'] = None

    yield allmgp_dict
    del allmgp_dict

@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
@pytest.mark.parametrize('map_force', map_force_list)
def test_init(bodies, multihyps, map_force, all_mgp, all_gp):
    """
    test the init function
    """

    gp_model = all_gp[f'{bodies}{multihyps}']

    grid_num_2 = 64
    grid_num_3 = 25
    lower_cut = 0.01
    two_cut = gp_model.cutoffs['bond']
    three_cut = gp_model.cutoffs['triplet']
    if map_force:
        lower_cut_3 = -1
        three_cut_3 = 1
    else:
        lower_cut_3 = lower_cut
        three_cut_3 = three_cut
    lammps_location = f'{bodies}{multihyps}{map_force}.mgp'

    # set struc params. cell and masses arbitrary?
    mapped_cell = np.eye(3) * 2
    struc_params = {'species': [1, 2],
                    'cube_lat': mapped_cell,
                    'mass_dict': {'0': 27, '1': 16}}

    # grid parameters
    blist = []
    if ('2' in bodies):
        blist += [2]
    if ('3' in bodies):
        blist += [3]
    train_size = len(gp_model.training_data)
    grid_params = {'bodies': blist,
                   'cutoffs':gp_model.cutoffs,
                   'bounds_2': [[lower_cut], [two_cut]],
                   'bounds_3': [[lower_cut, lower_cut, lower_cut_3],
                                [three_cut, three_cut, three_cut_3]],
                   'grid_num_2': grid_num_2,
                   'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
                   'svd_rank_2': 14,
                   'svd_rank_3': 14,
                   'load_grid': None,
                   'update': False}

    struc_params = {'species': [1, 2],
                    'cube_lat': np.eye(3)*2,
                    'mass_dict': {'0': 27, '1': 16}}
    mgp_model = MappedGaussianProcess(grid_params, struc_params, n_cpus=4,
                map_force=map_force, lmp_file_name=lammps_location)
    all_mgp[f'{bodies}{multihyps}{map_force}'] = mgp_model



@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
@pytest.mark.parametrize('map_force', map_force_list)
def test_build_map(all_gp, all_mgp, bodies, multihyps, map_force):
    """
    test the mapping for mc_simple kernel
    """
    gp_model = all_gp[f'{bodies}{multihyps}']
    mgp_model = all_mgp[f'{bodies}{multihyps}{map_force}']
    mgp_model.build_map(gp_model)



@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
@pytest.mark.parametrize('map_force', map_force_list)
def test_write_model(all_mgp, bodies, multihyps, map_force):
    """
    test the mapping for mc_simple kernel
    """
    mgp_model = all_mgp[f'{bodies}{multihyps}{map_force}']
    mgp_model.mean_only = True
    mgp_model.write_model(f'my_mgp_{bodies}_{multihyps}_{map_force}')

    mgp_model.write_model(f'my_mgp_{bodies}_{multihyps}_{map_force}', format='pickle')

    # Ensure that user is warned when a non-mean_only
    # model is serialized into a Dictionary
    with pytest.warns(Warning):
        mgp_model.mean_only = False
        mgp_model.as_dict()
        mgp_model.mean_only = True


@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
@pytest.mark.parametrize('map_force', map_force_list)
def test_load_model(all_mgp, bodies, multihyps, map_force):
    """
    test the mapping for mc_simple kernel
    """
    name = f'my_mgp_{bodies}_{multihyps}_{map_force}.json'
    all_mgp[f'{bodies}{multihyps}'] = MappedGaussianProcess.from_file(name)
    os.remove(name)

    name = f'my_mgp_{bodies}_{multihyps}_{map_force}.pickle'
    all_mgp[f'{bodies}{multihyps}'] = MappedGaussianProcess.from_file(name)
    os.remove(name)


@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
@pytest.mark.parametrize('map_force', map_force_list)
def test_predict(all_gp, all_mgp, bodies, multihyps, map_force):
    """
    test the predict for mc_simple kernel
    """
    gp_model = all_gp[f'{bodies}{multihyps}']
    mgp_model = all_mgp[f'{bodies}{multihyps}{map_force}']

    nenv=10
    cell = np.eye(3)
    cutoffs = gp_model.cutoffs
    unique_species = gp_model.training_data[0].species
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    test_envi = env.AtomicEnvironment(struc_test, 1, cutoffs)

    gp_pred_en = gp_model.predict_local_energy(test_envi)
    gp_pred_f = [gp_model.predict(test_envi, d+1)[0] for d in range(3)]
    mgp_pred = mgp_model.predict(test_envi, mean_only=True)

    # check mgp is within 2 meV/A of the gp
    if not map_force:
        assert(np.abs(mgp_pred[3] - gp_pred_en) < 2e-3), \
                f"{bodies} body energy mapping is wrong"
    assert(np.abs(mgp_pred[0][0] - gp_pred_f[0]) < 2e-3), \
            f"{bodies} body mapping is wrong"

    clean()

@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
@pytest.mark.parametrize('map_force', map_force_list)
def test_lmp_predict(all_gp, all_mgp, bodies, multihyps, map_force):
    """
    test the lammps implementation
    """
    prefix = f'tmp{bodies}{multihyps}{map_force}'
    for f in os.listdir("./"):
        if prefix in f:
            os.remove(f)
    clean()

    mgp_model = all_mgp[f'{bodies}{multihyps}{map_force}']
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
    atom_num = 1
    test_envi = env.AtomicEnvironment(struc_test, atom_num, cutoffs)
    atom_types = [1, 2]
    atom_masses = [108, 127]
    atom_species = struc_test.coded_species

    # create data file
    data_file_name = f'{prefix}.data'
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

    if map_force:
        style_string = 'mgpf'
    else:
        style_string = 'mgp'

    coeff_string = f'* * {lammps_location} H He {by} {ty}'
    lammps_executable = os.environ.get('lmp')
    dump_file_name = f'{prefix}.dump'
    input_file_name = f'{prefix}.in'
    output_file_name = f'{prefix}.out'
    input_text = \
        lammps_calculator.generic_lammps_input(data_file_name, style_string,
                                               coeff_string, dump_file_name,
                                               newton=True)
    lammps_calculator.write_text(input_file_name, input_text)

    lammps_calculator.run_lammps(lammps_executable, input_file_name,
                                 output_file_name)

    lammps_forces = lammps_calculator.lammps_parser(dump_file_name)
    mgp_forces = mgp_model.predict(test_envi, mean_only=True)

    # check that lammps agrees with gp to within 1 meV/A
    for i in range(3):
        assert (np.abs(lammps_forces[atom_num, i] - mgp_forces[0][i]) < 1e-3)

    for f in os.listdir("./"):
        if prefix in f:
            os.remove(f)
    clean()
