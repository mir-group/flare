import numpy as np
import time
import pytest
import os, pickle, re, shutil

from flare import struc, env, gp
from flare import otf_parser
from flare.mgp.mgp import MappedGaussianProcess
from flare.lammps import lammps_calculator
from flare.ase.calculator import FLARE_Calculator
from ase.calculators.lammpsrun import LAMMPS

from tests.fake_gp import get_gp, get_random_structure

body_list = ['2', '3']
multi_list = [False, True]
curr_path = os.getcwd()

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
            gp_model = get_gp(bodies, 'mc', multihyps)
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

@pytest.fixture(scope='module')
def all_ase_calc():

    all_ase_calc_dict = {}
    for bodies in ['2', '3', '2+3']:
        for multihyps in [False, True]:
            all_ase_calc_dict[f'{bodies}{multihyps}'] = None

    yield all_ase_calc_dict
    del all_ase_calc_dict

@pytest.fixture(scope='module')
def all_lmp_calc():

    if 'tmp' not in  os.listdir("./"):
        os.mkdir('tmp')

    all_lmp_calc_dict = {}
    for bodies in ['2', '3', '2+3']:
        for multihyps in [False, True]:
            all_lmp_calc_dict[f'{bodies}{multihyps}'] = None

    yield all_lmp_calc_dict
    del all_lmp_calc_dict


@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_init(bodies, multihyps, all_mgp, all_gp):
    """
    test the init function
    """

    gp_model = all_gp[f'{bodies}{multihyps}']

    grid_num_2 = 64
    grid_num_3 = 16
    lower_cut = 0.01
    two_cut = gp_model.cutoffs['bond']
    three_cut = gp_model.cutoffs['triplet']
    lammps_location = f'{bodies}{multihyps}.mgp'

    # set struc params. cell and masses arbitrary?
    mapped_cell = np.eye(3) * 20
    struc_params = {'species': [1, 2],
                    'cube_lat': mapped_cell,
                    'mass_dict': {'0': 2, '1': 4}}

    # grid parameters
    blist = []
    if ('2' in bodies):
        blist+= [2]
    if ('3' in bodies):
        blist+= [3]
    train_size = len(gp_model.training_data)
    grid_params = {'bodies': blist,
                   'cutoffs':gp_model.cutoffs,
                   'bounds_2': [[lower_cut], [two_cut]],
                   'bounds_3': [[lower_cut, lower_cut, lower_cut],
                                [three_cut, three_cut, three_cut]],
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
                mean_only=True, lmp_file_name=lammps_location)
    all_mgp[f'{bodies}{multihyps}'] = mgp_model


@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_build_map(all_gp, all_mgp, all_ase_calc, bodies, multihyps):
    """
    test the mapping for mc_simple kernel
    """

    # multihyps = False
    gp_model = all_gp[f'{bodies}{multihyps}']
    mgp_model = all_mgp[f'{bodies}{multihyps}']

    mgp_model.build_map(gp_model)

    all_ase_calc[f'{bodies}{multihyps}'] = FLARE_Calculator(gp_model,
            mgp_model, par=False, use_mapping=True)

    clean()

@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_lmp_calc(bodies, multihyps, all_lmp_calc):

    label = f'{bodies}{multihyps}'
    # set up input params

    by = 'no'
    ty = 'no'
    if '2' in bodies:
        by = 'yes'
    if '3' in bodies:
        ty = 'yes'

    parameters = {'command': os.environ.get('lmp'), # set up executable for ASE
                  'newton': 'off',
                  'pair_style': 'mgp',
                  'pair_coeff': [f'* * {label}.mgp H He {by} {ty}'],
                  'mass': ['1 2', '2 4']}
    files = [f'{label}.mgp']


    # create ASE calc
    lmp_calc = LAMMPS(label=f'tmp{label}', keep_tmp_files=True, tmp_dir='./tmp/',
            parameters=parameters, files=files)

    all_lmp_calc[label] = lmp_calc


@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_lmp_predict(all_ase_calc, all_lmp_calc, bodies, multihyps):
    """
    test the lammps implementation
    """

    label = f'{bodies}{multihyps}'

    for f in os.listdir("./"):
        if label in f:
            os.remove(f)
        if f in ['log.lammps']:
            os.remove(f)
    clean()

    flare_calc = all_ase_calc[label]
    lmp_calc = all_lmp_calc[label]

    gp_model = flare_calc.gp_model
    mgp_model = flare_calc.mgp_model
    lammps_location = mgp_model.lmp_file_name

    # lmp file is automatically written now every time MGP is constructed
    mgp_model.write_lmp_file(lammps_location)

    # create test structure
    cell = np.diag(np.array([1, 1, 1.5])) * 4
    nenv = 10
    unique_species = gp_model.training_data[0].species
    cutoffs = gp_model.cutoffs
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    struc_test.positions *= 4

    # build ase atom from struc
    ase_atoms_flare = struc_test.to_ase_atoms()
    ase_atoms_flare.set_calculator(flare_calc)
    ase_atoms_lmp = struc_test.to_ase_atoms()
    ase_atoms_lmp.set_calculator(lmp_calc)

    lmp_en = ase_atoms_lmp.get_potential_energy()
    flare_en = ase_atoms_flare.get_potential_energy()

    lmp_stress = ase_atoms_lmp.get_stress()
    flare_stress = ase_atoms_flare.get_stress()

    lmp_forces = ase_atoms_lmp.get_forces()
    flare_forces = ase_atoms_flare.get_forces()

    # check that lammps agrees with gp to within 1 meV/A
    assert np.all(np.abs(lmp_en - flare_en) < 1e-4)
    assert np.all(np.abs(lmp_forces - flare_forces) < 1e-4)
    assert np.all(np.abs(lmp_stress - flare_stress) < 1e-3)

    for f in os.listdir('./'):
        if (label in f) or (f in ['log.lammps']):
            os.remove(f)

