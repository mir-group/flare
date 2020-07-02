import numpy as np
import time
import pytest
import os, pickle, re, shutil

from flare import struc, env, gp
from flare import otf_parser
from flare.ase.calculator import FLARE_Calculator
from flare.mgp import MappedGaussianProcess
from flare.lammps import lammps_calculator
from flare.utils.element_coder import _Z_to_mass, _element_to_Z
from ase.calculators.lammpsrun import LAMMPS

from fake_gp import get_gp, get_random_structure

curr_path = os.getcwd()
force_block_only = True

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
def gp_model():
    np.random.seed(0)
    # TO DO, should be 2+3 eventually
    gauss = get_gp('2', 'mc', False, cellabc=[1, 1, 1],
                   force_only=force_block_only, noa=5)
    gauss.parallel = True
    gauss.n_cpus = 2
    yield gauss
    del gauss


@pytest.fixture(scope='module')
def mgp_model(gp_model):
    """
    test the init function
    """

    grid_params={}
    if 'twobody' in gp_model.kernels:
        grid_params['twobody']={'grid_num': [64],
                                'lower_bound':[0.1],
                                'svd_rank': 14}
    if 'threebody' in gp_model.kernels:
        grid_params['threebody']={'grid_num': [16]*3,
                                  'lower_bound':[0.1]*3,
                                  'svd_rank': 14}
    species_list = [1, 2, 3]
    lammps_location = f'tmp_lmp.mgp'
    mapped_model = MappedGaussianProcess(grid_params=grid_params, unique_species=species_list, n_cpus=1,
                map_force=False, lmp_file_name=lammps_location, mean_only=True)

    # import flare.mgp.mapxb
    # flare.mgp.mapxb.global_use_grid_kern = False

    mapped_model.build_map(gp_model)

    yield mapped_model
    del mapped_model


@pytest.fixture(scope='module')
def ase_calculator(gp_model, mgp_model):
    """
    test the mapping for mc_simple kernel
    """
    cal = FLARE_Calculator(gp_model, mgp_model, par=False, use_mapping=True)
    yield cal
    del cal


@pytest.fixture(scope='module')
def lmp_calculator(gp_model, mgp_model):

    species = gp_model.training_statistics['species']
    specie_symbol_list = " ".join(species)
    masses=[f"{i} {_Z_to_mass[_element_to_Z[species[i]]]}" for i in range(len(species))]

    # set up input params
    label = 'tmp_lmp'
    by = 'yes' if 'twobody' in gp_model.kernels else 'no'
    ty = 'yes' if 'threebody' in gp_model.kernels else 'no'
    parameters = {'command': os.environ.get('lmp'), # set up executable for ASE
                  'newton': 'off',
                  'pair_style': 'mgp',
                  'pair_coeff': [f'* * {label}.mgp {specie_symbol_list} {by} {ty}'],
                  'mass': masses}
    files = [f'{label}.mgp']

    # create ASE calc
    lmp_calc = LAMMPS(label=f'tmp{label}', keep_tmp_files=True, tmp_dir='./tmp/',
            parameters=parameters, files=files, specorder=species)
    yield lmp_calc
    del lmp_calc


@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
def test_lmp_predict(gp_model, mgp_model, ase_calculator, lmp_calculator):
    """
    test the lammps implementation
    """

    currdir = os.getcwd()

    label = 'tmp_lmp'

    for f in os.listdir("./"):
        if label in f:
            os.remove(f)
        if f in ['log.lammps']:
            os.remove(f)
    clean()

    lammps_location = mgp_model.lmp_file_name

    # lmp file is automatically written now every time MGP is constructed
    mgp_model.write_lmp_file(lammps_location)

    # create test structure
    np.random.seed(1)
    cell = np.diag(np.array([1, 1, 1])) * 4
    nenv = 10
    unique_species = gp_model.training_statistics['species']
    cutoffs = gp_model.cutoffs
    struc_test, f = get_random_structure(cell, unique_species, nenv)

    # build ase atom from struc
    ase_atoms_flare = struc_test.to_ase_atoms()
    ase_atoms_flare.set_calculator(ase_calculator)

    ase_atoms_lmp = struc_test.to_ase_atoms()
    ase_atoms_lmp.set_calculator(lmp_calculator)

    try:
        lmp_en = ase_atoms_lmp.get_potential_energy()
        flare_en = ase_atoms_flare.get_potential_energy()

        lmp_stress = ase_atoms_lmp.get_stress()
        flare_stress = ase_atoms_flare.get_stress()

        lmp_forces = ase_atoms_lmp.get_forces()
        flare_forces = ase_atoms_flare.get_forces()
    except Exception as e:
        os.chdir(currdir)
        print(e)
        raise e

    os.chdir(currdir)

    # check that lammps agrees with mgp to within 1 meV/A
    print("energy", lmp_en, flare_en)
    assert np.isclose(lmp_en, flare_en, atol=1e-3).all()
    print("force", lmp_forces, flare_forces)
    assert np.isclose(lmp_forces, flare_forces, atol=1e-3).all()
    print("stress", lmp_stress, flare_stress)
    assert np.isclose(lmp_stress, flare_stress, atol=1e-3).all()

    # for f in os.listdir('./'):
    #     if (label in f) or (f in ['log.lammps']):
    #         os.remove(f)

