import numpy as np
import os
import pickle
import pytest
import re
import time
import shutil

from copy import deepcopy
from numpy import allclose, isclose

from flare import struc, env, gp
from flare.parameters import Parameters
from flare.mgp import MappedGaussianProcess
from flare.lammps import lammps_calculator
from flare.utils.element_coder import _Z_to_mass, _Z_to_element

from .fake_gp import get_gp, get_random_structure
from .mgp_test import clean, compare_triplet, predict_atom_diag_var

body_list = ['2', '3']
multi_list = [False, True]
force_block_only = False


@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
@pytest.fixture(scope='module')
def all_gp():

    allgp_dict = {}
    np.random.seed(123)
    for bodies in body_list:
        for multihyps in multi_list:
            gp_model = get_gp(bodies, 'mc', multihyps, cellabc=[1.5, 1, 2],
                              force_only=force_block_only, noa=5)
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
def test_init(bodies, multihyps, all_mgp, all_gp):
    """
    test the init function
    """

    clean()

    gp_model = all_gp[f'{bodies}{multihyps}']

    # grid parameters
    grid_params = {}
    if ('2' in bodies):
        grid_params['twobody'] = {'grid_num': [128], 'lower_bound': [0.01]}
    if ('3' in bodies):
        grid_params['threebody'] = {'grid_num': [31, 32, 33], 'lower_bound':[0.02]*3}

    lammps_location = f'{bodies}{multihyps}'
    data = gp_model.training_statistics

    try:       
        mgp_model = MappedGaussianProcess(grid_params=grid_params, 
            unique_species=data['species'], n_cpus=1, 
            lmp_file_name=lammps_location, var_map='simple')
    except:
        mgp_model = MappedGaussianProcess(grid_params=grid_params, 
            unique_species=data['species'], n_cpus=1, 
            lmp_file_name=lammps_location, var_map=None)
       
    all_mgp[f'{bodies}{multihyps}'] = mgp_model



@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_build_map(all_gp, all_mgp, bodies, multihyps):
    """
    test the mapping for mc_simple kernel
    """
    gp_model = all_gp[f'{bodies}{multihyps}']
    mgp_model = all_mgp[f'{bodies}{multihyps}']
    mgp_model.build_map(gp_model)
#    with open(f'grid_{bodies}_{multihyps}.pickle', 'wb') as f:
#        pickle.dump(mgp_model, f)


@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_write_model(all_mgp, bodies, multihyps):
    """
    test the mapping for mc_simple kernel
    """
    mgp_model = all_mgp[f'{bodies}{multihyps}']
    mgp_model.write_model(f'my_mgp_{bodies}_{multihyps}')

    mgp_model.write_model(f'my_mgp_{bodies}_{multihyps}', format='pickle')

    # Ensure that user is warned when a non-mean_only
    # model is serialized into a Dictionary
    with pytest.warns(Warning):
        mgp_model.var_map = 'pca'
        mgp_model.as_dict()
    
    mgp_model.var_map = 'simple'
    mgp_model.as_dict()


@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_load_model(all_mgp, bodies, multihyps):
    """
    test the mapping for mc_simple kernel
    """
    name = f'my_mgp_{bodies}_{multihyps}.json'
    all_mgp[f'{bodies}{multihyps}'] = MappedGaussianProcess.from_file(name)
    os.remove(name)

    name = f'my_mgp_{bodies}_{multihyps}.pickle'
    all_mgp[f'{bodies}{multihyps}'] = MappedGaussianProcess.from_file(name)
    os.remove(name)

@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_cubic_spline(all_gp, all_mgp, bodies, multihyps):
    """
    test the predict for mc_simple kernel
    """

    mgp_model = all_mgp[f'{bodies}{multihyps}']
    delta = 1e-4

    if '3' in bodies:
        body_name = 'threebody'
    elif '2' in bodies:
        body_name = 'twobody'

    nmap = len(mgp_model.maps[body_name].maps)
    print('nmap', nmap)
    for i in range(nmap):
        maxvalue = np.max(np.abs(mgp_model.maps[body_name].maps[i].mean.__coeffs__))
        if maxvalue >0:
            comp_code = mgp_model.maps[body_name].maps[i].species_code

            if '3' in bodies:

                c_pt = np.array([[0.3, 0.4, 0.5]])
                c, cderv = mgp_model.maps[body_name].maps[i].mean(c_pt, with_derivatives=True)
                cderv = cderv.reshape([-1])

                for j in range(3):
                    a_pt = deepcopy(c_pt)
                    b_pt = deepcopy(c_pt)
                    a_pt[0][j]+=delta
                    b_pt[0][j]-=delta
                    a = mgp_model.maps[body_name].maps[i].mean(a_pt)[0]
                    b = mgp_model.maps[body_name].maps[i].mean(b_pt)[0]
                    num_derv = (a-b)/(2*delta)
                    print("spline", comp_code, num_derv, cderv[j])
                    assert np.isclose(num_derv, cderv[j], rtol=1e-2)

            elif '2' in bodies:
                center = np.sum(mgp_model.maps[body_name].maps[i].bounds)/2.
                a_pt = np.array([[center+delta]])
                b_pt = np.array([[center-delta]])
                c_pt = np.array([[center]])
                a = mgp_model.maps[body_name].maps[i].mean(a_pt)[0]
                b = mgp_model.maps[body_name].maps[i].mean(b_pt)[0]
                c, cderv = mgp_model.maps[body_name].maps[i].mean(c_pt, with_derivatives=True)
                cderv = cderv.reshape([-1])[0]
                num_derv = (a-b)/(2*delta)
                print("spline", num_derv, cderv)
                assert np.isclose(num_derv, cderv, rtol=1e-2)


@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_predict(all_gp, all_mgp, bodies, multihyps):
    """
    test the predict for mc_simple kernel
    """

    gp_model = all_gp[f'{bodies}{multihyps}']
    mgp_model = all_mgp[f'{bodies}{multihyps}']

    # # debug
    # filename = f'grid_{bodies}_{multihyps}.pickle'
    # with open(filename, 'rb') as f:
    #     mgp_model = pickle.load(f)

    nenv = 8
    cell = 1.0 * np.eye(3)
    cutoffs = gp_model.cutoffs
    unique_species = gp_model.training_statistics['species']
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    test_envi = env.AtomicEnvironment(struc_test, 0, cutoffs, cutoffs_mask=gp_model.hyps_mask)

    if '2' in bodies:
        kernel_name = 'twobody'
    elif '3' in bodies:
        kernel_name = 'threebody'
        #compare_triplet(mgp_model.maps['threebody'], gp_model, test_envi)

    assert Parameters.compare_dict(gp_model.hyps_mask, mgp_model.maps[kernel_name].hyps_mask)

    gp_pred_en, gp_pred_envar = gp_model.predict_local_energy_and_var(test_envi)
    gp_pred = np.array([gp_model.predict(test_envi, d+1) for d in range(3)]).T
    print('mgp pred')
    mgp_pred = mgp_model.predict(test_envi)


    # check mgp is within 2 meV/A of the gp
    map_str = 'energy'
    gp_pred_var = gp_pred_envar
    print('mgp_en, gp_en', mgp_pred[3], gp_pred_en)
    assert(np.allclose(mgp_pred[3], gp_pred_en, rtol=2e-3), \
            f"{bodies} body {map_str} mapping is wrong")

#    if multihyps and ('3' in bodies):
#        pytest.skip()

    print('mgp_pred', mgp_pred[0])
    print('gp_pred', gp_pred[0])

    print("isclose?", mgp_pred[0]-gp_pred[0], gp_pred[0])
    assert(np.allclose(mgp_pred[0], gp_pred[0], atol=1e-3)), \
            f"{bodies} body {map_str} mapping is wrong"


    if mgp_model.var_map == 'simple':
        print(bodies, multihyps)
        for i in range(struc_test.nat):
            test_envi = env.AtomicEnvironment(struc_test, i, cutoffs, cutoffs_mask=gp_model.hyps_mask)
            mgp_pred = mgp_model.predict(test_envi)
            mgp_var = mgp_pred[1]
            gp_var = predict_atom_diag_var(test_envi, gp_model, kernel_name)
            print('mgp_var, gp_var', mgp_var, gp_var)
            assert np.allclose(mgp_var, gp_var, rtol=1e-2)

    print('struc_test positions', struc_test.positions, struc_test.species_labels)


@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
def test_lmp_predict(all_gp, all_mgp, bodies, multihyps):
    """
    test the lammps implementation
    """

    pytest.skip()

    prefix = f'{bodies}{multihyps}'

    mgp_model = all_mgp[f'{bodies}{multihyps}']
    gp_model = all_gp[f'{bodies}{multihyps}']
    lammps_location = mgp_model.lmp_file_name

    # create test structure
    cell = 5*np.eye(3)
    nenv = 10
    unique_species = gp_model.training_data[0].species
    cutoffs = gp_model.cutoffs
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    atom_num = 1
    test_envi = env.AtomicEnvironment(struc_test, atom_num, cutoffs, cutoffs_mask=gp_model.hyps_mask)

    all_species=list(set(struc_test.coded_species))
    atom_types = list(np.arange(len(all_species))+1)
    atom_masses=[_Z_to_mass[spec] for spec in all_species]
    atom_species = [ all_species.index(spec)+1 for spec in struc_test.coded_species]
    specie_symbol_list = " ".join([_Z_to_element[spec] for spec in all_species])

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

    style_string = 'mgp'

    coeff_string = f'* * {lammps_location}.mgp {specie_symbol_list} {by} {ty}'
    std_string = f'{lammps_location}.var {specie_symbol_list} {by} {ty}'
    lammps_executable = os.environ.get('lmp')
    dump_file_name = f'{prefix}.dump'
    input_file_name = f'{prefix}.in'
    output_file_name = f'{prefix}.out'
    input_text = \
        lammps_calculator.generic_lammps_input(data_file_name, style_string,
                                               coeff_string, dump_file_name,
                                               newton=False, std_string=std_string)
    lammps_calculator.write_text(input_file_name, input_text)

    lammps_calculator.run_lammps(lammps_executable, input_file_name,
                                 output_file_name)

    pred_std = True
    lammps_forces, lammps_stds = lammps_calculator.lammps_parser(dump_file_name, std=pred_std)
    mgp_pred = mgp_model.predict(test_envi)

    # check that lammps agrees with gp to within 1 meV/A
    for i in range(3):
        print("isclose? diff:", lammps_forces[atom_num, i]-mgp_pred[0][i], "mgp value", mgp_pred[0][i])
        assert np.isclose(lammps_forces[atom_num, i], mgp_pred[0][i], rtol=1e-2)

    # check the lmp var
    mgp_std = np.sqrt(mgp_pred[1])
    print("isclose? diff:", lammps_stds[atom_num]-mgp_std, "mgp value", mgp_std)
    assert np.isclose(lammps_stds[atom_num], mgp_std, rtol=1e-2)

    clean(prefix=prefix)
