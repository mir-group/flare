import numpy as np
import os
import pickle
import pytest
import re
import time

from copy import deepcopy
from numpy import allclose, isclose

from flare import struc, env, gp
from flare.parameters import Parameters
from flare.mgp import MappedGaussianProcess
from flare.lammps import lammps_calculator
from flare.utils.element_coder import _Z_to_mass, _Z_to_element

from .fake_gp import generate_hm 

body_list = ['3'] #['2', '3']
multi_list = [False] #[False, True]
map_force_list = [False] #[False, True]
force_block_only = True

def clean():
    for f in os.listdir("./"):
        if re.search(r"grid.*npy", f):
            os.remove(f)
        if re.search("kv3", f):
            os.rmdir(f)


def get_gp(bodies, kernel_type='mc', multihyps=True, cellabc=[1, 1, 1.5],
           force_only=False, noa=5):
    print("Setting up...")

    # params
    cell = np.diag(cellabc)

    ntwobody = 0
    nthreebody = 0
    prefix = bodies
    if ('2' in bodies or 'two' in bodies):
        ntwobody = 1
    if ('3' in bodies or 'three' in bodies):
        nthreebody = 1

    hyps, hm, _ = generate_hm(ntwobody, nthreebody, nmanybody=0, multihyps=multihyps)
    cutoffs = hm['cutoffs']
    kernels = hm['kernels']
    hl = hm['hyp_labels']

    # create test structure
    perturb = np.random.rand(3, 3) * 0.1
    positions = np.array([[0, 0, 0],
                          [0.3, 0, 0],
                          [0, 0.4, 0]])
    positions += perturb
    print('perturbed positions', positions)
    species = [1, 2, 2]
    test_structure = struc.Structure(cell, species, positions)
    forces = np.array([[0.1, 2.3, 0.45],
                       [0.6, 0.07, 0.0],
                       [0.89, 1.0, 1.1]])

    energy = 3.14

    # test update_db
    gaussian = \
        gp.GaussianProcess(kernels=kernels,
                        component=kernel_type,
                        hyps=hyps,
                        hyp_labels=hl,
                        cutoffs=cutoffs, hyps_mask=hm,
                        parallel=False, n_cpus=1)
    if force_only:
        gaussian.update_db(test_structure, forces)
    else:
        gaussian.update_db(test_structure, forces, energy=energy)
    gaussian.check_L_alpha()

    #print(gaussian.alpha)

    return gaussian




@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')
@pytest.fixture(scope='module')
def all_gp():

    allgp_dict = {}
    np.random.seed(0)
    for bodies in body_list:
        for multihyps in multi_list:
            gp_model = get_gp(bodies, 'mc', multihyps, cellabc=[1.5, 1, 2],
                              force_only=force_block_only, noa=5) #int(bodies)**2)
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

    # grid parameters
    grid_num_2 = 128
    grid_num_3 = 16
    grid_params = {}
    if ('2' in bodies):
        grid_params['twobody'] = {'grid_num': [grid_num_2]}# 'lower_bound': [0.05]}
    if ('3' in bodies):
        grid_params['threebody'] = {'grid_num': [grid_num_3]*3, 'lower_bound':[0.1]*3}

    lammps_location = f'{bodies}{multihyps}{map_force}.mgp'
    data = gp_model.training_statistics

    mgp_model = MappedGaussianProcess(grid_params, data['species'], n_cpus=1,
                map_force=map_force, lmp_file_name=lammps_location)#, mean_only=False)
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
    with open(f'grid_{bodies}_{multihyps}_{map_force}.pickle', 'wb') as f:
        pickle.dump(mgp_model, f)



def compare_triplet(mgp_model, gp_model, atom_env):
    spcs, comp_r, comp_xyz = mgp_model.get_arrays(atom_env)
    for i, spc in enumerate(spcs):
        lengths = np.array(comp_r[i])
        xyzs = np.array(comp_xyz[i])

        print('compare triplet spc, lengths, xyz', spc)
        print(np.hstack([lengths, xyzs]))

        gp_f = []
        grid_env = get_grid_env(gp_model, spc, 3)
        for l in range(lengths.shape[0]):
            r1, r2, r12 = lengths[l, :]
            grid_env = get_triplet_env(r1, r2, r12, grid_env)
            gp_pred = np.array([gp_model.predict(grid_env, d+1) for d in range(3)]).T
            gp_f.append(gp_pred[0])
        gp_force = np.sum(gp_f, axis=0)
        print('gp_f')
        print(gp_f)

        map_ind = mgp_model.find_map_index(spc)
        xyzs = np.zeros_like(xyzs)
        xyzs[:, 0] = np.ones_like(xyzs[:, 0])
        f, vir, v, e = mgp_model.maps[map_ind].predict(lengths, xyzs,
            mgp_model.map_force, mean_only=True)

        assert np.allclose(gp_force, f, rtol=1e-2)
        

def get_triplet_env(r1, r2, r12, grid_env):
    grid_env.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
    grid_env.cross_bond_dists = np.array([[0, r12], [r12, 0]])

    return grid_env


def get_grid_env(GP, species, bodies):
    if isinstance(GP.cutoffs, dict):
        max_cut = np.max(list(GP.cutoffs.values()))
    else:
        max_cut = np.max(GP.cutoffs)
    big_cell = np.eye(3) * (2 * max_cut + 1)
    positions = [[(i+1)/(bodies+1)*0.1, 0, 0]
                 for i in range(bodies)]
    grid_struc = struc.Structure(big_cell, species, positions)
    grid_env = env.AtomicEnvironment(grid_struc, 0, GP.cutoffs,
        cutoffs_mask=GP.hyps_mask)

    return grid_env




@pytest.mark.parametrize('bodies', body_list)
@pytest.mark.parametrize('multihyps', multi_list)
@pytest.mark.parametrize('map_force', map_force_list)
def test_predict(all_gp, all_mgp, bodies, multihyps, map_force):
    """
    test the predict for mc_simple kernel
    """

    gp_model = all_gp[f'{bodies}{multihyps}']
    mgp_model = all_mgp[f'{bodies}{multihyps}{map_force}']

    # test on training data
    test_env = gp_model.training_data[1]
    assert test_env.bond_array_2[0][0] >= mgp_model.maps['threebody'].maps[0].bounds[0][0]

    compare_triplet(mgp_model.maps['threebody'], gp_model, test_env)

    gp_pred = np.array([gp_model.predict(test_env, d+1) for d in range(3)]).T
    mgp_pred = mgp_model.predict(test_env, mean_only=True)

    print('mgp_pred', mgp_pred[0])
    print('gp_pred', gp_pred[0])

    print("isclose?", mgp_pred[0]-gp_pred[0])
    assert(np.allclose(mgp_pred[0], gp_pred[0], atol=5e-3)) 


#    # create test structure
#    np.random.seed(0)
#    positions = np.random.rand(4, 3)
#    species = [2, 1, 2, 1]
#    cell = np.diag([1.5, 1.5, 1.5])
#    test_structure = struc.Structure(cell, species, positions)
#    test_env = env.AtomicEnvironment(test_structure, 0, gp_model.cutoffs)
#    print('test positions', positions)
#    assert test_env.bond_array_2[0][0] >= mgp_model.maps['threebody'].maps[0].bounds[0][0]
#    
#    compare_triplet(mgp_model.maps['threebody'], gp_model, test_env)
#
#    gp_pred = np.array([gp_model.predict(test_env, d+1) for d in range(3)]).T
#    mgp_pred = mgp_model.predict(test_env, mean_only=True)
#
#    print('mgp_pred', mgp_pred[0])
#    print('gp_pred', gp_pred[0])
#
#    print("isclose?", mgp_pred[0]-gp_pred[0])
#    assert(np.allclose(mgp_pred[0], gp_pred[0], atol=5e-3)) 
#


