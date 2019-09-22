import pytest
import numpy as np
import sys
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
import flare.kernels as en
from flare import mc_simple
from flare.otf_parser import OtfAnalysis


def get_random_structure(cell, unique_species, noa):
    """Create a random test structure """
    np.random.seed(0)

    positions = []
    forces = []
    species = []

    for n in range(noa):
        positions.append(np.random.uniform(-1, 1, 3))
        forces.append(np.random.uniform(-1, 1, 3))
        species.append(unique_species[np.random.randint(0,
                                                        len(unique_species))])

    test_structure = Structure(cell, species, positions)

    return test_structure, forces


# ------------------------------------------------------
#          fixtures
# ------------------------------------------------------


# set the scope to module so it will only be setup once
@pytest.fixture(scope='module')
def two_body_gp():
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoff = 0.8
    cutoffs = np.array([0.8, 0.8])
    noa = 5

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)

    # test update_db
    gaussian = \
        GaussianProcess(kernel=en.three_body,
                        kernel_grad=en.three_body_grad,
                        hyps=np.array([1, 1, 1]),
                        cutoffs=cutoffs)
    gaussian.update_db(test_structure, forces)

    # return gaussian
    yield gaussian

    # code after yield will be executed once all tests are run
    # this will not be run if an exception is raised in the setup
    print("\n\nTearing down\n")
    del gaussian


@pytest.fixture(scope='module')
def params():
    parameters = {'unique_species': [2, 1],
                  'cutoff': 0.8,
                  'noa': 5,
                  'cell': np.eye(3),
                  'db_pts': 30}
    yield parameters
    del parameters


@pytest.fixture(scope='module')
def test_point():
    """Create test point for kernel to compare against"""
    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoff = 0.8
    noa = 10

    test_structure_2, _ = get_random_structure(cell, unique_species,
                                               noa)

    test_pt = AtomicEnvironment(test_structure_2, 0,
                                np.array([cutoff, cutoff]))

    yield test_pt
    del test_pt


# ------------------------------------------------------
#                   test GP methods
# ------------------------------------------------------

def test_update_db(two_body_gp, params):
    # params
    test_structure, forces = get_random_structure(params['cell'],
                                                  params['unique_species'],
                                                  params['noa'])

    # add structure and forces to db
    two_body_gp.update_db(test_structure, forces)

    assert (len(two_body_gp.training_data) == params['noa'] * 2)
    assert (len(two_body_gp.training_labels_np) == params['noa'] * 2 * 3)


def test_get_kernel_vector(two_body_gp, test_point, params):
    assert (two_body_gp.get_kernel_vector(test_point, 1).shape ==
            (params['db_pts'],))


def test_train(two_body_gp, params):
    hyp = list(two_body_gp.hyps)

    # add struc and forces to db
    test_structure, forces = get_random_structure(params['cell'],
                                                  params['unique_species'],
                                                  params['noa'])
    two_body_gp.update_db(test_structure, forces)

    # train gp
    two_body_gp.train()

    hyp_post = list(two_body_gp.hyps)

    # check if hyperparams have been updated
    assert (hyp != hyp_post)


def test_predict(two_body_gp, test_point):
    pred = two_body_gp.predict(x_t=test_point, d=1)
    assert (len(pred) == 2)
    assert (isinstance(pred[0], float))
    assert (isinstance(pred[1], float))


def test_set_L_alpha(two_body_gp, params):
    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoff = 0.8
    cutoffs = np.array([0.8, 0.8])
    noa = 2

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)

    # set gp model
    kernel = en.two_plus_three_body
    kernel_grad = en.two_plus_three_body_grad
    hyps = np.array([2.23751151e-01,  8.19990316e-01, 1.28421842e-04,
                    1.07467158e+00, 5.50677932e-02])
    cutoffs = np.array([5.4, 5.4])
    hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
    energy_force_kernel = en.two_plus_three_force_en
    energy_kernel = en.two_plus_three_en
    opt_algorithm = 'BFGS'

    # test update_db
    gaussian = \
        GaussianProcess(kernel, kernel_grad, hyps, cutoffs, hyp_labels,
                        energy_force_kernel, energy_kernel,
                        opt_algorithm)
    gaussian.update_db(test_structure, forces)

    gaussian.set_L_alpha()


def test_update_L_alpha():
    # set up gp model
    kernel = mc_simple.two_plus_three_body_mc
    kernel_grad = mc_simple.two_plus_three_body_mc_grad
    cutoffs = [6.0, 5.0]
    hyps = np.array([0.001770, 0.183868, -0.001415, 0.372588, 0.026315])
    
    # get an otf traj from file for training data
    old_otf = OtfAnalysis('test_files/AgI_snippet.out')
    call_no = 1 
    cell = old_otf.header['cell']
    gp_model = old_otf.make_gp(kernel=kernel,
                               kernel_grad=kernel_grad,
                               call_no=call_no, 
                               cutoffs=cutoffs, 
                               hyps=hyps)
    
    # update database & use update_L_alpha to get ky_mat
    for n in range(call_no, call_no+1): 
        positions = old_otf.gp_position_list[n]
        species = old_otf.gp_species_list[n]
        atoms = old_otf.gp_atom_list[n]
        forces = old_otf.gp_force_list[n]

        struc_curr = Structure(cell, species, positions)
        gp_model.update_db(struc_curr, forces, custom_range=atoms)
        gp_model.update_L_alpha()

    ky_mat_from_update = np.copy(gp_model.ky_mat)

    # use set_L_alpha to get ky_mat
    gp_model.set_L_alpha()
    ky_mat_from_set = np.copy(gp_model.ky_mat)
    
    assert (np.all(np.absolute(ky_mat_from_update-ky_mat_from_set)) < 1e-6)
