import pytest
import numpy as np
from numpy.random import random, randint, permutation

import flare.kernels.mc_sephyps as en

from flare import env, struc, gp
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare import mc_simple
from flare.otf_parser import OtfAnalysis
from flare.kernels.mc_sephyps import _str_to_kernel as stk

from .fake_gp import get_random_structure, generate_hm, \
                    get_gp, get_params, get_tstp

# ------------------------------------------------------
#          fixtures
# ------------------------------------------------------


# set the scope to module so it will only be setup once
@pytest.fixture(scope='module')
def two_body_gp() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""

    gaussian = get_gp([2], 'mc', True)

    yield gaussian
    del gaussian

@pytest.fixture(scope='module')
def three_body_gp() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""

    gaussian = get_gp([3], 'mc', True)

    yield gaussian
    del gaussian

# set the scope to module so it will only be setup once
@pytest.fixture(scope='module')
def two_plus_three_gp() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""

    gaussian = get_gp([2, 3], 'mc', True)

    yield gaussian
    del gaussian


@pytest.fixture(scope='module')
def params():
    parameters = get_params()
    yield parameters
    del parameters


@pytest.fixture(scope='module')
def test_point() -> AtomicEnvironment:
    """Create test point for kernel to compare against"""
    # params
    test_pt = get_tstp()
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



def test_train(two_body_gp, params):
    hyp = list(two_body_gp.hyps)

    # add struc and forces to db
    test_structure, forces = get_random_structure(params['cell'],
                                                  params['unique_species'],
                                                  params['noa'])
    two_body_gp.update_db(test_structure, forces)

    # train gp
    res = two_body_gp.train()

    hyp_post = list(two_body_gp.hyps)
    print(res)

    # check if hyperparams have been updated
    assert (hyp != hyp_post)


def test_predict(two_body_gp, test_point):
    pred = two_body_gp.predict(x_t=test_point, d=1)
    assert (len(pred) == 2)
    assert (isinstance(pred[0], float))
    assert (isinstance(pred[1], float))


def test_set_L_alpha(two_plus_three_gp):
    two_plus_three_gp.set_L_alpha()


def test_update_L_alpha(two_plus_three_gp):
    two_plus_three_gp.set_L_alpha()
    # update database & use update_L_alpha to get ky_mat
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = np.array([0.8, 0.8])
    noa = 5
    for n in range(3):
        positions = random([noa, 3])
        forces = random([noa, 3])
        species = randint(2, size=noa)
        atoms = list(permutation(noa)[:2])

        struc_curr = Structure(cell, species, positions)
        two_plus_three_gp.update_db(struc_curr, forces, custom_range=atoms)
        two_plus_three_gp.update_L_alpha()

    ky_mat_from_update = np.copy(two_plus_three_gp.ky_mat)

    # use set_L_alpha to get ky_mat
    two_plus_three_gp.set_L_alpha()
    ky_mat_from_set = np.copy(two_plus_three_gp.ky_mat)

    assert (np.all(np.absolute(ky_mat_from_update - ky_mat_from_set)) < 1e-6)


def test_representation_method(two_body_gp):
    the_str = str(two_body_gp)
    assert 'GaussianProcess Object' in the_str
    assert 'Kernel: 2mc' in the_str
    assert 'Cutoffs: [0.8 0.8]' in the_str
    assert 'Model Likelihood: ' in the_str
    assert 'ls2: ' in the_str
    assert 'sig2: ' in the_str
    assert "noise: " in the_str


def test_serialization_method(two_body_gp, test_point):
    """
    Serialize and then un-serialize a GP and ensure that no info was lost.
    Compare one calculation to ensure predictions work correctly.
    :param two_body_gp:
    :return:
    """
    old_gp_dict = two_body_gp.as_dict()
    new_gp = GaussianProcess.from_dict(old_gp_dict)
    new_gp_dict = new_gp.as_dict()

    dumpcompare(new_gp_dict, old_gp_dict)

    for d in [0, 1, 2]:
        assert np.all(two_body_gp.predict(x_t=test_point, d=d) ==
                      new_gp.predict(x_t=test_point, d=d))

def dumpcompare(obj1, obj2):
    '''this source code comes from
    http://stackoverflow.com/questions/15785719/how-to-print-a-dictionary-line-by-line-in-python'''

    assert isinstance(obj1, type(obj2)), "the two objects are of different types"

    if isinstance(obj1, dict):

        assert len(obj1.keys()) == len(obj2.keys())

        for k1, k2 in zip(sorted(obj1.keys()), sorted(obj2.keys())):

            assert k1==k2, f"key {k1} is not the same as {k2}"
            assert dumpcompare(obj1[k1], obj2[k2]), f"value {k1} is not the same as {k2}"

    elif isinstance(obj1, (list, tuple)):

        assert len(obj1) == len(obj2)
        for k1, k2 in zip(obj1, obj2):
            assert dumpcompare(k1, k2), f"list elements are different"

    elif isinstance(obj1, np.ndarray):

        assert obj1.shape == obj2.shape

        if (not isinstance(obj1[0], np.str_)):
            assert np.equal(obj1, obj2).all(), "ndarray is not all the same"
        else:
            for xx, yy in zip(obj1, obj2):
                assert dumpcompare(xx, yy)
    else:
        assert obj1==obj2

    return True

def test_constrained_optimization_simple():
    """
    Test constrained optimization with a standard
    number of hyperparameters (3 for a 3-body)
    :return:
    """

    # params
    cell = np.eye(3)
    species = [1,1,2,2,2]
    positions = np.random.uniform(0,1,(5,3))
    forces = np.random.uniform(0,1,(5,3))

    two_species_structure = Structure(cell=cell,species=species,
                               positions=positions,
                               forces=forces)


    hyp_labels=['2-Body_sig2,',
                '2-Body_l2',
                '3-Body_sig2',
                '3-Body_l2',
                'noise']
    hyps = np.array([1.2, 2.2, 3.2, 4.2, 12.])
    cutoffs = np.array((.8, .8))

    # Define hyp masks

    spec_mask = np.zeros(118, dtype=int)
    spec_mask[1] = 1

    hyps_mask = {'nspec': 2,
                 'spec_mask': spec_mask,
                 'nbond': 2,
                 'bond_mask': [0, 1, 1, 1],
                 'ntriplet': 2,
                 'triplet_mask': [0, 1, 1, 1, 1, 1, 1, 1],
                 'original': np.array([1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2,
                                    12.]),
                 'train_noise': True,
                 'map': [1, 3, 5, 7, 8]}

    gp = GaussianProcess(kernel_name='23mc',
                        hyps=hyps,
                        hyp_labels=hyp_labels,
                        cutoffs=cutoffs, par=False, n_cpus=1,
                        hyps_mask=hyps_mask,
                        maxiter=1,multihyps=True)


    gp.update_db(two_species_structure,
                 two_species_structure.forces)

    # Check that the hyperparameters were updated
    results = gp.train()
    assert not np.equal(results.x, hyps).all()

    #TODO check that the predictions match up with a 2+3 body
    # kernel without hyperparameter masking
