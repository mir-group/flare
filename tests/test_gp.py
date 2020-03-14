import pytest
import pickle
import os
import json
import numpy as np

from typing import List
from pytest import raises

from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
import flare.kernels.sc as en
import flare.kernels.mc_simple as mc_simple
from flare.otf_parser import OtfAnalysis

from .fake_gp import generate_hm, get_tstp, get_random_structure

multihyps_list = [True, False]


@pytest.fixture(scope='module')
def all_gps() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""

    gp_dict = {True: None, False: None}
    yield gp_dict
    del gp_dict


@pytest.fixture(scope='module')
def params():
    parameters = {'unique_species': [2, 1],
                  'cutoff': 0.8,
                  'noa': 5,
                  'cell': np.eye(3),
                  'db_pts': 30}
    yield parameters
    del parameters


@pytest.mark.parametrize('multihyps', multihyps_list)
def test_init(multihyps, all_gps):
    cutoffs = np.ones(2)*0.8
    hyps, hm, _ = generate_hm(1, 1, multihyps=multihyps)
    hl = hm['hyps_label']
    if (multihyps is False):
        hm = None

    # test update_db
    gpname = '2+3mc'
    if multihyps is False:
        gpname += 'mb'
        hyps = np.hstack([hyps, [1, 1]])
        cutoffs = np.ones(3)*0.8

    all_gps[multihyps] = \
        GaussianProcess(kernel_name=gpname,
                        hyps=hyps,
                        hyp_labels=hl,
                        cutoffs=cutoffs, multihyps=multihyps, hyps_mask=hm,
                        parallel=False, n_cpus=1)


@pytest.fixture(scope='module')
def test_point() -> AtomicEnvironment:
    test_pt = get_tstp()
    yield test_pt
    del test_pt

# ------------------------------------------------------
#                   test GP methods
# ------------------------------------------------------


@pytest.mark.parametrize('multihyps', multihyps_list)
def test_update_db(all_gps, multihyps, params):

    test_gp = all_gps[multihyps]
    oldsize = len(test_gp.training_data)

    # add structure and forces to db
    test_structure, forces = get_random_structure(params['cell'],
                                                  params['unique_species'],
                                                  params['noa'])

    test_gp.update_db(test_structure, forces)

    assert (len(test_gp.training_data) == params['noa']+oldsize)
    assert (len(test_gp.training_labels_np) == (params['noa']+oldsize)*3)


@pytest.mark.parametrize('par, n_cpus', [(True, 2),
                                         (False, 1)])
@pytest.mark.parametrize('multihyps', multihyps_list)
def test_train(all_gps, params, par, n_cpus, multihyps):

    test_gp = all_gps[multihyps]
    test_gp.parallel = par
    test_gp.n_cpus = n_cpus

    test_gp.maxiter = 1

    # train gp
    test_gp.hyps = np.ones(len(test_gp.hyps))
    hyp = list(test_gp.hyps)
    test_gp.train()

    hyp_post = list(test_gp.hyps)

    # check if hyperparams have been updated
    assert (hyp != hyp_post)


@pytest.mark.parametrize('par, per_atom_par, n_cpus',
                         [(False, False, 1),
                          (True, True, 2),
                          (True, False, 2)])
@pytest.mark.parametrize('multihyps', multihyps_list)
def test_predict(all_gps, test_point, par, per_atom_par, n_cpus, multihyps):
    test_gp = all_gps[multihyps]
    test_gp.parallel = par
    test_gp.per_atom_par = per_atom_par
    pred = test_gp.predict(x_t=test_point, d=1)
    assert (len(pred) == 2)
    assert (isinstance(pred[0], float))
    assert (isinstance(pred[1], float))


@pytest.mark.parametrize('par, n_cpus', [(True, 2),
                                         (False, 1)])
@pytest.mark.parametrize('multihyps', multihyps_list)
def test_set_L_alpha(all_gps, params, par, n_cpus, multihyps):
    test_gp = all_gps[multihyps]
    test_gp.parallel = par
    test_gp.n_cpus = n_cpus
    test_gp.set_L_alpha()


@pytest.mark.parametrize('par, n_cpus', [(True, 2),
                                         (False, 1)])
@pytest.mark.parametrize('multihyps', multihyps_list)
def test_update_L_alpha(all_gps, params, par, n_cpus, multihyps):
    # set up gp model
    test_gp = all_gps[multihyps]
    test_gp.parallel = par
    test_gp.n_cpus = n_cpus

    test_structure, forces = get_random_structure(params['cell'],
                                                  params['unique_species'],
                                                  params['noa'])
    test_gp.check_L_alpha()
    test_gp.update_db(test_structure, forces)
    test_gp.update_L_alpha()

    # compare results with set_L_alpha
    ky_mat_from_update = np.copy(test_gp.ky_mat)
    test_gp.set_L_alpha()
    ky_mat_from_set = np.copy(test_gp.ky_mat)

    assert (np.all(np.absolute(ky_mat_from_update - ky_mat_from_set)) < 1e-6)


@pytest.mark.parametrize('multihyps', multihyps_list)
def test_representation_method(all_gps, multihyps):
    test_gp = all_gps[multihyps]
    the_str = str(test_gp)
    assert 'GaussianProcess Object' in the_str
    assert 'Kernel: 2+3mc' in the_str
    if (multihyps):
        assert 'Cutoffs: [0.8 0.8]' in the_str
    else:
        assert 'Cutoffs: [0.8 0.8 0.8]' in the_str
    assert 'Model Likelihood: ' in the_str
    if not multihyps:
        assert 'Length: ' in the_str
        assert 'Signal Var.: ' in the_str
        assert "Noise Var.: " in the_str


@pytest.mark.parametrize('multihyps', multihyps_list)
def test_serialization_method(all_gps, test_point, multihyps):
    """
    Serialize and then un-serialize a GP and ensure that no info was lost.
    Compare one calculation to ensure predictions work correctly.
    :param test_gp:
    :return:
    """
    test_gp = all_gps[multihyps]
    old_gp_dict = test_gp.as_dict()
    new_gp = GaussianProcess.from_dict(old_gp_dict)
    new_gp_dict = new_gp.as_dict()

    assert len(new_gp_dict) == len(old_gp_dict)

    dumpcompare(new_gp_dict, old_gp_dict)

    for d in [0, 1, 2]:
        assert np.all(test_gp.predict(x_t=test_point, d=d) ==
                      new_gp.predict(x_t=test_point, d=d))


@pytest.mark.parametrize('multihyps', multihyps_list)
def test_load_and_reload(all_gps, test_point, multihyps):

    test_gp = all_gps[multihyps]

    test_gp.write_model('test_gp_write', 'pickle')

    with open('test_gp_write.pickle', 'rb') as f:
        new_gp = pickle.load(f)

    for d in [0, 1, 2]:
        assert np.all(test_gp.predict(x_t=test_point, d=d) ==
                      new_gp.predict(x_t=test_point, d=d))
    os.remove('test_gp_write.pickle')

    test_gp.write_model('test_gp_write', 'json')

    with open('test_gp_write.json', 'r') as f:
        new_gp = GaussianProcess.from_dict(json.loads(f.readline()))
    for d in [0, 1, 2]:
        assert np.all(test_gp.predict(x_t=test_point, d=d) ==
                      new_gp.predict(x_t=test_point, d=d))
    os.remove('test_gp_write.json')

    with raises(ValueError):
        test_gp.write_model('test_gp_write', 'cucumber')


def dumpcompare(obj1, obj2):
    '''this source code comes from
    http://stackoverflow.com/questions/15785719/how-to-print-a-dictionary-line-by-line-in-python'''

    assert isinstance(obj1, type(
        obj2)), "the two objects are of different types"

    if isinstance(obj1, dict):

        assert len(obj1.keys()) == len(
            obj2.keys()), f"key1 {list(obj1.keys())}, \n key2 {list(obj2.keys())}"

        for k1, k2 in zip(sorted(obj1.keys()), sorted(obj2.keys())):

            assert k1 == k2, f"key {k1} is not the same as {k2}"
            assert dumpcompare(obj1[k1], obj2[k2]
                               ), f"value {k1} is not the same as {k2}"

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
        assert obj1 == obj2

    return True


def test_constrained_optimization_simple(all_gps):
    """
    Test constrained optimization with a standard
    number of hyperparameters (3 for a 3-body)
    :return:
    """

    test_gp = all_gps[True]
    test_gp.hyp_labels = ['2-Body_sig2,', '2-Body_l2',
                          '3-Body_sig2', '3-Body_l2', 'noise']
    test_gp.hyps = np.array([1.2, 2.2, 3.2, 4.2, 12.])
    init_hyps = np.copy(test_gp.hyps)

    # Define hyp masks
    spec_mask = np.zeros(118, dtype=int)
    spec_mask[1] = 1
    test_gp.hyps_mask = {
        'nspec': 2,
        'spec_mask': spec_mask,
        'nbond': 2,
        'bond_mask': [0, 1, 1, 1],
        'ntriplet': 2,
        'triplet_mask': [0, 1, 1, 1, 1, 1, 1, 1],
        'original': np.array([1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2,
                              12.]),
        'train_noise': True,
        'map': [1, 3, 5, 7, 8]}

    # Check that the hyperparameters were updated
    results = test_gp.train()
    assert not np.equal(results.x, init_hyps).all()
