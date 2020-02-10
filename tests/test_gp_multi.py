import pytest
import numpy as np
from numpy.random import random, randint, permutation

import flare.gp_algebra
import flare.gp_algebra_multi
import flare.mc_sephyps as en

from flare import env, struc, gp
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare import mc_simple
from flare.otf_parser import OtfAnalysis
from flare.mc_sephyps import _str_to_kernel as stk
import flare.mc_sephyps as en

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


def generate_hm(nbond, ntriplet):

    specs_mask = np.zeros(118, dtype=int)
    specs_mask[1] = 0
    specs_mask[2] = 1
    nspecs = 2

    specs_mask = np.zeros(118, dtype=int)
    specs_mask[1] = 0
    specs_mask[2] = 1
    nspecs = 2

    if (nbond==2):
        sig1 = [1, 1]
        ls1 = [1, 1]
        bond_mask = np.ones(nspecs**2, dtype=int)
        bond_mask[0] = 0
        bond_name = ["sig2"]*2+["ls2"]*2
    elif (nbond==1):
        sig1 = [1]
        ls1 = [1]
        bond_mask = np.zeros(nspecs**2, dtype=int)
        bond_name = ["sig2"]+["ls2"]
    else:
        bond_mask = None
        bond_name = []

    if (ntriplet==2):
        sig2 = [1, 1]
        ls2 = [1, 1]
        triplet_mask = np.ones(nspecs**3, dtype=int)
        triplet_mask[0] = 0
        triplet_name = ["sig3"]*2+["ls3"]*2
    elif (ntriplet==1):
        sig2 = [1]
        ls2 = [1]
        triplet_mask = np.zeros(nspecs**3, dtype=int)
        triplet_name = ["sig3"]+["ls3"]
    else:
        triplet_mask = None
        triplet_name = []

    if (nbond>0 and ntriplet>0):
        hyps = np.hstack([sig1, ls1, sig2, ls2, [1]])
        hyps_label = np.hstack([bond_name, triplet_name, ['noise']])
    elif (nbond>0):
        hyps = np.hstack([sig1, ls1, [1]])
        hyps_label = np.hstack([bond_name, ['noise']])
    else:
        hyps = np.hstack([sig2, ls2, [1]])
        hyps_label = np.hstack([triplet_name, ['noise']])

    hyps_mask = {'nspec': nspecs,
                 'spec_mask': specs_mask,
                 'nbond': nbond,
                 'bond_mask': bond_mask,
                 'ntriplet': ntriplet,
                 'triplet_mask': triplet_mask,
                 'hyps_label': hyps_label,
                 'train_noise': True}

    return hyps, hyps_mask


# ------------------------------------------------------
#          fixtures
# ------------------------------------------------------


# set the scope to module so it will only be setup once
@pytest.fixture(scope='module')
def two_body_gp() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = np.array([0.8, 0.8])
    noa = 5

    nbond = 1
    ntriplet = 0
    hyps, hm = generate_hm(nbond, ntriplet)

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)

    # test update_db
    gaussian = \
        GaussianProcess(kernel=en.two_body_mc,
                        kernel_grad=en.two_body_mc_grad,
                        hyps=hyps,
                        hyp_labels=hm['hyps_label'],
                        cutoffs=cutoffs, multihyps=True, hyps_mask=hm,
                        par=False, ncpus=1)
    gaussian.update_db(test_structure, forces)

    # return gaussian
    yield gaussian

    # code after yield will be executed once all tests are run
    # this will not be run if an exception is raised in the setup
    print("\n\nTearing down\n")
    del gaussian

@pytest.fixture(scope='module')
def three_body_gp() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = np.array([0.8, 0.8])
    noa = 5

    nbond = 0
    ntriplet = 1
    hyps, hm = generate_hm(nbond, ntriplet)

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)

    # test update_db
    gaussian = \
        GaussianProcess(kernel=en.three_body_mc,
                        kernel_grad=en.three_body_mc_grad,
                        hyps=hyps,
                        hyp_labels=hm['hyps_label'],
                        cutoffs=cutoffs, multihyps=True, hyps_mask=hm)
    gaussian.update_db(test_structure, forces)

    # return gaussian
    yield gaussian

    # code after yield will be executed once all tests are run
    # this will not be run if an exception is raised in the setup
    print("\n\nTearing down\n")
    del gaussian

# set the scope to module so it will only be setup once
@pytest.fixture(scope='module')
def two_plus_three_gp() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""
    print("\nSetting up...\n")

    # params
    cell = np.eye(3)
    unique_species = [2, 1]
    cutoffs = np.array([0.8, 0.8])
    noa = 5

    nbond = 1
    ntriplet = 1
    hyps, hm = generate_hm(nbond, ntriplet)

    # create test structure
    test_structure, forces = get_random_structure(cell, unique_species,
                                                  noa)
    efk = en.two_plus_three_mc_force_en
    ek = en.two_plus_three_mc_en

    # test update_db
    gaussian = \
        GaussianProcess(kernel=en.two_plus_three_body_mc,
                        kernel_grad=en.two_plus_three_body_mc_grad,
                        hyps=hyps,
                        energy_force_kernel = efk,
                        energy_kernel=ek,
                        opt_algorithm = 'BFGS',
                        hyp_labels=hm['hyps_label'],
                        cutoffs=cutoffs, multihyps=True, hyps_mask=hm)
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
def test_point() -> AtomicEnvironment:
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

    print("hello", two_body_gp.get_kernel_vector(test_point, 1).shape)
    print("hello", two_body_gp.get_kernel_vector(test_point, 1).shape)
    assert (two_body_gp.get_kernel_vector(test_point, 1).shape ==
            (params['db_pts'],)), \
           f"{two_body_gp.get_kernel_vector(test_point, 1)} {params['db_pts']}"


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
    assert 'Kernel: two_body_mc' in the_str
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

    gp = GaussianProcess(kernel=en.two_plus_three_body_mc,
                        kernel_grad=en.two_plus_three_body_mc_grad,
                        hyps=hyps,
                        hyp_labels=hyp_labels,
                        cutoffs=cutoffs, par=False, ncpus=1,
                        hyps_mask=hyps_mask,
                        maxiter=1,multihyps=True)


    gp.update_db(two_species_structure,
                 two_species_structure.forces)

    # Check that the hyperparameters were updated
    results = gp.train()
    assert not np.equal(results.x, hyps).all()

    #TODO check that the predictions match up with a 2+3 body
    # kernel without hyperparameter masking
