import time
import pytest
import numpy as np

from flare import gp, gp_algebra
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.mc_simple import two_plus_three_body_mc, \
        two_plus_three_body_mc_grad

from flare.gp_algebra import get_like_grad_from_mats


def get_random_training_set(nenv):
    """Create a random training_set array with parameters """

    np.random.seed(0)

    cutoffs = np.array([0.8, 0.8])
    hyps = np.ones(5, dtype=float)
    kernel = (two_plus_three_body_mc, two_plus_three_body_mc_grad)

    # create test data
    cell = np.eye(3)
    unique_species = [2, 1]
    noa = 5
    training_data = []
    training_labels = []
    for idenv in range(nenv):
        positions = np.random.uniform(-1, 1, [noa,3])
        species = np.random.randint(0, len(unique_species), noa)
        struc = Structure(cell, species, positions)
        training_data += [AtomicEnvironment(struc, 1, cutoffs)]
        training_labels += [np.random.uniform(-1, 1, 3)]
    training_labels = np.hstack(training_labels)

    return hyps, training_data, training_labels, kernel, cutoffs


def test_ky_mat():
    """
    test function get_ky_mat in gp_algebra
    using gp_algebra_origin as reference
    TO DO: store the reference... and call it explicitely
    """

    hyps, training_data, training_labels, kernel, cutoffs = \
            get_random_training_set(10)

    func = [gp_algebra.get_ky_mat,
            gp_algebra.get_ky_mat_par]

    # get the reference
    # timer0 = time.time()
    ky_mat0 = func[0](hyps, training_data,
                      kernel[0], cutoffs)

    # print("linear", time.time()-timer0)

    # parallel version
    ky_mat = func[1](hyps, training_data,
                     kernel[0], cutoffs,
                     ncpus=2)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"


def test_ky_and_hyp():

    hyps, training_data, training_labels, kernel, cutoffs = \
            get_random_training_set(10)

    func = [gp_algebra.get_ky_and_hyp,
            gp_algebra.get_ky_and_hyp_par]

    hypmat_0, ky_mat0 = func[0](hyps, training_data,
                       kernel[1], cutoffs)

    # parallel version
    hypmat, ky_mat = func[1](hyps, training_data,
                     kernel[1], cutoffs, ncpus=2)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"


def test_grad():

    hyps, training_data, training_labels, kernel, cutoffs = \
            get_random_training_set(10)

    # obtain reference
    func = gp_algebra.get_ky_and_hyp
    hyp_mat, ky_mat = func(hyps, training_data,
                       kernel[1], cutoffs)
    like0, like_grad0 = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, training_labels)

    # serial implementation
    func = gp_algebra.get_ky_and_hyp
    hyp_mat, ky_mat = func(hyps, training_data,
                       kernel[1], cutoffs)
    like, like_grad = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, training_labels)

    assert (like==like0), "wrong likelihood"
    assert np.max(np.abs(like_grad-like_grad0))==0, "wrong likelihood"


def test_ky_hyp_grad():

    hyps, training_data, training_labels, kernel, cutoffs = \
            get_random_training_set(10)

    func = gp_algebra.get_ky_and_hyp

    hyp_mat, ky_mat = func(hyps, training_data,
                       kernel[1], cutoffs)

    print(hyp_mat.shape, ky_mat.shape, len(training_labels), training_labels[0])

    like, like_grad = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, training_labels)
    delta = 0.001
    for i in range(len(hyps)):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        hyp_mat_p, ky_mat_p = func(newhyps, training_data,
                           kernel[1], cutoffs)
        like_p, like_grad_p = \
                         get_like_grad_from_mats(ky_mat_p, hyp_mat_p, training_labels)
        newhyps[i] -= 2*delta
        hyp_mat_m, ky_mat_m = func(newhyps, training_data,
                           kernel[1], cutoffs)
        like_m, like_grad_m = \
                         get_like_grad_from_mats(ky_mat_m, hyp_mat_m, training_labels)
        diff = np.abs(like_grad[i]-(like_p-like_m)/2./delta)
        assert (diff < 1e-3), "wrong calculation of hyp_mat"
