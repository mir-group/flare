import time
import pytest
import numpy as np

from flare import gp_algebra, gp_algebra_multi
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.mc_simple import two_plus_three_body_mc, \
        two_plus_three_body_mc_grad
from flare.mc_sephyps import two_plus_three_body_mc \
        as two_plus_three_body_mc_multi
from flare.mc_sephyps import two_plus_three_body_mc_grad \
        as two_plus_three_body_mc_grad_multi


def get_random_training_set(nenv):
    """Create a random training_set array with parameters """

    np.random.seed(0)

    cutoffs = np.array([0.8, 0.8])
    hyps = np.ones(5, dtype=float)
    kernel = (two_plus_three_body_mc, two_plus_three_body_mc_grad)
    kernel_m = (two_plus_three_body_mc_multi, two_plus_three_body_mc_grad_multi)
    hyps_mask = {'nspec': 2,
                 'spec_mask': np.zeros(118, dtype=int),
                 'nbond': 2,
                 'bond_mask': np.array([0, 1]),
                 'ntriplet': 2,
                 'triplet_mask': np.array([0, 1])}
    hyps_mask['spec_mask'][2] = 1

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

    return hyps, training_data, training_labels, kernel, cutoffs, \
           kernel_m, hyps_mask


def test_ky_mat():

    hyps, training_data, training_labels, kernel, cutoffs, \
            kernel_m, hyps_mask = \
            get_random_training_set(50)


    func = [gp_algebra.get_ky_mat,
            gp_algebra.get_ky_mat_par,
            gp_algebra_multi.get_ky_mat,
            gp_algebra_multi.get_ky_mat_par]

    # get the reference
    # timer0 = time.time()
    ky_mat0 = func[0](hyps, training_data,
                       training_labels, kernel[0], cutoffs)
    # print("linear", time.time()-timer0)

    # parallel version
    ky_mat = func[1](hyps, training_data,
                     training_labels, kernel[0], cutoffs,
                     no_cpus=2, nsample=20)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"

    # this part of the code can be use for timing the parallel performance
    # for n in [10, 50, 100]:
    #     timer0 = time.time()
    #     ky_mat = func[1](hyps, training_data,
    #                      training_labels, kernel[0], cutoffs,
    #                      no_cpus=8, nsample=n)
    #     diff = (np.max(np.abs(ky_mat-ky_mat0)))
    #     print("parallel", n, time.time()-timer0, diff)
    #     assert (diff==0), "parallel implementation is wrong"

    hyps = np.ones(9, dtype=float)
    # check multi hyps implementation
    ky_mat = func[2](hyps, training_data,
                      training_labels, kernel_m[0], cutoffs, hyps_mask)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "multi hyps parameter implementation is wrong"

    # check multi hyps parallel implementation
    ky_mat = func[3](hyps, training_data,
                     training_labels, kernel_m[0], cutoffs,
                     hyps_mask, no_cpus=2, nsample=20)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "multi hyps parameter parallel "\
            "implementation is wrong"


def test_ky_and_hyp():

    hyps, training_data, training_labels, kernel, cutoffs, \
            kernel_m, hyps_mask = \
            get_random_training_set(10)

    func = [gp_algebra.get_ky_and_hyp,
            gp_algebra.get_ky_and_hyp_par,
            gp_algebra_multi.get_ky_and_hyp,
            gp_algebra_multi.get_ky_and_hyp_par]

    hypmat_0, ky_mat0 = func[0](hyps, training_data,
                       training_labels, kernel[1], cutoffs)
    # parallel version
    hypmat, ky_mat = func[1](hyps, training_data,
                     training_labels, kernel[1], cutoffs, no_cpus=2)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"

    hyps = np.ones(9, dtype=float)
    # check multi hyps implementation
    hypmat, ky_mat = func[2](hyps, hyps_mask, training_data,
                      training_labels, kernel_m[1], cutoffs)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "multi hyps parameter implementation is wrong"

    # check multi hyps parallel implementation
    hypmat, ky_mat = func[3](hyps, hyps_mask, training_data,
                     training_labels, kernel_m[1], cutoffs,
                     no_cpus=2)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "multi hyps parameter parallel "\
            "implementation is wrong"
