import time
import pytest
import numpy as np

from flare import gp_algebra, gp_algebra_multi, gp_algebra_origin
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.mc_simple import two_plus_three_body_mc, \
        two_plus_three_body_mc_grad
from flare.mc_sephyps import two_plus_three_body_mc \
        as two_plus_three_body_mc_multi
from flare.mc_sephyps import two_plus_three_body_mc_grad \
        as two_plus_three_body_mc_grad_multi
from flare.gp_algebra import get_like_grad_from_mats


def get_random_training_set(nenv):
    """Create a random training_set array with parameters """

    np.random.seed(0)

    cutoffs = np.array([0.8, 0.8])
    hyps = np.ones(5, dtype=float)
    kernel = (two_plus_three_body_mc, two_plus_three_body_mc_grad)
    kernel_m = (two_plus_three_body_mc_multi, two_plus_three_body_mc_grad_multi)
    hyps_mask1 = {'nspec': 2,
                 'spec_mask': np.zeros(118, dtype=int),
                 'nbond': 2,
                 'bond_mask': np.array([0, 1, 1, 1]),
                 'triplet_mask': np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                 'ntriplet': 2}
    hyps_mask1['spec_mask'][2] = 1
    hyps1 = np.ones(9, dtype=float)

    hyps_mask2 = {'nspec': 2,
                 'spec_mask': np.zeros(118, dtype=int),
                 'nbond': 2,
                 'bond_mask': np.array([0, 1, 1, 1]),
                 'ntriplet': 2,
                 'triplet_mask': np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                 'train_noise':True,
                 'map':[0,2,4,6,8],
                 'original':np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])}
    hyps_mask2['spec_mask'][2] = 1
    hyps2 = np.ones(5, dtype=float)

    hyps_mask3 = {'nspec': 2,
                 'spec_mask': np.zeros(118, dtype=int),
                 'nbond': 2,
                 'bond_mask': np.array([0, 1, 1, 1]),
                 'ntriplet': 2,
                 'triplet_mask': np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                 'train_noise':False,
                 'map':[0,2,4,6],
                 'original':np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])}
    hyps_mask3['spec_mask'][2] = 1
    hyps3 = np.ones(4, dtype=float)

    hyps_mask4 = {'nspec': 1,
                 'spec_mask': np.zeros(118, dtype=int),
                 'nbond': 1,
                 'bond_mask': np.array([0]),
                 'ntriplet': 1,
                 'triplet_mask': np.array([0])}
    hyps4 = np.ones(4, dtype=float)

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
    hyps_list = [hyps1, hyps2, hyps3, hyps4]
    hyps_mask_list = [hyps_mask1, hyps_mask2, hyps_mask3, hyps_mask4]

    return hyps, training_data, training_labels, kernel, cutoffs, \
           kernel_m, hyps_list, hyps_mask_list


def test_ky_mat():

    hyps, training_data, training_labels, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = \
            get_random_training_set(50)


    func = [gp_algebra_origin.get_ky_mat,
            gp_algebra.get_ky_mat_par,
            gp_algebra_multi.get_ky_mat,
            gp_algebra_multi.get_ky_mat_par]

    # get the reference
    # timer0 = time.time()
    ky_mat0 = func[0](hyps, training_data,
                      training_labels,
                      kernel[0], cutoffs)

    # print("linear", time.time()-timer0)

    # parallel version
    ky_mat = func[1](hyps, training_data,
                     kernel[0], cutoffs,
                     ncpus=2, nsample=20)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"

    # this part of the code can be use for timing the parallel performance
    # for n in [10, 50, 100]:
    #     timer0 = time.time()
    #     ky_mat = func[1](hyps, training_data,
    #                      kernel[0], cutoffs,
    #                      ncpus=8, nsample=n)
    #     diff = (np.max(np.abs(ky_mat-ky_mat0)))
    #     print("parallel", n, time.time()-timer0, diff)
    #     assert (diff==0), "parallel implementation is wrong"

    # check multi hyps implementation
    for i in range(3):
        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]
        ky_mat = func[2](hyps, training_data,
                          kernel_m[0], cutoffs, hyps_mask)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff==0), "multi hyps parameter implementation is wrong"

        # check multi hyps parallel implementation
        ky_mat = func[3](hyps, training_data,
                         kernel_m[0],
                         cutoffs, hyps_mask, ncpus=2, nsample=20)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff==0), "multi hyps parameter parallel "\
                "implementation is wrong"

def test_ky_mat_update():

    hyps, training_data, training_labels, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = \
            get_random_training_set(50)


    func = [gp_algebra.get_ky_mat,
            gp_algebra.get_ky_mat_update,
            gp_algebra.get_ky_mat_update_par,
            gp_algebra_multi.get_ky_mat_update,
            gp_algebra_multi.get_ky_mat_update_par]

    # get the reference
    # timer0 = time.time()
    ky_mat0 = func[0](hyps, training_data,
                       kernel[0], cutoffs)
    n = 20
    ky_mat_old = func[0](hyps, training_data[:n],
                       kernel[0], cutoffs)
    # print("linear", time.time()-timer0)

    # parallel version
    ky_mat = func[1](ky_mat_old, hyps, training_data,
                     kernel[0], cutoffs)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff<1e-12), "update function is wrong"

    ky_mat = func[2](ky_mat_old, hyps, training_data,
                     kernel[0], cutoffs,
                     ncpus=2, nsample=20)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff<1e-12), "parallel implementation is wrong"

    for i in range(3):
        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]
        # check multi hyps implementation
        ky_mat = func[3](ky_mat_old, hyps, training_data,
                         kernel_m[0], cutoffs, hyps_mask)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff<1e-12), "multi hyps parameter implementation is wrong"

        # check multi hyps parallel implementation
        ky_mat = func[4](ky_mat_old, hyps, training_data,
                         kernel_m[0], cutoffs,
                         hyps_mask, ncpus=2, nsample=20)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff<1e-12), "multi hyps parameter parallel "\
                "implementation is wrong"

def test_ky_and_hyp():

    hyps, training_data, training_labels, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = \
            get_random_training_set(50)

    func = [gp_algebra_origin.get_ky_and_hyp,
            gp_algebra.get_ky_and_hyp_par,
            gp_algebra_multi.get_ky_and_hyp,
            gp_algebra_multi.get_ky_and_hyp_par]

    hypmat_0, ky_mat0 = func[0](hyps, training_data,
                       training_labels,
                       kernel[1], cutoffs)

    # parallel version
    hypmat, ky_mat = func[1](hyps, training_data,
                     kernel[1], cutoffs, ncpus=2)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"

    for i in range(3):
        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]
        # check multi hyps implementation
        hypmat, ky_mat = func[2](hyps, training_data,
                          kernel_m[1], cutoffs, hyps_mask)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff==0), "multi hyps parameter implementation is wrong"

        if (i==1):
            diff = (np.max(np.abs(hypmat-hypmat_0)))
            assert (diff==0), "multi hyps parameter implementation is wrong"

        # check multi hyps parallel implementation
        hypmat, ky_mat = func[3](hyps, training_data,
                         kernel_m[1], cutoffs, hyps_mask,
                         ncpus=2)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff==0), "multi hyps parameter parallel "\
                "implementation is wrong"
        if (i==1):
            diff = (np.max(np.abs(hypmat-hysmat_0)))
            assert (diff==0), "multi hyps parameter implementation is wrong"

def test_grad():


    hyps, training_data, training_labels, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = \
            get_random_training_set(50)

    func = gp_algebra.get_ky_and_hyp

    hyp_mat, ky_mat = func(hyps, training_data,
                       kernel[1], cutoffs)

    like, like_grad = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, training_labels)

    func = gp_algebra_origin.get_ky_and_hyp

    hyp_mat, ky_mat = func(hyps, training_data,
                       training_labels,
                       kernel[1], cutoffs)

    like0, like_grad0 = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, training_labels)
    print(like, like0)
    assert (like==like0), "wrong likelihood"
    print(like_grad, like_grad0)
    assert np.max(np.abs(like_grad-like_grad0))==0, "wrong likelihood"

    func = gp_algebra_multi.get_ky_and_hyp

    hyp_mat, ky_mat = func(hyps2, training_data,
                      kernel_m[1], cutoffs, hyps_mask2)

    like, like_grad = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, training_labels)
    print(like, like0)
    assert (like==like0), "wrong likelihood"
    print(like_grad, like_grad0)
    assert np.max(np.abs(like_grad-like_grad0))==0, "wrong likelihood"

def test_ky_hyp_grad():


    hyps, training_data, training_labels, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = \
            get_random_training_set(50)

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

