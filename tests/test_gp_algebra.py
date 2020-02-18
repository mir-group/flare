import time
import pytest
import numpy as np

import flare.gp_algebra
from flare import gp
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.kernels.mc_simple import two_plus_three_body_mc, \
        two_plus_three_body_mc_grad
from flare.kernels.mc_sephyps import two_plus_three_body_mc \
        as two_plus_three_body_mc_multi
from flare.kernels.mc_sephyps import two_plus_three_body_mc_grad \
        as two_plus_three_body_mc_grad_multi

from flare.gp_algebra import get_like_grad_from_mats, \
        get_kernel_vector, get_ky_mat, \
        get_ky_mat_update, get_ky_and_hyp

from .fake_gp import get_tstp

@pytest.fixture(scope='module')
def params():

    parameters = get_random_training_set(10)

    yield parameters
    del parameters


def get_random_training_set(nenv):
    """Create a random training_set array with parameters
    And generate four different kinds of hyperparameter sets:
    * multi hypper parameters with two bond type and two triplet type
    * constrained optimization, with noise parameter optimized
    * constrained optimization, without noise parameter optimized
    * simple hyper parameters without multihyps set up
    """

    np.random.seed(0)

    cutoffs = np.array([0.8, 0.8])
    hyps = np.ones(5, dtype=float)
    kernel = (two_plus_three_body_mc, two_plus_three_body_mc_grad)
    kernel_m = (two_plus_three_body_mc_multi, two_plus_three_body_mc_grad_multi)

    # 9 different hyper-parameters
    hyps_mask1 = {'nspec': 2,
                 'spec_mask': np.zeros(118, dtype=int),
                 'nbond': 2,
                 'bond_mask': np.array([0, 1, 1, 1]),
                 'triplet_mask': np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                 'ntriplet': 2}
    hyps_mask1['spec_mask'][2] = 1
    hyps1 = np.ones(9, dtype=float)

    # 9 different hyper-parameters, onlye train the 0, 2, 4, 6, 8
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

    # 9 different hyper-parameters, only train the 0, 2, 4, 6
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

    # 5 different hyper-parameters, equivalent to no multihyps
    hyps_mask4 = {'nspec': 1,
                 'spec_mask': np.zeros(118, dtype=int),
                 'nbond': 1,
                 'bond_mask': np.array([0]),
                 'ntriplet': 1,
                 'triplet_mask': np.array([0])}
    hyps4 = np.ones(5, dtype=float)
    hyps_list = [hyps1, hyps2, hyps3, hyps4, hyps]
    hyps_mask_list = [hyps_mask1, hyps_mask2, hyps_mask3, hyps_mask4, None]

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

    # store it as global variables
    name = "unit_test"
    flare.gp_algebra._global_training_data[name] = training_data
    flare.gp_algebra._global_training_labels[name] = training_labels

    return hyps, name, kernel, cutoffs, \
           kernel_m, hyps_list, hyps_mask_list


def test_ky_mat(params):
    """
    test function get_ky_mat in gp_algebra, gp_algebra_multi
    using gp_algebra_origin as reference
    TO DO: store the reference... and call it explicitely
    """

    hyps, name, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = params


    # get the reference
    # without multi hyps
    time0 = time.time()
    ky_mat0 = get_ky_mat(hyps, name, kernel[0], cutoffs)
    print("compute ky_mat serial", time.time()-time0)

    # parallel version
    time0 = time.time()
    ky_mat = get_ky_mat(hyps, name,
                     kernel[0], cutoffs,
                     n_cpus=2, n_sample=20)
    print("compute ky_mat parallel", time.time()-time0)

    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"

    # this part of the code can be use for timing the parallel performance
    # for n in [10, 50, 100]:
    #     timer0 = time.time()
    #     ky_mat = get_ky_mat(hyps, name,
    #                      kernel[0], cutoffs,
    #                      ncpus=8, n_sample=n)
    #     diff = (np.max(np.abs(ky_mat-ky_mat0)))
    #     print("parallel", n, time.time()-timer0, diff)
    #     assert (diff==0), "parallel implementation is wrong"

    # check multi hyps implementation
    # compute the ky_mat with different parameters
    for i in range(len(hyps_list)):

        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]

        if hyps_mask is None:
            ker = kernel[0]
        else:
            ker = kernel_m[0]

        # serial implementation
        time0 = time.time()
        ky_mat = get_ky_mat(hyps, name,
                          ker, cutoffs, hyps_mask)
        print(f"compute ky_mat with multihyps, test {i}, n_cpus=1", time.time()-time0)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff==0), "multi hyps implementation is wrong"\
                          f"with case {i}"

        # parallel implementation
        time0 = time.time()
        ky_mat = get_ky_mat(hyps, name,
                         ker,
                         cutoffs, hyps_mask, n_cpus=2, n_sample=20)
        print(f"compute ky_mat with multihyps, test {i}, n_cpus=2", time.time()-time0)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff==0), "multi hyps  parallel "\
                          "implementation is wrong"\
                          f"with case {i}"

def test_ky_mat_update(params):
    """
    check ky_mat_update function
    """

    hyps, name, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = params

    # prepare old data set as the starting point
    n = 5
    training_data = flare.gp_algebra._global_training_data[name]
    flare.gp_algebra._global_training_data['old'] = training_data[:n]
    training_labels = flare.gp_algebra._global_training_labels[name]
    flare.gp_algebra._global_training_labels['old'] = training_labels[:n]

    func = [get_ky_mat,
            get_ky_mat_update]

    # get the reference
    ky_mat0 = func[0](hyps, name,
                      kernel[0], cutoffs)
    ky_mat_old = func[0](hyps, 'old',
                         kernel[0], cutoffs)

    # update
    ky_mat = func[1](ky_mat_old, hyps, name,
                     kernel[0], cutoffs)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "update function is wrong"

    # parallel version
    ky_mat = func[1](ky_mat_old, hyps, name,
                     kernel[0], cutoffs,
                     n_cpus=2, n_sample=20)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"

    # check multi hyps implementation
    for i in range(len(hyps_list)):

        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]

        if hyps_mask is None:
            ker = kernel[0]
        else:
            ker = kernel_m[0]

        # serial implementation
        ky_mat = func[1](ky_mat_old, hyps, name,
                         ker, cutoffs, hyps_mask)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff<1e-12), "multi hyps parameter implementation is wrong"

        # parallel implementation
        ky_mat = func[1](ky_mat_old, hyps, name,
                         ker, cutoffs,
                         hyps_mask, n_cpus=2, n_sample=20)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff<1e-12), "multi hyps parameter parallel "\
                "implementation is wrong"

def test_get_kernel_vector(params):

    hyps, name, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = params

    test_point = get_tstp()

    size = len(flare.gp_algebra._global_training_data[name])

    # test the parallel implementation for multihyps
    vec = get_kernel_vector(name, kernel_m[0],
                          test_point, 1, hyps,
                          cutoffs, hyps_mask_list[0])

    vec_par = get_kernel_vector(name, kernel_m[0],
                          test_point, 1, hyps,
                          cutoffs, hyps_mask_list[0],
                          n_cpus=2, n_sample=100)

    assert (all(np.equal(vec, vec_par))), "parallel implementation is wrong"
    assert (vec.shape[0] == size*3), \
            f"{vec} {size}"

def test_ky_and_hyp(params):

    hyps, name, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = params

    hypmat_0, ky_mat0 = get_ky_and_hyp(hyps, name,
                       kernel[1], cutoffs)

    # parallel version
    hypmat, ky_mat = get_ky_and_hyp(hyps, name,
                     kernel[1], cutoffs, n_cpus=2)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff==0), "parallel implementation is wrong"

    # check all cases
    for i in range(len(hyps_list)):
        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]

        if hyps_mask is None:
            ker = kernel[1]
        else:
            ker = kernel_m[1]

        # serial implementation
        hypmat, ky_mat = get_ky_and_hyp(
                hyps, name, ker, cutoffs, hyps_mask)

        if (i == 0):
            hypmat9 = hypmat
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff==0), "multi hyps parameter implementation is wrong"

        # compare to no hyps_mask version
        diff = 0
        if (i == 1):
            diff = (np.max(np.abs(hypmat-hypmat9[[0,2,4,6,8], :, :])))
        elif (i==2):
            diff = (np.max(np.abs(hypmat-hypmat9[[0,2,4,6], :, :])))
        elif (i==3):
            diff = (np.max(np.abs(hypmat-hypmat_0)))
        elif (i==4):
            diff = (np.max(np.abs(hypmat-hypmat_0)))
        assert (diff==0), "multi hyps implementation is wrong"\
                          f"in case {i}"

        # parallel implementation
        hypmat_par, ky_mat_par = get_ky_and_hyp(hyps, name,
                         ker, cutoffs, hyps_mask,
                         n_cpus=2, n_sample=2)

        # compare to serial implementation
        diff = (np.max(np.abs(ky_mat-ky_mat_par)))
        assert (diff==0), f"multi hyps parallel "\
                f"implementation is wrong in case {i}"

        diff = (np.max(np.abs(hypmat_par-hypmat)))
        assert (diff==0), f"multi hyps parallel implementation is wrong"\
                f" in case{i}"

def test_grad(params):


    hyps, name, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = params

    # obtain reference
    func = get_ky_and_hyp
    hyp_mat, ky_mat = func(hyps, name,
                       kernel[1], cutoffs)
    like0, like_grad0 = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, name)

    # serial implementation
    func = get_ky_and_hyp
    hyp_mat, ky_mat = func(hyps, name,
                       kernel[1], cutoffs)
    like, like_grad = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, name)

    assert (like==like0), "wrong likelihood"
    assert np.max(np.abs(like_grad-like_grad0))==0, "wrong likelihood"

    func = get_ky_and_hyp
    for i in range(len(hyps_list)):
        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]

        if hyps_mask is None:
            ker = kernel[1]
        else:
            ker = kernel_m[1]

        hyp_mat, ky_mat = func(hyps, name,
                          ker, cutoffs, hyps_mask)
        like, like_grad = \
                         get_like_grad_from_mats(ky_mat, hyp_mat, name)
        assert (like==like0), "wrong likelihood"

        if (i==0):
            like_grad9 = like_grad

        diff = 0
        if (i==1):
            diff = (np.max(np.abs(like_grad-like_grad9[[0,2,4,6,8]])))
        elif (i==2):
            diff = (np.max(np.abs(like_grad-like_grad9[[0,2,4,6]])))
        elif (i==3):
            diff = (np.max(np.abs(like_grad-like_grad0)))
        elif (i==4):
            diff = (np.max(np.abs(like_grad-like_grad0)))
        assert (diff==0), "multi hyps implementation is wrong"\
                          f"in case {i}"

def test_ky_hyp_grad(params):


    hyps, name, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list = params

    func = get_ky_and_hyp

    hyp_mat, ky_mat = func(hyps, name,
                       kernel[1], cutoffs)

    size = len(flare.gp_algebra._global_training_data[name])

    like, like_grad = \
                     get_like_grad_from_mats(ky_mat, hyp_mat, name)
    delta = 0.001
    for i in range(len(hyps)):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        hyp_mat_p, ky_mat_p = func(newhyps, name,
                           kernel[1], cutoffs)
        like_p, like_grad_p = \
                         get_like_grad_from_mats(ky_mat_p, hyp_mat_p, name)
        newhyps[i] -= 2*delta
        hyp_mat_m, ky_mat_m = func(newhyps, name,
                           kernel[1], cutoffs)
        like_m, like_grad_m = \
                         get_like_grad_from_mats(ky_mat_m, hyp_mat_m, name)
        diff = np.abs(like_grad[i]-(like_p-like_m)/2./delta)
        assert (diff < 1e-3), "wrong calculation of hyp_mat"
