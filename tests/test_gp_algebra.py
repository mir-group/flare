import time
import pytest
import numpy as np

import flare.gp_algebra
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.kernels import mc_sephyps
from flare.kernels.mc_simple import two_plus_three_body_mc, \
        two_plus_three_body_mc_grad, two_plus_three_mc_en,\
        two_plus_three_mc_force_en
from flare.kernels.mc_sephyps import two_plus_three_body_mc \
        as two_plus_three_body_mc_multi
from flare.kernels.mc_sephyps import two_plus_three_body_mc_grad \
        as two_plus_three_body_mc_grad_multi

from flare.gp_algebra import get_like_grad_from_mats, \
        get_force_block, get_force_energy_block, \
        energy_energy_vector, energy_force_vector, force_force_vector, \
        force_energy_vector, get_kernel_vector, en_kern_vec, \
        get_ky_mat_update, get_ky_and_hyp, get_energy_block, \
        get_Ky_mat, update_force_block, update_energy_block, \
        update_force_energy_block

from .fake_gp import get_tstp


@pytest.fixture(scope='module')
def params():

    parameters = get_random_training_set(10, 2)

    yield parameters
    del parameters


def get_random_training_set(nenv, nstruc):
    """Create a random training_set array with parameters
    And generate four different kinds of hyperparameter sets:
    * multi hypper parameters with two twobody type and two threebody type
    * constrained optimization, with noise parameter optimized
    * constrained optimization, without noise parameter optimized
    * simple hyper parameters without multihyps set up
    """

    np.random.seed(0)

    cutoffs = {'twobody':0.8, 'threebody':0.8}
    hyps = np.ones(5, dtype=float)
    kernel = (two_plus_three_body_mc, two_plus_three_body_mc_grad,
              two_plus_three_mc_en, two_plus_three_mc_force_en)
    kernel_m = \
        (two_plus_three_body_mc_multi, two_plus_three_body_mc_grad_multi,
         mc_sephyps.two_plus_three_mc_en,
         mc_sephyps.two_plus_three_mc_force_en)

    # 9 different hyper-parameters
    hyps_mask1 = {'kernels':['twobody', 'threebody'],
                 'twobody_start': 0,
                 'threebody_start': 4,
                 'nspecie': 2,
                 'specie_mask': np.zeros(118, dtype=int),
                 'ntwobody': 2,
                 'twobody_mask': np.array([0, 1, 1, 1]),
                 'threebody_mask': np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                 'nthreebody': 2}
    hyps_mask1['specie_mask'][2] = 1
    hyps1 = np.ones(9, dtype=float)

    # 9 different hyper-parameters, onlye train the 0, 2, 4, 6, 8
    hyps_mask2 = {'kernels':['twobody', 'threebody'],
                 'twobody_start': 0,
                 'threebody_start': 4,
                 'nspecie': 2,
                 'specie_mask': np.zeros(118, dtype=int),
                 'ntwobody': 2,
                 'twobody_mask': np.array([0, 1, 1, 1]),
                 'nthreebody': 2,
                 'threebody_mask': np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                 'train_noise':True,
                 'map':[0,2,4,6,8],
                 'original_hyps':np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])}
    hyps_mask2['specie_mask'][2] = 1
    hyps2 = np.ones(5, dtype=float)

    # 9 different hyper-parameters, only train the 0, 2, 4, 6
    hyps_mask3 = {'kernels':['twobody', 'threebody'],
                 'twobody_start': 0,
                 'threebody_start': 4,
                 'nspecie': 2,
                 'specie_mask': np.zeros(118, dtype=int),
                 'ntwobody': 2,
                 'twobody_mask': np.array([0, 1, 1, 1]),
                 'nthreebody': 2,
                 'threebody_mask': np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                 'train_noise':False,
                 'map':[0,2,4,6],
                 'original_hyps':np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])}
    hyps_mask3['specie_mask'][2] = 1
    hyps3 = np.ones(4, dtype=float)

    # 5 different hyper-parameters, equivalent to no multihyps
    hyps_mask4 = {'kernels':['twobody', 'threebody'],
                 'twobody_start': 0,
                 'threebody_start': 4,
                 'nspecie': 1,
                 'specie_mask': np.zeros(118, dtype=int),
                 'ntwobody': 1,
                 'twobody_mask': np.array([0]),
                 'nthreebody': 1,
                 'threebody_mask': np.array([0])}
    hyps4 = np.ones(5, dtype=float)
    hyps_list = [hyps1, hyps2, hyps3, hyps4, hyps]
    hyps_mask_list = [hyps_mask1, hyps_mask2, hyps_mask3, hyps_mask4, None]

    # create training environments and forces
    cell = np.eye(3)
    unique_species = [0, 1]
    noa = 5
    training_data = []
    training_labels = []
    for _ in range(nenv):
        positions = np.random.uniform(-1, 1, [noa, 3])
        species = np.random.randint(0, len(unique_species), noa) + 1
        struc = Structure(cell, species, positions)
        training_data += [AtomicEnvironment(struc, 1, cutoffs)]
        training_labels += [np.random.uniform(-1, 1, 3)]
    training_labels = np.hstack(training_labels)

    # create training structures and energies
    training_structures = []
    energy_labels = []
    for _ in range(nstruc):
        positions = np.random.uniform(-1, 1, [noa, 3])
        species = np.random.randint(0, len(unique_species), noa) + 1
        struc = Structure(cell, species, positions)
        struc_envs = []
        for n in range(noa):
            struc_envs.append(AtomicEnvironment(struc, n, cutoffs))
        training_structures.append(struc_envs)
        energy_labels.append(np.random.uniform(-1, 1))
    energy_labels = np.array(energy_labels)

    # store it as global variables
    name = "unit_test"
    flare.gp_algebra._global_training_data[name] = training_data
    flare.gp_algebra._global_training_labels[name] = training_labels
    flare.gp_algebra._global_training_structures[name] = training_structures
    flare.gp_algebra._global_energy_labels[name] = energy_labels

    energy_noise = 0.01

    return hyps, name, kernel, cutoffs, kernel_m, hyps_list, hyps_mask_list, \
        energy_noise


def test_ky_mat(params):
    hyps, name, kernel, cutoffs, kernel_m, hyps_list, hyps_mask_list, \
        energy_noise = params

    # get the reference without multi hyps
    time0 = time.time()
    ky_mat0 = get_Ky_mat(hyps, name, kernel[0], kernel[2], kernel[3],
                         energy_noise, cutoffs)
    print("compute ky_mat serial", time.time()-time0)

    # parallel version
    time0 = time.time()
    ky_mat = \
        get_Ky_mat(hyps, name, kernel[0], kernel[2], kernel[3],
                   energy_noise, cutoffs, n_cpus=2, n_sample=5)
    print("compute ky_mat parallel", time.time()-time0)

    print(ky_mat)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff == 0), "parallel implementation is wrong"

    # compute the ky_mat with different parameters
    for i in range(len(hyps_list)):

        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]

        multihyps = True
        if hyps_mask is None:
            multihyps = False
        elif hyps_mask['nspecie'] == 1:
            multihyps = False

        if not multihyps:
            ker1 = kernel[0]
            ker2 = kernel[2]
            ker3 = kernel[3]
        else:
            ker1 = kernel_m[0]
            ker2 = kernel_m[2]
            ker3 = kernel_m[3]

        # serial implementation
        time0 = time.time()
        ky_mat = get_Ky_mat(hyps, name, ker1, ker2, ker3,
                            energy_noise, cutoffs, hyps_mask)
        print(f"compute ky_mat with multihyps, test {i}, n_cpus=1",
              time.time()-time0)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff == 0), "multi hyps implementation is wrong"\
            f"with case {i}"

        # parallel implementation
        time0 = time.time()
        ky_mat = get_Ky_mat(hyps, name, ker1, ker2, ker3,
                            energy_noise, cutoffs, hyps_mask, n_cpus=2,
                            n_sample=20)
        print(f"compute ky_mat with multihyps, test {i}, n_cpus=2",
              time.time()-time0)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff == 0), "multi hyps  parallel "\
            "implementation is wrong with case {i}"


def test_ky_mat_update(params):
    """
    check ky_mat_update function
    """

    hyps, name, kernel, cutoffs, \
        kernel_m, hyps_list, hyps_mask_list, energy_noise = params

    # prepare old data set as the starting point
    n = 5
    s = 1
    training_data = flare.gp_algebra._global_training_data[name]
    training_structures = flare.gp_algebra._global_training_structures[name]
    flare.gp_algebra._global_training_data['old'] = training_data[:n]
    flare.gp_algebra._global_training_structures['old'] = \
        training_structures[:s]
    func = [get_Ky_mat, get_ky_mat_update]

    # get the reference
    ky_mat0 = func[0](hyps, name, kernel[0], kernel[2], kernel[3],
                      energy_noise, cutoffs)
    ky_mat_old = func[0](hyps, 'old', kernel[0], kernel[2], kernel[3],
                         energy_noise, cutoffs)

    # update
    ky_mat = func[1](ky_mat_old, n, hyps, name, kernel[0], kernel[2],
                     kernel[3], energy_noise, cutoffs)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))

    assert (diff <= 1e-15), "update function is wrong"

    # parallel version
    ky_mat = func[1](ky_mat_old, n, hyps, name, kernel[0], kernel[2],
                     kernel[3], energy_noise, cutoffs, n_cpus=2, n_sample=20)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))

    assert (diff == 0), "parallel implementation is wrong"

    # check multi hyps implementation
    for i in range(len(hyps_list)):

        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]

        multihyps = True
        if hyps_mask is None:
            multihyps = False
        elif hyps_mask['nspecie'] == 1:
            multihyps = False

        if not multihyps:
            ker1 = kernel[0]
            ker2 = kernel[2]
            ker3 = kernel[3]
        else:
            ker1 = kernel_m[0]
            ker2 = kernel_m[2]
            ker3 = kernel_m[3]

        # serial implementation
        ky_mat = func[1](ky_mat_old, n, hyps, name, ker1, ker2, ker3,
                         energy_noise, cutoffs, hyps_mask)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff < 1e-12), "multi hyps parameter implementation is wrong"

        # parallel implementation
        ky_mat = func[1](ky_mat_old, n, hyps, name, ker1, ker2, ker3,
                         energy_noise, cutoffs, hyps_mask, n_cpus=2,
                         n_sample=20)
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff < 1e-12), "multi hyps parameter parallel "\
            "implementation is wrong"


def test_kernel_vector(params):

    hyps, name, kernel, cutoffs, _, _, _, _ = params

    test_point = get_tstp()

    size1 = len(flare.gp_algebra._global_training_data[name])
    size2 = len(flare.gp_algebra._global_training_structures[name])

    # test the parallel implementation for multihyps
    vec = get_kernel_vector(name, kernel[0], kernel[3], test_point, 1,
                            hyps, cutoffs)

    vec_par = \
        get_kernel_vector(name, kernel[0], kernel[3], test_point, 1, hyps,
                          cutoffs, n_cpus=2, n_sample=100)

    assert (all(np.equal(vec, vec_par))), "parallel implementation is wrong"
    assert (vec.shape[0] == size1 * 3 + size2)


def test_en_kern_vec(params):

    hyps, name, kernel, cutoffs, _, _, _, _ = params

    test_point = get_tstp()

    size1 = len(flare.gp_algebra._global_training_data[name])
    size2 = len(flare.gp_algebra._global_training_structures[name])

    # test the parallel implementation for multihyps
    vec = en_kern_vec(name, kernel[3], kernel[2], test_point, hyps, cutoffs)

    vec_par = \
        en_kern_vec(name, kernel[3], kernel[2], test_point, hyps, cutoffs,
                    n_cpus=2, n_sample=100)

    assert (all(np.equal(vec, vec_par))), "parallel implementation is wrong"
    assert (vec.shape[0] == size1 * 3 + size2)


def test_ky_and_hyp(params):

    hyps, name, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list, _ = params

    hypmat_0, ky_mat0 = get_ky_and_hyp(hyps, name, kernel[1], cutoffs)

    # parallel version
    hypmat, ky_mat = get_ky_and_hyp(hyps, name, kernel[1], cutoffs, n_cpus=2)
    diff = (np.max(np.abs(ky_mat-ky_mat0)))
    assert (diff == 0), "parallel implementation is wrong"

    # check all cases
    for i in range(len(hyps_list)):
        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]

        multihyps = True
        if hyps_mask is None:
            multihyps = False
        elif hyps_mask['nspecie'] == 1:
            multihyps = False

        if not multihyps:
            ker = kernel[1]
        else:
            ker = kernel_m[1]

        # serial implementation
        hypmat, ky_mat = get_ky_and_hyp(hyps, name, ker, cutoffs, hyps_mask)

        if (i == 0):
            hypmat9 = hypmat
        diff = (np.max(np.abs(ky_mat-ky_mat0)))
        assert (diff == 0), "multi hyps parameter implementation is wrong"

        # compare to no hyps_mask version
        diff = 0
        if (i == 1):
            diff = (np.max(np.abs(hypmat-hypmat9[[0, 2, 4, 6, 8], :, :])))
        elif (i == 2):
            diff = (np.max(np.abs(hypmat-hypmat9[[0, 2, 4, 6], :, :])))
        elif (i == 3):
            diff = (np.max(np.abs(hypmat-hypmat_0)))
        elif (i == 4):
            diff = (np.max(np.abs(hypmat-hypmat_0)))
        assert (diff == 0), "multi hyps implementation is wrong"\
            f"in case {i}"

        # parallel implementation
        hypmat_par, ky_mat_par = \
            get_ky_and_hyp(hyps, name, ker, cutoffs, hyps_mask,
                           n_cpus=2, n_sample=2)

        # compare to serial implementation
        diff = (np.max(np.abs(ky_mat-ky_mat_par)))
        assert (diff == 0), f"multi hyps parallel "\
            f"implementation is wrong in case {i}"

        diff = (np.max(np.abs(hypmat_par-hypmat)))
        assert (diff == 0), f"multi hyps parallel implementation is wrong"\
            f" in case{i}"


def test_grad(params):
    hyps, name, kernel, cutoffs, \
            kernel_m, hyps_list, hyps_mask_list, _ = params

    # obtain reference
    func = get_ky_and_hyp
    hyp_mat, ky_mat = func(hyps, name, kernel[1], cutoffs)
    like0, like_grad0 = \
        get_like_grad_from_mats(ky_mat, hyp_mat, name)

    # serial implementation
    func = get_ky_and_hyp
    hyp_mat, ky_mat = func(hyps, name, kernel[1], cutoffs)
    like, like_grad = \
        get_like_grad_from_mats(ky_mat, hyp_mat, name)

    assert (like == like0), "wrong likelihood"
    assert np.max(np.abs(like_grad-like_grad0)) == 0, "wrong likelihood"

    func = get_ky_and_hyp
    for i in range(len(hyps_list)):
        hyps = hyps_list[i]
        hyps_mask = hyps_mask_list[i]

        multihyps = True
        if hyps_mask is None:
            multihyps = False
        elif hyps_mask['nspecie'] == 1:
            multihyps = False

        if not multihyps:
            ker = kernel[1]
        else:
            ker = kernel_m[1]

        hyp_mat, ky_mat = func(hyps, name, ker, cutoffs, hyps_mask)
        like, like_grad = get_like_grad_from_mats(ky_mat, hyp_mat, name)
        assert (like == like0), "wrong likelihood"

        if (i == 0):
            like_grad9 = like_grad

        diff = 0
        if (i == 1):
            diff = (np.max(np.abs(like_grad-like_grad9[[0, 2, 4, 6, 8]])))
        elif (i == 2):
            diff = (np.max(np.abs(like_grad-like_grad9[[0, 2, 4, 6]])))
        elif (i == 3):
            diff = (np.max(np.abs(like_grad-like_grad0)))
        elif (i == 4):
            diff = (np.max(np.abs(like_grad-like_grad0)))
        assert (diff == 0), "multi hyps implementation is wrong"\
            f"in case {i}"


def test_ky_hyp_grad(params):
    hyps, name, kernel, cutoffs, _, _, _, _ = params

    func = get_ky_and_hyp

    hyp_mat, ky_mat = func(hyps, name, kernel[1], cutoffs)

    _, like_grad = get_like_grad_from_mats(ky_mat, hyp_mat, name)
    delta = 0.001
    for i in range(len(hyps)):
        newhyps = np.copy(hyps)
        newhyps[i] += delta
        hyp_mat_p, ky_mat_p = func(newhyps, name, kernel[1], cutoffs)
        like_p, _ = \
            get_like_grad_from_mats(ky_mat_p, hyp_mat_p, name)
        newhyps[i] -= 2*delta
        hyp_mat_m, ky_mat_m = func(newhyps, name, kernel[1], cutoffs)
        like_m, _ = \
            get_like_grad_from_mats(ky_mat_m, hyp_mat_m, name)
        diff = np.abs(like_grad[i]-(like_p-like_m)/2./delta)
        assert (diff < 1e-3), "wrong calculation of hyp_mat"
