import pytest
import numpy as np
import sys
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.kernels import two_plus_three_body, two_plus_three_body_grad
from flare.gp import GaussianProcess
from flare.gp_from_aimd import TrajectoryTrainer
from flare.mc_simple import two_plus_three_body_mc, two_plus_three_body_mc_grad
from json import loads
import os


@pytest.fixture
def methanol_gp():
    the_gp = GaussianProcess(kernel=two_plus_three_body_mc,
                             kernel_grad=two_plus_three_body_mc_grad,
                             hyps=np.array([3.75996759e-06, 1.53990678e-02,
                                            2.50624782e-05, 5.07884426e-01,
                                            1.70172923e-03]),
                             cutoffs=np.array([7, 7]),
                             hyp_labels=['l2', 's2', 'l3', 's3', 'n0'],
                             maxiter=1,
                             opt_algorithm='L-BFGS-B')
    with open('./test_files/methanol_envs.json') as f:
        dicts = [loads(s) for s in f.readlines()]

    for cur_dict in dicts:
        force = cur_dict['forces']
        env = AtomicEnvironment.from_dict(cur_dict)
        the_gp.add_one_env(env, force)

    the_gp.set_L_alpha()

    return the_gp


@pytest.fixture
def fake_gp():
    return GaussianProcess(kernel=two_plus_three_body,
                           kernel_grad=two_plus_three_body_grad,
                           hyps=np.array([1]),
                           cutoffs=np.array([]))


def test_instantiation_of_trajectory_trainer(fake_gp):
    a = TrajectoryTrainer(frames=[], gp=fake_gp)

    assert isinstance(a, TrajectoryTrainer)

    _ = TrajectoryTrainer([], fake_gp, parallel=True, calculate_energy=True)
    _ = TrajectoryTrainer([], fake_gp, parallel=True, calculate_energy=False)
    _ = TrajectoryTrainer([], fake_gp, parallel=False, calculate_energy=True)
    _ = TrajectoryTrainer([], fake_gp, parallel=False, calculate_energy=False)


def test_load_trained_gp_and_run(methanol_gp):
    with open('./test_files/methanol_frames.json', 'r') as f:
        frames = [Structure.from_dict(loads(s)) for s in f.readlines()]

    tt = TrajectoryTrainer(frames,
                           gp=methanol_gp,
                           rel_std_tolerance=0,
                           abs_std_tolerance=0,
                           skip=15)

    tt.run()
    os.system('rm ./gp_from_aimd.out')
    os.system('rm ./gp_from_aimd.xyz')
    os.system('rm ./gp_from_aimd-f.xyz')

def test_load_one_frame_and_run():
    the_gp = GaussianProcess(kernel=two_plus_three_body_mc,
                             kernel_grad=two_plus_three_body_mc_grad,
                             hyps=np.array([3.75996759e-06, 1.53990678e-02,
                                            2.50624782e-05, 5.07884426e-01,
                                            1.70172923e-03]),
                             cutoffs=np.array([7, 7]),
                             hyp_labels=['l2', 's2', 'l3', 's3', 'n0'],
                             maxiter=1,
                             opt_algorithm='L-BFGS-B')

    with open('./test_files/methanol_frames.json', 'r') as f:
        frames = [Structure.from_dict(loads(s)) for s in f.readlines()]

    tt = TrajectoryTrainer(frames,
                           gp=the_gp, shuffle_frames=True,
                           rel_std_tolerance=0,
                           abs_std_tolerance=0,
                           skip=15)

    tt.run()
    os.system('rm ./gp_from_aimd.out')
    os.system('rm ./gp_from_aimd.xyz')
    os.system('rm ./gp_from_aimd-f.xyz')


def test_seed_and_run():
        the_gp = GaussianProcess(kernel=two_plus_three_body_mc,
                                 kernel_grad=two_plus_three_body_mc_grad,
                                 hyps=np.array([3.75996759e-06, 1.53990678e-02,
                                                2.50624782e-05, 5.07884426e-01,
                                                1.70172923e-03]),
                                 cutoffs=np.array([7, 7]),
                                 hyp_labels=['l2', 's2', 'l3', 's3', 'n0'],
                                 maxiter=1,
                                 opt_algorithm='L-BFGS-B')

        with open('./test_files/methanol_frames.json', 'r') as f:
            frames = [Structure.from_dict(loads(s)) for s in f.readlines()]

        with open('./test_files/methanol_envs.json', 'r') as f:
            data_dicts = [loads(s) for s in f.readlines()[:6]]
            envs = [AtomicEnvironment.from_dict(d) for d in data_dicts]
            forces = [np.array(d['forces']) for d in data_dicts]
            seeds = list(zip(envs, forces))

        tt = TrajectoryTrainer(frames,
                               gp=the_gp, shuffle_frames=True,
                               rel_std_tolerance=0,
                               abs_std_tolerance=0,
                               skip=15,
                               pre_train_seed_envs=seeds,
                               pre_train_seed_frames=[frames[-1]],
                               model_write='test_methanol_gp.pickle',
                               max_atoms_from_frame=4)

        tt.run()

        os.system('rm ./gp_from_aimd.out')
        os.system('rm ./gp_from_aimd.xyz')
        os.system('rm ./gp_from_aimd-f.xyz')
        os.system('rm ./test_methanol_gp.pickle')


def test_uncertainty_threshold(fake_gp):
    tt = TrajectoryTrainer([], fake_gp, rel_std_tolerance=.5,
                           abs_std_tolerance=.01)

    fake_structure = Structure(cell=np.eye(3), species=["H"],
                               positions=np.array([[0, 0, 0]]))

    # Test a structure with no variance passes
    fake_structure.stds = np.array([[0, 0, 0]])

    res1, res2 = tt.is_std_in_bound(fake_structure)
    assert res1 is True
    assert res2 == [-1]

    # Test that the absolute criteria trips the threshold
    fake_structure.stds = np.array([[.02, 0, 0]])

    res1, res2 = tt.is_std_in_bound(fake_structure)
    assert res1 is False
    assert res2 == [0]

    tt.abs_std_tolerance = 100

    # Test that the relative criteria trips the threshold
    fake_structure.stds = np.array([[.6, 0, 0]])

    res1, res2 = tt.is_std_in_bound(fake_structure)
    assert res1 is False
    assert res2 == [0]

    # Test that 'test mode' works, where no GP modification occurs
    tt.abs_std_tolerance = 0
    tt.rel_std_tolerance = 0

    res1, res2 = tt.is_std_in_bound(fake_structure)
    assert res1 is True
    assert res2 == [-1]

    # Test permutations of one / another being off
    tt.abs_std_tolerance = 1
    tt.rel_std_tolerance = 0

    res1, res2 = tt.is_std_in_bound(fake_structure)
    assert res1 is True
    assert res2 == [-1]

    tt.abs_std_tolerance = 0
    tt.rel_std_tolerance = 1

    res1, res2 = tt.is_std_in_bound(fake_structure)
    assert res1 is True
    assert res2 == [-1]
