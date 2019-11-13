import pytest
import numpy as np
import pickle

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
    os.system('rm ./gp_from_aimd*')


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
    os.system('rm ./gp_from_aimd*')


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
                           skip=10,
                           pre_train_seed_envs=seeds,
                           pre_train_seed_frames=[frames[-1]],
                           max_atoms_from_frame=4,
                           output_name='meth_test',
                           model_format='pickle',
                           checkpoint_interval=1,
                           pre_train_atoms_per_element={'H': 1})

    tt.run()

    with open('meth_test_model.pickle', 'rb') as f:
        new_gp = pickle.load(f)

    test_env = envs[0]

    for d in [0, 1, 2]:
        assert np.all(the_gp.predict(x_t=test_env, d=d) ==
                      new_gp.predict(x_t=test_env, d=d))

    os.system('rm ./meth_test*')
