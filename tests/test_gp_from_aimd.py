import numpy as np
import pickle
import pytest
import json as json

from copy import deepcopy
from json import loads
from glob import glob
from os import remove, path

from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.gp import GaussianProcess
from flare.mgp import MappedGaussianProcess
from flare.gp_from_aimd import TrajectoryTrainer, parse_trajectory_trainer_output
from flare.utils.learner import subset_of_frame_by_element

from tests.test_mgp import all_mgp, all_gp, get_random_structure
from .fake_gp import get_gp


TEST_DIR = path.dirname(__file__)
TEST_FILE_DIR = path.join(TEST_DIR, "test_files")


@pytest.fixture
def methanol_gp():
    the_gp = GaussianProcess(
        kernel_name="2+3_mc",
        hyps=np.array(
            [
                3.75996759e-06,
                1.53990678e-02,
                2.50624782e-05,
                5.07884426e-01,
                1.70172923e-03,
            ]
        ),
        cutoffs=np.array([5, 3]),
        hyp_labels=["l2", "s2", "l3", "s3", "n0"],
        maxiter=1,
        opt_algorithm="L-BFGS-B",
    )

    with open(path.join(TEST_FILE_DIR, "methanol_envs.json"), "r") as f:
        dicts = [loads(s) for s in f.readlines()]

    for cur_dict in dicts:
        force = cur_dict["forces"]
        env = AtomicEnvironment.from_dict(cur_dict)
        the_gp.add_one_env(env, force)

    the_gp.set_L_alpha()

    return the_gp


@pytest.fixture
def fake_gp():
    return GaussianProcess(
        kernel_name="2+3", hyps=np.array([1, 1, 1, 1, 1]), cutoffs=np.array([4, 3])
    )


def test_instantiation_of_trajectory_trainer(fake_gp):
    a = TrajectoryTrainer(frames=[], gp=fake_gp)

    assert isinstance(a, TrajectoryTrainer)

    fake_gp.parallel = True
    _ = TrajectoryTrainer([], fake_gp, n_cpus=2, calculate_energy=True)
    _ = TrajectoryTrainer([], fake_gp, n_cpus=2, calculate_energy=False)

    fake_gp.parallel = False
    _ = TrajectoryTrainer([], fake_gp, n_cpus=2, calculate_energy=True)
    _ = TrajectoryTrainer([], fake_gp, n_cpus=2, calculate_energy=False)


def test_load_trained_gp_and_run(methanol_gp):
    with open(path.join(TEST_FILE_DIR, "methanol_frames.json"), "r") as f:
        frames = [Structure.from_dict(loads(s)) for s in f.readlines()]

    tt = TrajectoryTrainer(
        frames,
        gp=methanol_gp,
        rel_std_tolerance=0,
        abs_std_tolerance=0,
        skip=15,
        train_checkpoint_interval=10,
    )

    tt.run()
    for f in glob(f"gp_from_aimd*"):
        remove(f)


def test_load_one_frame_and_run():
    the_gp = GaussianProcess(
        kernel_name="2+3_mc",
        hyps=np.array(
            [
                3.75996759e-06,
                1.53990678e-02,
                2.50624782e-05,
                5.07884426e-01,
                1.70172923e-03,
            ]
        ),
        cutoffs=np.array([5, 3]),
        hyp_labels=["l2", "s2", "l3", "s3", "n0"],
        maxiter=1,
        opt_algorithm="L-BFGS-B",
    )

    with open(path.join(TEST_FILE_DIR, "methanol_frames.json"), "r") as f:
        frames = [Structure.from_dict(loads(s)) for s in f.readlines()]

    tt = TrajectoryTrainer(
        frames,
        gp=the_gp,
        shuffle_frames=True,
        print_as_xyz=True,
        rel_std_tolerance=0,
        abs_std_tolerance=0,
        skip=15,
    )

    tt.run()
    for f in glob(f"gp_from_aimd*"):
        remove(f)


def test_seed_and_run():
    the_gp = GaussianProcess(
        kernel_name="2+3_mc",
        hyps=np.array(
            [
                3.75996759e-06,
                1.53990678e-02,
                2.50624782e-05,
                5.07884426e-01,
                1.70172923e-03,
            ]
        ),
        cutoffs=np.array([5, 3]),
        hyp_labels=["l2", "s2", "l3", "s3", "n0"],
        maxiter=1,
        opt_algorithm="L-BFGS-B",
    )

    with open(path.join(TEST_FILE_DIR, "methanol_frames.json"), "r") as f:
        frames = [Structure.from_dict(loads(s)) for s in f.readlines()]

    with open(path.join(TEST_FILE_DIR, "methanol_envs.json"), "r") as f:
        data_dicts = [loads(s) for s in f.readlines()[:6]]
        envs = [AtomicEnvironment.from_dict(d) for d in data_dicts]
        forces = [np.array(d["forces"]) for d in data_dicts]
        seeds = list(zip(envs, forces))

    tt = TrajectoryTrainer(
        frames,
        gp=the_gp,
        shuffle_frames=True,
        rel_std_tolerance=0,
        abs_std_tolerance=0,
        skip=10,
        pre_train_seed_envs=seeds,
        pre_train_seed_frames=[frames[-1]],
        max_atoms_from_frame=4,
        output_name="meth_test",
        model_format="pickle",
        train_checkpoint_interval=1,
        pre_train_atoms_per_element={"H": 1},
    )

    tt.run()

    with open("meth_test_model.pickle", "rb") as f:
        new_gp = pickle.load(f)

    test_env = envs[0]

    for d in [1, 2, 3]:
        assert np.all(
            the_gp.predict(x_t=test_env, d=d) == new_gp.predict(x_t=test_env, d=d)
        )

    for f in glob(f"meth_test*"):
        remove(f)


def test_pred_on_elements():
    the_gp = GaussianProcess(
        kernel_name="2+3_mc",
        hyps=np.array(
            [
                3.75996759e-06,
                1.53990678e-02,
                2.50624782e-05,
                5.07884426e-01,
                1.70172923e-03,
            ]
        ),
        cutoffs=np.array([5, 3]),
        hyp_labels=["l2", "s2", "l3", "s3", "n0"],
        maxiter=1,
        opt_algorithm="L-BFGS-B",
    )

    with open(path.join(TEST_FILE_DIR, "methanol_frames.json"), "r") as f:
        frames = [Structure.from_dict(loads(s)) for s in f.readlines()]

    with open(path.join(TEST_FILE_DIR, "methanol_envs.json"), "r") as f:
        data_dicts = [loads(s) for s in f.readlines()[:6]]
        envs = [AtomicEnvironment.from_dict(d) for d in data_dicts]
        forces = [np.array(d["forces"]) for d in data_dicts]
        seeds = list(zip(envs, forces))

    all_frames = deepcopy(frames)
    tt = TrajectoryTrainer(
        frames,
        gp=the_gp,
        shuffle_frames=False,
        rel_std_tolerance=0,
        abs_std_tolerance=0,
        abs_force_tolerance=0.001,
        skip=5,
        min_atoms_per_train=100,
        pre_train_seed_envs=seeds,
        pre_train_seed_frames=[frames[-1]],
        max_atoms_from_frame=4,
        output_name="meth_test",
        print_as_xyz=True,
        model_format="json",
        atom_checkpoint_interval=50,
        pre_train_atoms_per_element={"H": 1},
        predict_atoms_per_element={"H": 0, "C": 1, "O": 0},
    )
    # Set to predict only on Carbon after training on H to ensure errors are
    #  high and that they get added to the gp
    tt.run()

    # Ensure forces weren't written directly to structure
    for i in range(len(all_frames)):
        assert np.array_equal(all_frames[i].forces, frames[i].forces)

    # Assert that Carbon atoms were correctly added
    assert the_gp.training_statistics["envs_by_species"]["C"] > 2

    # for f in glob(f"meth_test*"):
    #    remove(f)

    # for f in glob(f"gp_from_aimd*"):
    #    remove(f)


def test_mgp_gpfa(all_mgp, all_gp):
    """
    Ensure that passing in an MGP also works for the trajectory trainer
    :param all_mgp:
    :param all_gp:
    :return:
    """

    np.random.seed(10)
    gp_model = get_gp("3", "mc", False)
    gp_model.set_L_alpha()

    grid_num_2 = 5
    grid_num_3 = 3
    lower_cut = 0.01
    grid_params_3b = {
        "lower_bound": [lower_cut for d in range(3)],
        "grid_num": [grid_num_3 for d in range(3)],
        "svd_rank": "auto",
    }
    grid_params = {"load_grid": None, "update": False}
    grid_params["threebody"] = grid_params_3b
    unique_species = gp_model.training_statistics["species"]

    mgp_model = MappedGaussianProcess(
        grid_params=grid_params, unique_species=unique_species, n_cpus=1
    )

    mgp_model.build_map(gp_model)

    nenv = 10
    cell = np.eye(3)
    struc, f = get_random_structure(cell, unique_species, nenv)

    struc.forces = np.array(f)

    frames = [struc]

    tt = TrajectoryTrainer(
        frames,
        mgp_model,
        rel_std_tolerance=0,
        abs_std_tolerance=0,
        abs_force_tolerance=1e-8,
        print_training_plan=True,
    )
    assert tt.mgp is True
    tt.run()

    # Test that training plan is properly written
    with open("gp_from_aimd_training_plan.json", "r") as f:
        plan = json.loads(f.readline())
    assert isinstance(plan["0"], list)
    assert len(plan["0"]) == len(struc)
    assert [p[0] for p in plan["0"]] == list(range(len(struc)))

    for f in glob(f"gp_from_aimd*"):
        remove(f)


def test_parse_gpfa_output():
    """
    Compare parsing against known answers.
    :return:
    """
    frames, gp_data = parse_trajectory_trainer_output(
        path.join(TEST_FILE_DIR, "gpfa_parse_test.out"), True
    )

    assert len(frames) == 5
    assert isinstance(frames[0], dict)
    for frame in frames:
        for key in ["species", "positions", "gp_forces", "dft_forces", "gp_stds"]:

            assert len(frame[key]) == 6

        assert len(frame["added_atoms"]) == 0 or len(frame["added_atoms"]) == 1

        assert frame["maes_by_species"]["C"]
        assert frame["maes_by_species"].get("H") is None

    assert gp_data["init_stats"]["N"] == 0
    assert gp_data["init_stats"]["species"] == []
    assert gp_data["init_stats"]["envs_by_species"] == {}

    assert gp_data["cumulative_gp_size"][-1] > 2
    assert len(gp_data["mae_by_elt"]["C"]) == 5

    assert gp_data["pre_train_stats"]["N"] == 9
    assert gp_data["pre_train_stats"]["envs_by_species"]["C"] == 2
    assert gp_data["pre_train_stats"]["envs_by_species"]["H"] == 5
    assert gp_data["pre_train_stats"]["envs_by_species"]["O"] == 2
    assert gp_data["pre_train_stats"]["species"] == ["H", "C", "O"]

    assert gp_data["cumulative_gp_size"] == [0, 9, 9, 10, 11, 12, 13]
