import numpy as np
import pickle
import pytest
import json as json

from copy import deepcopy
from json import loads
from glob import glob
from os import remove, path

from flare.descriptors.env import AtomicEnvironment
from flare.atoms import FLARE_Atoms
from flare.bffs.gp import GaussianProcess
from flare.bffs.mgp import MappedGaussianProcess
from flare.learners.gp_from_aimd import (
    TrajectoryTrainer,
    parse_trajectory_trainer_output,
    structures_from_gpfa_output,
)
from flare.learners.utils import subset_of_frame_by_element
from ase.data import chemical_symbols
from ase.io import read

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
    frames = read(path.join(TEST_FILE_DIR, "methanol_frames.json"), index=":")
    frames = [FLARE_Atoms.from_ase_atoms(f, copy_calc_results=True) for f in frames]

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

    frames = read(path.join(TEST_FILE_DIR, "methanol_frames.json"), index=":")
    frames = [FLARE_Atoms.from_ase_atoms(f, copy_calc_results=True) for f in frames]

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

    frames = read(path.join(TEST_FILE_DIR, "methanol_frames.json"), index=":")
    frames = [FLARE_Atoms.from_ase_atoms(f, copy_calc_results=True) for f in frames]

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

    frames = read(path.join(TEST_FILE_DIR, "methanol_frames.json"), index=":")
    frames = [FLARE_Atoms.from_ase_atoms(f, copy_calc_results=True) for f in frames]

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

    for f in glob(f"meth_test*"):
        remove(f)

    for f in glob(f"gp_from_aimd*"):
        remove(f)


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

    grid_num_3 = 3
    lower_cut = 0.01
    grid_params_3b = {
        "lower_bound": [lower_cut] * 3,
        "grid_num": [grid_num_3] * 3,
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
    assert tt.gp_is_mapped is True
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
    Answer is based off of the result of the unit test `test_pred_on_elements`.
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
    assert set(gp_data["pre_train_stats"]["species"]) == set(["H", "C", "O"])

    assert gp_data["cumulative_gp_size"] == [0, 9, 9, 10, 11, 12, 13]

    # Ensure that structures can correctly be read from the GPFA output
    structures = structures_from_gpfa_output(frames)
    for struc, frame in zip(structures, frames):
        assert np.array_equal(struc.symbols, frame["species"])
        assert np.array_equal(struc.positions, frame["positions"])
        assert np.array_equal(struc.forces, frame["dft_forces"])


def test_passive_learning():
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

    frames = read(path.join(TEST_FILE_DIR, "methanol_frames.json"), index=":")
    frames = [FLARE_Atoms.from_ase_atoms(f, copy_calc_results=True) for f in frames]

    envs = AtomicEnvironment.from_file(path.join(TEST_FILE_DIR, "methanol_envs.json"))
    cur_gp = deepcopy(the_gp)
    tt = TrajectoryTrainer(frames=None, gp=cur_gp)

    # TEST ENVIRONMENT ADDITION
    envs_species = set(chemical_symbols[env.ctype] for env in envs)
    tt.run_passive_learning(environments=envs, post_build_matrices=False)

    assert cur_gp.training_statistics["N"] == len(envs)
    assert set(cur_gp.training_statistics["species"]) == envs_species

    # TEST FRAME ADDITION: ALL ARE ADDED
    cur_gp = deepcopy(the_gp)
    tt.gp = cur_gp
    tt.run_passive_learning(frames=frames, post_build_matrices=False)
    assert len(cur_gp.training_data) == sum([len(fr) for fr in frames])

    # TEST FRAME ADDITION: MAX OUT MODEL SIZE AT 1
    cur_gp = deepcopy(the_gp)
    tt.gp = cur_gp
    tt.run_passive_learning(frames=frames, max_model_size=1, post_training_iterations=1)
    assert len(cur_gp.training_data) == 1

    # TEST FRAME ADDITION: EXCLUDE OXYGEN, LIMIT CARBON TO 1, 1 H PER FRAME
    cur_gp = deepcopy(the_gp)
    tt.gp = cur_gp
    tt.run_passive_learning(
        frames=frames,
        max_model_elts={"O": 0, "C": 1, "H": 5},
        max_elts_per_frame={"H": 1},
        post_build_matrices=False,
    )

    assert "O" not in cur_gp.training_statistics["species"]
    assert cur_gp.training_statistics["envs_by_species"]["C"] == 1
    assert cur_gp.training_statistics["envs_by_species"]["H"] == 5


def test_active_learning_simple_run():
    """
    Test simple mechanics of active learning method.
    :return:
    """

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

    frames = read(path.join(TEST_FILE_DIR, "methanol_frames.json"), index=":")
    frames = [FLARE_Atoms.from_ase_atoms(f, copy_calc_results=True) for f in frames]

    # Assign fake energies to structures
    for frame in frames:
        frame.energy = np.random.random()

    tt = TrajectoryTrainer(gp=the_gp, include_energies=True)

    tt.run_passive_learning(
        frames=frames[:1],
        max_elts_per_frame={"C": 1, "O": 1, "H": 1},
        post_training_iterations=0,
        post_build_matrices=True,
    )

    assert len(the_gp.training_structures) == 1
    prev_gp_len = len(the_gp)
    prev_gp_stats = the_gp.training_statistics
    tt.run_active_learning(
        frames[:2], rel_std_tolerance=0, abs_std_tolerance=0, abs_force_tolerance=0
    )
    assert len(the_gp) == prev_gp_len
    # Try on a frame where the Carbon atom is guaranteed to trip the
    # abs. force tolerance condition.
    # Turn off include energies so that the number of training structures
    # does not change.
    tt.include_energies = False
    tt.run_active_learning(
        frames[1:2],
        rel_std_tolerance=0,
        abs_std_tolerance=0,
        abs_force_tolerance=0.1,
        max_elts_per_frame={"H": 0, "O": 0},
        max_model_elts={"C": 2},
    )
    assert len(the_gp) == prev_gp_len + 1
    assert len(the_gp.training_structures) == 1
    prev_carbon_atoms = prev_gp_stats["envs_by_species"]["C"]
    assert the_gp.training_statistics["envs_by_species"]["C"] == prev_carbon_atoms + 1

    prev_gp_len = len(the_gp)
    tt.run_active_learning(
        frames[3:4],
        rel_std_tolerance=0,
        abs_std_tolerance=0,
        abs_force_tolerance=0.1,
        max_model_size=prev_gp_len,
    )
    assert len(the_gp) == prev_gp_len

    # Test that model doesn't add atoms
    prev_gp_len = len(the_gp)
    tt.run_active_learning(
        frames[5:6],
        rel_std_tolerance=0,
        abs_std_tolerance=0,
        abs_force_tolerance=0.1,
        max_model_elts={"C": 2, "H": 1, "O": 1},
    )
    assert len(the_gp) == prev_gp_len

    for f in glob(f"gp_from_aimd*"):
        remove(f)
