"""
Helper functions which obtain forces and energies
corresponding to atoms in structures. These functions automatically
cast atoms into their respective atomic environments.
"""
import numpy as np
from flare.bffs.gp import GaussianProcess
from flare.atoms import FLARE_Atoms
from copy import deepcopy
from flare.bffs.gp.predict import (
    predict_on_structure_par,
    predict_on_atom,
    predict_on_atom_en,
    predict_on_structure_par_en,
)

from .fake_gp import generate_hm, get_tstp, get_random_structure
from flare.bffs.gp.predict import (
    predict_on_structure,
    predict_on_structure_par,
    predict_on_structure_efs,
    predict_on_structure_efs_par,
)
import pytest
import time


def fake_predict(_, __):
    return np.random.uniform(-1, 1), np.random.uniform(-1, 1)


def fake_predict_local_energy(_):
    return np.random.uniform(-1, 1)


@pytest.fixture(scope="class")
def two_plus_three_gp() -> GaussianProcess:
    """Returns a GP instance with a 2+3-body kernel."""
    cutoffs = {"twobody": 0.8, "threebody": 0.8}
    hyps = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    gp_model = GaussianProcess(
        kernels=["twobody", "threebody"],
        hyps=hyps,
        cutoffs=cutoffs,
        multihyps=False,
        parallel=False,
        n_cpus=1,
    )

    test_structure, forces = get_random_structure(np.eye(3), [1, 2], 3)
    energy = 3.14

    gp_model.update_db(test_structure, forces, energy=energy)

    yield gp_model
    del gp_model


_fake_gp = GaussianProcess(
    kernel_name="2+3", cutoffs=[5.0, 5.0], hyps=[1.0, 1.0, 1.0, 1.0, 1.0]
)
_fake_structure = FLARE_Atoms(
    cell=np.eye(3), symbols=[1, 1, 1], positions=np.random.uniform(0, 1, size=(3, 3))
)
_fake_gp.predict = fake_predict
_fake_gp.predict_local_energy = fake_predict_local_energy

assert isinstance(_fake_gp.predict(1, 1), tuple)
assert isinstance(_fake_gp.predict_local_energy(1), float)


@pytest.mark.parametrize("n_cpu", [None, 1, 2])
def test_predict_on_structure_par(n_cpu):
    # Predict only on the first atom, and make rest NAN
    selective_atoms = [0]

    skipped_atom_value = np.nan

    forces, stds = predict_on_structure_par(
        _fake_structure,
        _fake_gp,
        n_cpus=n_cpu,
        write_to_structure=False,
        selective_atoms=selective_atoms,
        skipped_atom_value=skipped_atom_value,
    )

    for x in forces[0][:]:
        assert isinstance(x, float)
    for x in forces[1:]:
        assert np.isnan(x).all()

    # Predict only on the second and third, and make rest 0
    selective_atoms = [1, 2]
    skipped_atom_value = 0

    forces, stds = predict_on_structure_par(
        _fake_structure,
        _fake_gp,
        write_to_structure=False,
        n_cpus=n_cpu,
        selective_atoms=selective_atoms,
        skipped_atom_value=skipped_atom_value,
    )

    for x in forces[1]:
        assert isinstance(x, float)
    for x in forces[2]:
        assert isinstance(x, float)

    assert np.equal(forces[0], 0).all()

    # Make selective atoms be all and ensure results are normal
    selective_atoms = [0, 1, 2]

    forces, stds = predict_on_structure_par(
        _fake_structure,
        _fake_gp,
        write_to_structure=True,
        n_cpus=n_cpu,
        selective_atoms=selective_atoms,
        skipped_atom_value=skipped_atom_value,
    )

    for x in forces.flatten():
        assert isinstance(x, float)
    for x in stds.flatten():
        assert isinstance(x, float)

    assert np.array_equal(_fake_structure.forces, forces)
    assert np.array_equal(_fake_structure.stds, stds)

    # Make selective atoms be nothing and ensure results are normal

    forces, stds = predict_on_structure_par(
        _fake_structure,
        _fake_gp,
        write_to_structure=True,
        n_cpus=n_cpu,
        selective_atoms=None,
        skipped_atom_value=skipped_atom_value,
    )

    for x in forces.flatten():
        assert isinstance(x, float)
    for x in stds.flatten():
        assert isinstance(x, float)

    assert np.array_equal(_fake_structure.forces, forces)
    assert np.array_equal(_fake_structure.stds, stds)

    # Get new examples to also test the results not being written
    selective_atoms = [0, 1]

    forces, stds = predict_on_structure_par(
        _fake_structure,
        _fake_gp,
        write_to_structure=True,
        n_cpus=n_cpu,
        selective_atoms=selective_atoms,
        skipped_atom_value=skipped_atom_value,
    )

    for x in forces.flatten():
        assert isinstance(x, float)

    for x in stds.flatten():
        assert isinstance(x, float)

    assert np.array_equal(_fake_structure.forces, forces)
    assert np.array_equal(_fake_structure.stds, stds)


def test_predict_efs(two_plus_three_gp):
    test_structure, _ = get_random_structure(np.eye(3), [1, 2], 3)

    ens, forces, stresses, en_stds, force_stds, stress_stds = predict_on_structure_efs(
        test_structure, two_plus_three_gp
    )

    forces_2, force_stds_2 = predict_on_structure(test_structure, two_plus_three_gp)

    (
        ens_p,
        forces_p,
        stresses_p,
        en_stds_p,
        force_stds_p,
        stress_stds_p,
    ) = predict_on_structure_efs_par(test_structure, two_plus_three_gp)

    # Check agreement between efs and standard prediction.
    assert np.isclose(forces, forces_2).all()
    assert np.isclose(force_stds, force_stds_2).all()

    # Check agreement between serial and parallel efs methods.
    assert np.isclose(ens, ens_p).all()
    assert np.isclose(forces, forces_p).all()
    assert np.isclose(stresses, stresses_p).all()
    assert np.isclose(en_stds, en_stds_p).all()
    assert np.isclose(force_stds, force_stds_p).all()
    assert np.isclose(stress_stds, stress_stds_p).all()

    # Check that the prediction has been recorded within the structure.
    assert np.equal(stress_stds, test_structure.partial_stress_stds).all()


def test_predict_on_atoms():
    pred_at_result = predict_on_atom((_fake_structure, 0, _fake_gp))
    assert len(pred_at_result) == 2
    assert len(pred_at_result[0]) == len(pred_at_result[1]) == 3

    # Test results are correctly compiled into np arrays
    pred_at_en_result = predict_on_atom_en((_fake_structure, 0, _fake_gp))
    assert isinstance(pred_at_en_result[0], np.ndarray)
    assert isinstance(pred_at_en_result[1], np.ndarray)

    # Test 3 things are returned; two vectors of length 3
    assert len(pred_at_en_result) == 3
    assert len(pred_at_en_result[0]) == len(pred_at_result[1]) == 3
    assert isinstance(pred_at_en_result[2], float)


@pytest.mark.parametrize("n_cpus", [1, 2, None])
@pytest.mark.parametrize(
    ["write_to_structure", "selective_atoms"],
    [(True, []), (True, [1]), (False, []), (False, [1])],
)
def test_predict_on_structure_en(n_cpus, write_to_structure, selective_atoms):
    old_structure = deepcopy(_fake_structure)

    old_structure.forces = np.random.uniform(-1, 1, (3, 3))

    used_structure = deepcopy(old_structure)

    forces, stds, energies = predict_on_structure_par_en(
        structure=used_structure,
        gp=_fake_gp,
        n_cpus=n_cpus,
        write_to_structure=write_to_structure,
        selective_atoms=selective_atoms,
        skipped_atom_value=0,
    )

    assert np.array_equal(forces.shape, old_structure.positions.shape)
    assert np.array_equal(stds.shape, old_structure.positions.shape)

    if write_to_structure:

        if selective_atoms == [1]:
            assert np.array_equal(np.zeros(3), used_structure.forces[0])
            assert np.array_equal(np.zeros(3), used_structure.forces[2])

            assert np.array_equal(used_structure.forces[1], forces[1])

        else:
            assert not np.array_equal(old_structure.forces[0], used_structure.forces[0])
            assert not np.array_equal(old_structure.forces[2], used_structure.forces[2])

            assert np.array_equal(forces, used_structure.forces)

        # These will be unequal no matter what
        assert not np.array_equal(old_structure.forces[1], used_structure.forces[1])

    else:
        assert np.array_equal(old_structure.forces, used_structure.forces)

    assert np.array_equal(forces.shape, (3, 3))
    assert np.array_equal(stds.shape, (3, 3))
    assert len(energies) == len(old_structure)
