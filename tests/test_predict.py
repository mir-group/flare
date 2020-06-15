"""
Helper functions which obtain forces and energies
corresponding to atoms in structures. These functions automatically
cast atoms into their respective atomic environments.
"""
import numpy as np
from flare.gp import GaussianProcess
from flare.struc import Structure

from .fake_gp import generate_hm, get_tstp, get_random_structure
from flare.predict import predict_on_structure, predict_on_structure_par, \
    predict_on_structure_efs, predict_on_structure_efs_par
import pytest
import time


@pytest.fixture(scope='class')
def two_plus_three_gp() -> GaussianProcess:
    """Returns a GP instance with a 2+3-body kernel."""

    cutoffs = np.array([0.8, 0.8])
    hyps = np.array([1., 1., 1., 1., 1.])

    # test update_db
    gpname = '2+3_mc'
    cutoffs = np.ones(2)*0.8

    gp_model = \
        GaussianProcess(kernel_name=gpname, hyps=hyps, cutoffs=cutoffs,
                        multihyps=False, parallel=False, n_cpus=1)

    test_structure, forces = \
        get_random_structure(np.eye(3), [1, 2], 3)
    energy = 3.14

    gp_model.update_db(test_structure, forces, energy=energy)

    yield gp_model
    del gp_model


def fake_predict(x, d):
    return np.random.uniform(-1, 1), np.random.uniform(-1, 1)


_fake_gp = GaussianProcess(kernel_name='2+3', cutoffs=[5., 5.],
                           hyps=[1, 1, 1])
_fake_structure = Structure(cell=np.eye(3), species=[1, 1, 1],
                            positions=np.random.uniform(0, 1, size=(3, 3)))
_fake_gp.predict = fake_predict


@pytest.mark.parametrize('n_cpu', [1, 2])
def test_predict_on_structure_par(n_cpu):

    # Predict only on the first atom, and make rest NAN
    selective_atoms = [0]

    skipped_atom_value = np.nan

    forces, stds = \
        predict_on_structure_par(_fake_structure, _fake_gp, n_cpus=n_cpu,
                                 write_to_structure=False,
                                 selective_atoms=selective_atoms,
                                 skipped_atom_value=skipped_atom_value)

    for x in forces[0][:]:
        assert isinstance(x, float)
    for x in forces[1:]:
        assert np.isnan(x).all()

    # Predict only on the second and third, and make rest 0
    selective_atoms = [1, 2]
    skipped_atom_value = 0

    forces, stds = \
        predict_on_structure_par(_fake_structure, _fake_gp,
                                 write_to_structure=False, n_cpus=n_cpu,
                                 selective_atoms=selective_atoms,
                                 skipped_atom_value=skipped_atom_value)

    for x in forces[1]:
        assert isinstance(x, float)
    for x in forces[2]:
        assert isinstance(x, float)

    assert np.equal(forces[0], 0).all()

    # Make selective atoms be all and ensure results are normal
    selective_atoms = [0, 1, 2]

    forces, stds = \
        predict_on_structure_par(_fake_structure, _fake_gp,
                                 write_to_structure=True, n_cpus=n_cpu,
                                 selective_atoms=selective_atoms,
                                 skipped_atom_value=skipped_atom_value)

    for x in forces.flatten():
        assert isinstance(x, float)
    for x in stds.flatten():
        assert isinstance(x, float)

    assert np.array_equal(_fake_structure.forces, forces)
    assert np.array_equal(_fake_structure.stds, stds)

    # Get new examples to also test the results not being written
    selective_atoms = [0, 1]

    forces, stds = \
        predict_on_structure_par(_fake_structure, _fake_gp,
                                 write_to_structure=True, n_cpus=n_cpu,
                                 selective_atoms=selective_atoms,
                                 skipped_atom_value=skipped_atom_value)

    for x in forces.flatten():
        assert isinstance(x, float)

    for x in stds.flatten():
        assert isinstance(x, float)

    assert np.array_equal(_fake_structure.forces[:2][:], forces[:2][:])
    assert not np.array_equal(_fake_structure.forces[2][:], forces[2][:])

    assert np.array_equal(_fake_structure.stds[:2][:], stds[:2][:])
    assert not np.array_equal(_fake_structure.stds[2][:], stds[2][:])


def test_predict_efs(two_plus_three_gp):
    test_structure, _ = \
        get_random_structure(np.eye(3), [1, 2], 3)

    ens, forces, stresses, en_stds, force_stds, stress_stds = \
        predict_on_structure_efs(test_structure, two_plus_three_gp)

    forces_2, force_stds_2 = \
        predict_on_structure(test_structure, two_plus_three_gp)

    ens_p, forces_p, stresses_p, en_stds_p, force_stds_p, stress_stds_p = \
        predict_on_structure_efs_par(test_structure, two_plus_three_gp)

    # Check agreement between efs and standard prediction.
    assert (np.isclose(forces, forces_2).all())
    assert (np.isclose(force_stds, force_stds_2).all())

    # Check agreement between serial and parallel efs methods.
    assert (np.isclose(ens, ens_p).all())
    assert (np.isclose(forces, forces_p).all())
    assert (np.isclose(stresses, stresses_p).all())
    assert (np.isclose(en_stds, en_stds_p).all())
    assert (np.isclose(force_stds, force_stds_p).all())
    assert (np.isclose(stress_stds, stress_stds_p).all())

    # Check that the prediction has been recorded within the structure.
    assert(np.equal(stress_stds, test_structure.partial_stress_stds).all())
