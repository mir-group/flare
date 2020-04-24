"""
Helper functions which obtain forces and energies
corresponding to atoms in structures. These functions automatically
cast atoms into their respective atomic environments.
"""
import numpy as np
from flare.gp import GaussianProcess
from flare.struc import Structure

from flare.predict import predict_on_structure, predict_on_structure_par
import pytest

def fake_predict(x,d):

    return   np.random.uniform(-1, 1), np.random.uniform(-1, 1)


_fake_gp = GaussianProcess(kernel_name='2_sc', cutoffs=[5], hyps=[1, 1, 1])

_fake_structure = Structure(cell=np.eye(3), species=[1, 1, 1],
                            positions=np.random.uniform(0, 1, size=(3, 3)))

_fake_gp.predict = fake_predict
    #lambda _, __: (
    #np.random.uniform(-1, 1), np.random.uniform(-1, 1))

print(_fake_gp.predict(1, 2))


@pytest.mark.parametrize('n_cpu', [1, 2])
def test_predict_on_structure_par(n_cpu):

    # Predict only on the first atom, and make rest NAN
    selective_atoms=[0]

    skipped_atom_value = np.nan

    forces, stds = predict_on_structure_par(_fake_structure,
                                            _fake_gp,
                                            n_cpus=n_cpu,
                                            write_to_structure=False,
                                            selective_atoms=selective_atoms,
                                            skipped_atom_value=skipped_atom_value)


    for x in forces[0][:]:
        assert isinstance(x,float)
    for x in forces[1:]:
        assert np.isnan(x).all()


    # Predict only on the second and third, and make rest 0

    selective_atoms = [1,2]
    skipped_atom_value =0

    forces, stds = predict_on_structure_par(_fake_structure,
                                            _fake_gp,
                                            write_to_structure=False,
                                            n_cpus=n_cpu,
                                            selective_atoms=selective_atoms,
                                            skipped_atom_value=skipped_atom_value)

    for x in forces[1]:
        assert isinstance(x, float)
    for x in forces[2]:
        assert isinstance(x, float)

    assert np.equal(forces[0], 0).all()



    # Make selective atoms be all and ensure results are normal

    selective_atoms = [0, 1, 2]

    forces, stds = predict_on_structure_par(_fake_structure,
                                        _fake_gp,
                                        write_to_structure=True,
                                        n_cpus=n_cpu,
                                        selective_atoms=selective_atoms,
                                        skipped_atom_value=skipped_atom_value)


    for x in forces.flatten():
        assert isinstance(x, float)
    for x in stds.flatten():
        assert isinstance(x, float)

    assert np.array_equal(_fake_structure.forces, forces)
    assert np.array_equal(_fake_structure.stds, stds)


    # Get new examples to also test the results not being written

    selective_atoms = [0,1]

    forces, stds = predict_on_structure_par(_fake_structure,
                                        _fake_gp,
                                        write_to_structure=True,
                                        n_cpus=n_cpu,
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








