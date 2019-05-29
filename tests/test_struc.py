import pytest
import numpy as np
import sys
from test_gp import get_random_structure
from flare.struc import Structure


def test_random_structure_setup():
    struct, forces = get_random_structure(cell=np.eye(3),
                                          unique_species=["A", "B", ],
                                          noa=2)

    assert np.equal(struct.cell, np.eye(3)).all()
    assert 'A' in struct.unique_species or 'B' in struct.unique_species
    assert len(struct.positions) == 2


def test_prev_positions_arg():

    np.random.seed(0)
    positions = []
    prev_positions = []
    species = ['A']*5
    cell = np.eye(3)
    for n in range(5):
        positions.append(np.random.uniform(-1, 1, 3))
        prev_positions.append(np.random.uniform(-1, 1, 3))

    test_structure1 = Structure(cell, species, positions)
    test_structure2 = Structure(cell, species, positions,
                                prev_positions=positions)
    test_structure3 = Structure(cell, species, positions,
                                prev_positions=prev_positions)

    assert np.equal(test_structure1.positions, test_structure2.positions).all()
    assert np.equal(test_structure1.prev_positions,
                    test_structure2.prev_positions).all()
    assert np.equal(test_structure2.positions,
                    test_structure2.prev_positions).all()
    assert not np.equal(test_structure3.positions,
                        test_structure3.prev_positions).all()


def test_raw_to_relative():
    """ Test that Cartesian and relative coordinates are equal. """

    cell = np.random.rand(3, 3)
    noa = 10
    positions = np.random.rand(noa, 3)
    species = ['Al'] * len(positions)

    test_struc = Structure(cell, species, positions)
    rel_vals = test_struc.raw_to_relative(test_struc.positions,
                                          test_struc.cell_transpose,
                                          test_struc.cell_dot_inverse)

    ind = np.random.randint(0, noa)
    assert(np.isclose(positions[ind], rel_vals[ind, 0] * test_struc.vec1 +
           rel_vals[ind, 1] * test_struc.vec2 +
           rel_vals[ind, 2] * test_struc.vec3).all())


def test_wrapped_coordinates():
    """ Check that wrapped coordinates are equivalent to Cartesian coordinates
    up to lattice translations. """

    cell = np.random.rand(3, 3)
    positions = np.random.rand(10, 3)
    species = ['Al'] * len(positions)

    test_struc = Structure(cell, species, positions)

    wrap_diff = test_struc.positions - test_struc.wrapped_positions
    wrap_rel = test_struc.raw_to_relative(wrap_diff,
                                          test_struc.cell_transpose,
                                          test_struc.cell_dot_inverse)

    assert(np.isclose(np.round(wrap_rel) - wrap_rel,
           np.zeros(positions.shape)).all())
