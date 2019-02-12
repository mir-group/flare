import pytest
import numpy as np
import sys
from test_GaussianProcess import get_random_structure
sys.path.append('../otf_engine')
from struc import Structure


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


def test_2_body_bond_order():
    lattice = np.eye(3)
    species = ['B', 'A']
    positions = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    test_structure = Structure(lattice, species, positions)

    # test species_to_bond
    assert (test_structure.bond_list == [['B', 'B'], ['B', 'A'], ['A', 'A']])
