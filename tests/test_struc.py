import pytest
import numpy as np
import sys
from tests.test_gp import get_random_structure
from flare.struc import Structure
from json import loads

try:
    import pymatgen.core.structure as pmgstruc
    _test_pmg = True
except ImportError:
    _test_pmg = False


def test_random_structure_setup():
    struct, forces = get_random_structure(cell=np.eye(3),
                                          unique_species=[1, 2],
                                          noa=2)

    assert np.equal(struct.cell, np.eye(3)).all()
    assert len(struct.positions) == 2


def test_prev_positions_arg():
    np.random.seed(0)
    positions = []
    prev_positions = []
    species = [1] * 5
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
    assert (np.isclose(positions[ind], rel_vals[ind, 0] * test_struc.vec1 +
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

    assert (np.isclose(np.round(wrap_rel) - wrap_rel,
                       np.zeros(positions.shape)).all())


@pytest.fixture
def varied_test_struc():
    struc = Structure(np.eye(3), species=[1, 2, 2, 3, 3, 4, 4, 4,
                                          4, 3],
                      positions=np.array([np.random.uniform(-1, 1, 3) for i
                                          in range(10)]))
    struc.forces = np.array([np.random.uniform(-1, 1, 3) for _ in
                             range(10)])
    return struc


def test_indices_of_specie(varied_test_struc):
    assert varied_test_struc.indices_of_specie(1) == [0]
    assert varied_test_struc.indices_of_specie(2) == [1, 2]
    assert varied_test_struc.indices_of_specie(3) == [3, 4, 9]
    assert varied_test_struc.indices_of_specie(4) == [5, 6, 7, 8]


def test_to_from_methods(varied_test_struc):
    test_dict = varied_test_struc.as_dict()

    assert isinstance(test_dict, dict)
    assert (test_dict['forces'] == varied_test_struc.forces).all()

    new_struc_1 = Structure.from_dict(test_dict)
    new_struc_2 = Structure.from_dict(loads(varied_test_struc.as_str()))

    for new_struc in [new_struc_1, new_struc_2]:
        assert np.equal(varied_test_struc.positions, new_struc.positions).all()
        assert np.equal(varied_test_struc.cell, new_struc.cell).all()
        assert np.equal(varied_test_struc.forces, new_struc.forces).all()


def test_rep_methods(varied_test_struc):
    assert len(varied_test_struc) == 10
    assert isinstance(str(varied_test_struc), str)

    thestr = str(varied_test_struc)
    assert 'Structure with 10 atoms of types {' in thestr

def test_struc_from_ase():
    from ase import Atoms
    uc = Atoms(['Pd' for i in range(10)]+['Ag' for i in range(10)],
               positions=np.random.rand(20, 3),
               cell=np.random.rand(3, 3))
    new_struc = Structure.from_ase_atoms(uc)
    assert np.all(new_struc.species_labels == uc.get_chemical_symbols())
    assert np.all(new_struc.positions == uc.get_positions())
    assert np.all(new_struc.cell == uc.get_cell())

def test_struc_to_ase():
    from ase import Atoms
    uc = Structure(species=['Pd' for i in range(10)]+['Ag' for i in range(10)],
                   positions=np.random.rand(20, 3),
                   cell=np.random.rand(3, 3))
    new_atoms = Structure.to_ase_atoms(uc)
    assert np.all(uc.species_labels == new_atoms.get_chemical_symbols())
    assert np.all(uc.positions == new_atoms.get_positions())
    assert np.all(uc.cell == new_atoms.get_cell())

   
@pytest.mark.skipif(not _test_pmg,reason='Pymatgen not present in available '
                                        'packages.')
def test_from_pmg_structure():

    pmg_struc = pmgstruc.Structure(lattice= np.eye(3),
                                   species=['H'],
                                   coords=[[.25, .5, 0]],
                                   site_properties={
                                       'force': [np.array((1., 1.,  1.))],
                                        'std':[np.array((1., 1., 1.))]},
                                   coords_are_cartesian=True)

    new_struc = Structure.from_pmg_structure(pmg_struc)

    assert len(new_struc) == 1

    assert np.equal(new_struc.positions, np.array([.25, .5, 0])).all()
    assert new_struc.coded_species == [1]
    assert new_struc.species_labels[0] == 'H'
    assert np.equal(new_struc.forces, np.array([1., 1., 1.])).all()

@pytest.mark.skipif(not _test_pmg,reason='Pymatgen not present in available '
                                        'packages.')
def test_to_pmg_structure(varied_test_struc):

    new_struc = Structure.to_pmg_structure(varied_test_struc)
    assert len(varied_test_struc) == len(varied_test_struc)
    assert np.equal(new_struc.cart_coords, varied_test_struc.positions).all()
    assert (new_struc.atomic_numbers == varied_test_struc.coded_species).all()
