import pytest
import numpy as np
import os

from tests.test_gp import get_random_structure
from flare.struc import Structure
from json import loads, dumps
from flare.utils.element_coder import Z_to_element, NumpyEncoder

try:
    import pymatgen.core.structure as pmgstruc

    _test_pmg = True
except ImportError:
    _test_pmg = False

from .test_gp import dumpcompare


def test_random_structure_setup():
    struct, forces = get_random_structure(cell=np.eye(3), unique_species=[1, 2], noa=2)

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
    test_structure2 = Structure(cell, species, positions, prev_positions=positions)
    test_structure3 = Structure(cell, species, positions, prev_positions=prev_positions)

    assert np.equal(test_structure1.positions, test_structure2.positions).all()
    assert np.equal(
        test_structure1.prev_positions, test_structure2.prev_positions
    ).all()
    assert np.equal(test_structure2.positions, test_structure2.prev_positions).all()
    assert not np.equal(test_structure3.positions, test_structure3.prev_positions).all()


def test_raw_to_relative():
    """ Test that Cartesian and relative coordinates are equal. """

    cell = np.random.rand(3, 3)
    noa = 10
    positions = np.random.rand(noa, 3)
    species = ["Al"] * len(positions)

    test_struc = Structure(cell, species, positions)
    rel_vals = test_struc.raw_to_relative(
        test_struc.positions, test_struc.cell_transpose, test_struc.cell_dot_inverse
    )

    ind = np.random.randint(0, noa)
    assert np.isclose(
        positions[ind],
        rel_vals[ind, 0] * test_struc.vec1
        + rel_vals[ind, 1] * test_struc.vec2
        + rel_vals[ind, 2] * test_struc.vec3,
    ).all()


def test_wrapped_coordinates():
    """Check that wrapped coordinates are equivalent to Cartesian coordinates
    up to lattice translations."""

    cell = np.random.rand(3, 3)
    positions = np.random.rand(10, 3)
    species = ["Al"] * len(positions)

    test_struc = Structure(cell, species, positions)

    wrap_diff = test_struc.positions - test_struc.wrapped_positions
    wrap_rel = test_struc.raw_to_relative(
        wrap_diff, test_struc.cell_transpose, test_struc.cell_dot_inverse
    )

    assert np.isclose(np.round(wrap_rel) - wrap_rel, np.zeros(positions.shape)).all()


@pytest.fixture
def varied_test_struc():
    struc = Structure(
        np.eye(3),
        species=[1, 2, 2, 3, 3, 4, 4, 4, 4, 3],
        positions=np.array([np.random.uniform(-1, 1, 3) for i in range(10)]),
    )
    struc.forces = np.array([np.random.uniform(-1, 1, 3) for _ in range(10)])
    return struc


def test_indices_of_specie(varied_test_struc):
    assert varied_test_struc.indices_of_specie(1) == [0]
    assert varied_test_struc.indices_of_specie(2) == [1, 2]
    assert varied_test_struc.indices_of_specie(3) == [3, 4, 9]
    assert varied_test_struc.indices_of_specie(4) == [5, 6, 7, 8]


def test_to_from_methods(varied_test_struc):
    test_dict = varied_test_struc.as_dict()

    assert isinstance(test_dict, dict)
    assert (test_dict["forces"] == varied_test_struc.forces).all()

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
    assert "Structure with 10 atoms of types {" in thestr


def test_struc_from_ase():
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    results = {
        "forces": np.random.randn(20, 3),
        "energy": np.random.rand(),
        "stress": np.random.randn(6),
    }

    uc = Atoms(
        ["Pd" for i in range(10)] + ["Ag" for i in range(10)],
        positions=np.random.rand(20, 3),
        cell=np.random.rand(3, 3),
    )

    calculator = SinglePointCalculator(uc, **results)
    uc.set_calculator(calculator)

    new_struc = Structure.from_ase_atoms(uc)

    assert np.all(new_struc.species_labels == uc.get_chemical_symbols())
    assert np.all(new_struc.positions == uc.get_positions())
    assert np.all(new_struc.cell == uc.get_cell())
    assert np.all(new_struc.forces == results["forces"])
    assert np.all(new_struc.energy == results["energy"])
    assert np.all(new_struc.stress == results["stress"])


def test_struc_to_ase():
    uc = Structure(
        species=["Pd" for i in range(10)] + ["Ag" for i in range(10)],
        positions=np.random.rand(20, 3),
        cell=np.random.rand(3, 3),
    )

    uc.forces = np.random.randn(20, 3)
    uc.energy = np.random.rand()
    uc.stress = np.random.randn(6)

    new_atoms = Structure.to_ase_atoms(uc)
    assert np.all(uc.species_labels == new_atoms.get_chemical_symbols())
    assert np.all(uc.positions == new_atoms.get_positions())
    assert np.all(uc.cell == new_atoms.get_cell())
    assert np.all(new_atoms.get_forces() == uc.forces)
    assert np.all(new_atoms.get_potential_energy() == uc.energy)
    assert np.all(new_atoms.get_stress() == uc.stress)


@pytest.mark.skipif(not _test_pmg, reason="Pymatgen not present in available packages.")
def test_from_pmg_structure():

    pmg_struc = pmgstruc.Structure(
        lattice=np.eye(3),
        species=["H"],
        coords=[[0.25, 0.5, 0]],
        site_properties={
            "force": [np.array((1.0, 1.0, 1.0))],
            "std": [np.array((1.0, 1.0, 1.0))],
        },
        coords_are_cartesian=True,
    )

    new_struc = Structure.from_pmg_structure(pmg_struc)

    assert len(new_struc) == 1

    assert np.equal(new_struc.positions, np.array([0.25, 0.5, 0])).all()
    assert new_struc.coded_species == [1]
    assert new_struc.species_labels[0] == "H"
    assert np.equal(new_struc.forces, np.array([1.0, 1.0, 1.0])).all()

    pmg_struc = pmgstruc.Structure(
        lattice=np.diag([1.2, 0.8, 1.5]),
        species=["H"],
        coords=[[0.25, 0.5, 0]],
        site_properties={
            "force": [np.array((1.0, 1.0, 1.0))],
            "std": [np.array((1.0, 1.0, 1.0))],
        },
        coords_are_cartesian=True,
    )

    new_struc = Structure.from_pmg_structure(pmg_struc)

    assert len(new_struc) == 1

    assert np.equal(new_struc.positions, np.array([0.25, 0.5, 0])).all()
    assert new_struc.coded_species == [1]
    assert new_struc.species_labels[0] == "H"
    assert np.equal(new_struc.forces, np.array([1.0, 1.0, 1.0])).all()


@pytest.mark.skipif(not _test_pmg, reason="Pymatgen not present in available packages.")
def test_to_pmg_structure(varied_test_struc):

    new_struc = Structure.to_pmg_structure(varied_test_struc)
    assert len(varied_test_struc) == len(varied_test_struc)
    assert np.equal(new_struc.cart_coords, varied_test_struc.positions).all()
    assert (new_struc.atomic_numbers == varied_test_struc.coded_species).all()


def test_to_xyz(varied_test_struc):

    simple_str = varied_test_struc.to_xyz(
        extended_xyz=False, print_stds=False, print_forces=False, print_max_stds=False
    )

    simple_str_by_line = simple_str.split("\n")

    assert len(simple_str_by_line) - 2 == len(varied_test_struc)

    for i, atom_line in enumerate(simple_str_by_line[2:-1]):
        split_line = atom_line.split()
        assert split_line[0] == Z_to_element(int(varied_test_struc.species_labels[i]))
        for j in range(3):
            assert float(split_line[1 + j]) == varied_test_struc.positions[i][j]

    complex_str = varied_test_struc.to_xyz(True, True, True, True)
    complex_str_by_line = complex_str.split("\n")

    assert len(complex_str_by_line) - 2 == len(varied_test_struc)

    for i, atom_line in enumerate(complex_str_by_line[2:-1]):
        split_line = atom_line.split()
        assert split_line[0] == Z_to_element(int(varied_test_struc.species_labels[i]))
        for j in range(1, 4):
            assert float(split_line[j]) == varied_test_struc.positions[i][j - 1]
        for j in range(4, 7):
            assert float(split_line[j]) == varied_test_struc.stds[i][j - 4]
        for j in range(7, 10):
            assert float(split_line[j]) == varied_test_struc.forces[i][j - 7]
        assert float(split_line[10]) == np.max(varied_test_struc.stds[i])


def test_file_load():
    struct1, forces = get_random_structure(cell=np.eye(3), unique_species=[1, 2], noa=2)
    struct2, forces = get_random_structure(cell=np.eye(3), unique_species=[1, 2], noa=2)

    with open("test_write.json", "w") as f:
        f.write(dumps(struct1.as_dict(), cls=NumpyEncoder))

    with pytest.raises(NotImplementedError):
        Structure.from_file(file_name="test_write.json", format="xyz")

    struct1a = Structure.from_file("test_write.json")
    assert dumpcompare(struct1.as_dict(), struct1a.as_dict())
    os.remove("test_write.json")

    with open("test_write_set.json", "w") as f:
        f.write(dumps(struct1.as_dict(), cls=NumpyEncoder) + "\n")
        f.write(dumps(struct2.as_dict(), cls=NumpyEncoder) + "\n")

    structs = Structure.from_file("test_write_set.json")
    for struc in structs:
        assert isinstance(struc, Structure)
    assert len(structs) == 2
    assert dumpcompare(structs[0].as_dict(), struct1.as_dict())
    assert dumpcompare(structs[1].as_dict(), struct2.as_dict())

    os.remove("test_write_set.json")

    with pytest.raises(FileNotFoundError):
        Structure.from_file("fhqwhgads")

    vasp_struct = Structure.from_file("./test_files/test_POSCAR")
    assert isinstance(vasp_struct, Structure)
    assert len(vasp_struct) == 6


def test_is_valid():
    """
    Try a trivial case of 1-len structure and then one above and below
    tolerance
    :return:
    """
    test_struc = Structure(
        cell=np.eye(3), species=["H"], positions=np.array([[0, 0, 0]])
    )

    assert test_struc.is_valid()

    test_struc = Structure(
        cell=np.eye(3), species=["H", "H"], positions=[[0, 0, 0], [0.3, 0, 0]]
    )
    assert not test_struc.is_valid()
    assert test_struc.is_valid(tolerance=0.25)
