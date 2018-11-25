#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" Punchout test suite based on py.test

Steven Torrisi
"""

import pytest

import numpy as np

import sys
sys.path.append('../otf_engine')

from punchout import is_within_d_box, punchout,punchout_structure
from test_GaussianProcess import get_random_structure
from struc import Structure
from env import ChemicalEnvironment
# -----------------------
# Test helper methods
# -----------------------

def test_d_within_box():
    """
    Run with a few examples
    :return:
    """

    r1 = np.array([0, 0, 0])
    r2 = np.array([0, 0, .5])
    r3 = np.array([.5, .5, .5])
    r4 = np.array([.6, .5, .5])
    r5 = np.array([1.0, 2.0, 3.0])

    assert is_within_d_box(r1, r2, d=1)
    assert is_within_d_box(r1, -r2, d=1)
    assert not is_within_d_box(r1, r2, d=.5)
    assert not is_within_d_box(r1, r2, d=.1)

    assert is_within_d_box(r1, r3, 1)
    assert is_within_d_box(r2, r3, 1)

    assert not is_within_d_box(r1, r4, 1)
    assert is_within_d_box(r1, r5, 6.0)
    assert not is_within_d_box(r1, r5, 3.0)

    assert is_within_d_box(r3, r4, .2)
    assert is_within_d_box(r3, r4, .5)
    assert not is_within_d_box(r3, r4, .1)

    # Run for a hundred random pairs
    for n in range(100):
        rand1 = np.random.randn(3)
        rand2 = np.random.randn(3)

        radius = np.linalg.norm(rand1 - rand2)
        assert is_within_d_box(rand1, rand2, radius * 2)
        assert not is_within_d_box(rand1, rand2, radius)


# -----------------------
# Test punchout
# -----------------------

@pytest.fixture(scope='module')
def test_punchout_random_struc():
    cell = 10 * np.eye(3)
    species = ['A', 'B', 'C', 'D', 'E'] * 3
    mass_dict = {spec: np.random.randn(1) for spec in species}
    noa = 15
    struct, _ = get_random_structure(cell, species, cutoff=5, noa=noa)
    struct.mass_dict = mass_dict

    print(struct)
    target_atom = np.random.randint(0, noa)
    d = np.random.uniform(1, 8)
    punchstruc = punchout_structure(structure=struct, atom=target_atom, d=d)

    assert isinstance(punchstruc, Structure)

    yield (punchstruc, d)

    del punchstruc


def test_punchout_boxed(test_punchout_random_struc):
    punchstruc = test_punchout_random_struc[0]
    d = test_punchout_random_struc[1]

    for pos1 in punchstruc.positions:
        for pos2 in punchstruc.positions:
            assert is_within_d_box(pos1, pos2, d)


def test_punchout_edges_1():
    """
    Test a few hand-defined edge cases to ensure punched out atoms near the
    edges correctly have their periodic neighbors
    :return:
    """
    species = ['A']
    cell = np.eye(3)
    positions = [.5 * np.ones(3)]
    struc = Structure(cell, species, positions, cutoff=1,
                        mass_dict={'A': 1})

    p1 = punchout_structure(structure=struc, atom=0, d=.25, center=False)
    assert len(p1.positions) == 1
    assert np.equal(p1.positions, .5 * np.ones(3)).all()

    p2 = punchout_structure(structure=struc, atom=0, d=2., center=False)
    assert len(p2.positions) == 27

    p3 = punchout_structure(structure=struc, atom=0, d=1.0, center=True)
    assert np.equal(p3.positions,np.zeros(3)).all()

def test_punchout_edges_2():
    species = ['A'] * 2
    cell = np.eye(3)
    positions = [np.zeros(3), .25 * np.ones(3)]
    struc = Structure(cell, species, positions, cutoff=1,
                      mass_dict={'A': 1})
    p1 = punchout_structure(structure=struc, atom=0, d=.01,center=False)
    print(p1)
    assert len(p1.positions) == 1
    assert np.equal(p1.positions, np.zeros(3)).all()

    p2 = punchout_structure(structure=struc, atom=0, d=.5,center=False)
    assert np.equal(p2.positions, positions).all()

    p3 = punchout_structure(structure=struc, atom=1, d=1.,center=False)
    assert np.equal(p3.positions, positions).all()

def test_punchout_edges_3():
    species = ['A'] * 2
    cell = np.eye(3)
    positions = [np.zeros(3), .9 * np.ones(3)]
    struc = Structure(cell, species, positions, cutoff=1,
                      mass_dict={'A': 1})
    p1 = punchout_structure(structure=struc, atom=0, d=.01,center=False)
    assert len(p1.positions) == 1
    assert np.equal(p1.positions, np.zeros(3)).all()

    p2 = punchout_structure(structure=struc, atom=0, d=.5,center=False)
    assert np.isclose(p2.positions, [np.zeros(3), -.1 * np.ones(3)]).all()

    p3 = punchout_structure(structure=struc, atom=1, d=.5,center=True)
    assert np.isclose(p3.positions, [.1 * np.ones(3),np.zeros(3) ]).all()

    p4 = punchout_structure(structure=struc, atom=1, d=1.,center=False)
    assert np.isclose(p4.positions, [np.ones(3), .9*np.ones(3)]).all()


def test_punchout_too_close1():

    cell = 10*np.eye(3)
    positions = [ [.01,0,0], [1,0,0], [1.99,0,0] ]
    species = ['A'] * len(positions)

    struc = Structure(cell, species, positions, cutoff=1,
                      mass_dict={'A':1})

    badstruc  = punchout_structure(struc,atom=1,d=2.0,center=False)
    for i in [0,2]:
        bond_array, _, _, _ = \
                    ChemicalEnvironment.get_atoms_within_cutoff(badstruc,
                                                                atom=i)
        distances =[array[0] for array in bond_array]
        assert np.min(distances) < .25


    newstruc = punchout(struc, atom = 1, settings={'d':2.0}, check_too_close=.25,
                        center=False)
    for i in range(newstruc.nat):
        bond_array, _, _, _ = \
                    ChemicalEnvironment.get_atoms_within_cutoff(newstruc, atom=i)
        distances =[array[0] for array in bond_array]
        assert np.min(distances) > .25


def test_punchout_stoichiometry():
    cell = 10 * np.eye(3)
    positions = [[.01, 0, 0], [1, 0, 0], [1.99, 0, 0], [2.4,0,0]]
    species = ['B','A','A','B']

    struc = Structure(cell, species, positions, cutoff=1,
                      mass_dict={'A': 1,'B':1})

    badstruc = punchout_structure(struc, atom=1, d=2.0, center=False)

    assert badstruc.get_species_count()["A"] == 2
    assert badstruc.get_species_count()['B'] == 1

    newstruc = punchout(struc, atom=1, settings={'d':2.0}, check_stoichiometry=['A','B'],
                        center=False)

    assert newstruc.get_species_count()["A"] == 2
    assert newstruc.get_species_count()['B'] == 2






