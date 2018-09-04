#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" OTF Regression test suite based on py.test

Steven Torrisi, Simon Batzner
"""

import pytest

import numpy as np

from otf import parse_qe_input, parse_qe_forces, OTF
from test_GaussianProcess import get_random_structure
from struc import Structure
import os


def fake_espresso(noa):
    """ Returns a list of random forces.
    """
    return


# ------------------------------------------------------
#          fixtures
# ------------------------------------------------------


@pytest.fixture(scope='module')
def test_params():
    params = {'qe_input': 'tests/test_files/qe_input_1.in',
              'dt': .01,
              'num_steps': 10,
              'kernel': 'two_body',
              'cutoff': 5.0}
    yield params


@pytest.fixture(scope='module')
def test_otf_engine(test_params):
    engine = OTF(qe_input=test_params['qe_input'],
                 dt=test_params['dt'],
                 number_of_steps=test_params['num_steps'],
                 kernel=test_params['kernel'],
                 cutoff=test_params['cutoff'])

    yield engine

    del engine


# ------------------------------------------------------
#                   test  otf helper functions
# ------------------------------------------------------

@pytest.mark.parametrize("qe_input,exp_pos",
                         [
                             ('tests/test_files/qe_input_1.in',
                              [np.array([2.51857, 2.5, 2.5]),
                               np.array([4.48143, 2.5, 2.5])])
                         ]
                         )
def test_position_parsing(qe_input, exp_pos):
    positions, species, cell = parse_qe_input(qe_input)
    assert len(positions) == len(exp_pos)

    for i, pos in enumerate(positions):
        assert np.all(pos == exp_pos[i])


@pytest.mark.parametrize("qe_input,exp_spec",
                         [
                             ('tests/test_files/qe_input_1.in',
                              ['H', 'H'])
                         ]
                         )
def test_species_parsing(qe_input, exp_spec):
    positions, species, cell = parse_qe_input(qe_input)
    assert len(species) == len(exp_spec)
    for i, spec in enumerate(species):
        assert spec == exp_spec[i]


@pytest.mark.parametrize("qe_input,exp_cell",
                         [
                             ('tests/test_files/qe_input_1.in',
                              5.0 * np.eye(3))
                         ]
                         )
def test_cell_parsing(qe_input, exp_cell):
    positions, species, cell = parse_qe_input(qe_input)
    assert np.all(exp_cell == cell)


@pytest.mark.parametrize('qe_output,exp_forces',
                         [
                             ('tests/test_files/qe_output_1.out',
                              [np.array([0.07413986, 0., 0.]),
                               np.array([-0.07413986, 0., 0.])])
                         ]
                         )
def test_force_parsing(qe_output, exp_forces):
    forces = parse_qe_forces(qe_output)
    assert len(forces) == len(exp_forces)

    for i, force in enumerate(forces):
        assert np.all(force == exp_forces[i])


def test_espresso_calling(test_otf_engine):
    assert os.environ.get('PWSCF_COMMAND',
                          False), 'PWSCF_COMMAND not found ' \
                                  'in environment'

    forces = test_otf_engine.run_espresso()
    assert isinstance(forces, list)
    assert len(forces) == len(test_otf_engine.structure.forces)

    if test_otf_engine.qe_input == 'tests/test_files/qe_input_1.in':
        test1forces = [np.array([0.07413986, 0.0, 0.0]),
                       np.array([-0.07413986,
                                 0.0, 0.0])]
        for i, force in enumerate(forces):
            assert np.equal(force, test1forces[i]).all()


# ------------------------------------------------------
#                   test  Structure functions
# ------------------------------------------------------


def test_random_structure_setup():
    struct, forces = get_random_structure(cell=np.eye(3),
                                          unique_species=["A", "B", ],
                                          cutoff=np.random.uniform(1, 10.),
                                          noa=2)

    assert np.equal(struct.lattice, np.eye(3)).all()
    assert 'A' in struct.elements or 'B' in struct.elements
    assert len(struct.positions) == 2


def test_2_body_bond_order():
    """
    Written by Simon B
    :return:
    """
    lattice = np.eye(3)
    species = ['B', 'A']
    positions = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    test_structure = Structure(lattice, species, positions, cutoff)

    # test species_to_bond
    assert (test_structure.bond_list == [['B', 'B'], ['B', 'A'], ['A', 'A']])

# TODO IO-based unit tests for pasrsing the output files of runs (even though
# some may be random
