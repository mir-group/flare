#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" OTF Regression test suite based on py.test

qe_input_1: H2 dimer

qe_input_2: 2x1x1 Al Supercell

Steven Torrisi, Simon Batzner
"""

import pytest
import os
import sys
import numpy as np
sys.path.append("../src")
from otf import parse_qe_input, parse_qe_forces, OTF
from gp import GaussianProcess

def fake_espresso(noa):
    """ Returns a list of random forces.
    """
    return

class Fake_GP(GaussianProcess):
    """
    Fake GP that returns random forces and variances when asked to predict.
    """

    def __init__(self, kernel):
        super(GaussianProcess, self).__init__()

        pass

    def train(self):
        """
        Invalidates the train method of GaussianProcess
        """
        pass

    def update_db(self, structure, forces):
        """
        Invalidates the update_db method of GaussianProcess
        """
        pass

    def predict(self, structure, _):
        """
        Substitutes in the predict method of GaussianProcess
        """
        structure.forces = [np.random.randn(3) for n in range(structure.nat)]
        structure.stds = [np.random.randn(3) for n in range(structure.nat)]

        return structure

    def predict_on_structure(self, structure):
        """
        Substitutes in the predict_on_structure method of GaussianProcess
        """

        structure.forces = [np.random.randn(3) for n in structure.positions]
        structure.stds = [np.random.randn(3) for n in structure.positions]

        return structure


def cleanup_otf_run():
    os.system('rm otf_run.out')
    os.system('rm pwscf.out')
    os.system('rm pwscf.wfc')

# ------------------------------------------------------
#          fixtures
# ------------------------------------------------------


@pytest.fixture(scope='module')
def test_params_1():
    params = {'qe_input': './test_files/qe_input_1.in',
              'dt': .01,
              'num_steps': 10,
              'kernel': 'two_body',
              'cutoff': 5.0}
    yield params


@pytest.fixture(scope='module')
def test_otf_engine_1(test_params_1):
    engine = OTF(qe_input=test_params_1['qe_input'],
                 dt=test_params_1['dt'],
                 number_of_steps=test_params_1['num_steps'],
                 kernel=test_params_1['kernel'],
                 cutoff=test_params_1['cutoff'])

    yield engine

    del engine


# ------------------------------------------------------
#                   test  otf helper functions
# ------------------------------------------------------

@pytest.mark.parametrize("qe_input,exp_pos",
                         [
                             ('./test_files/qe_input_1.in',
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
                             ('./test_files/qe_input_1.in',
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
                             ('./test_files/qe_input_1.in',
                              5.0 * np.eye(3))
                         ]
                         )
def test_cell_parsing(qe_input, exp_cell):
    positions, species, cell = parse_qe_input(qe_input)
    assert np.all(exp_cell == cell)


@pytest.mark.parametrize('qe_output,exp_forces',
                         [
                             ('./test_files/qe_output_1.out',
                              [np.array([0.07413986, 0., 0.]),
                               np.array([-0.07413986, 0., 0.])])
                         ]
                         )
def test_force_parsing(qe_output, exp_forces):
    forces = parse_qe_forces(qe_output)
    assert len(forces) == len(exp_forces)

    for i, force in enumerate(forces):
        assert np.all(force == exp_forces[i])


def test_espresso_calling_1(test_otf_engine_1):
    assert os.environ.get('PWSCF_COMMAND',
                          False), 'PWSCF_COMMAND not found ' \
                                  'in environment'

    forces = test_otf_engine_1.run_espresso()
    assert isinstance(forces, list)
    assert len(forces) == len(test_otf_engine_1.structure.forces)

    if test_otf_engine_1.qe_input == './test_files/qe_input_1.in':
        test1forces = [np.array([0.07413986, 0.0, 0.0]),
                       np.array([-0.07413986,
                                 0.0, 0.0])]
        for i, force in enumerate(forces):
            assert np.equal(force, test1forces[i]).all()

    cleanup_otf_run()

# ------------------------------------------------------
#                   test  otf methods
# ------------------------------------------------------


#TODO see if there is a better way to to set up the different input runs

def test_update_1(test_otf_engine_1):

    test_otf_engine_1.structure.prev_positions=[[2.5,2.5,2.5],
                                                [4.5,2.5,2.5]]

    test_otf_engine_1.structure.forces=[np.array([.07413986, 0.0, 0.0]),
                                      np.array([-0.07413986, 0.0, 0.0])]

    test_otf_engine_1.update_positions()

    target_positions=[  np.array([2.53714741,2.5,2.5]),
                        np.array([4.46285259,2.5,2.5])
                                 ]

    for i,pos in enumerate(test_otf_engine_1.structure.positions):
        assert np.isclose(pos,target_positions[i],rtol=1e-6).all()



