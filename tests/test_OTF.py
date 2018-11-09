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

sys.path.append('../otf_engine')
from otf import OTF
from gp import GaussianProcess
from struc import Structure
from test_qe_util import cleanup_espresso_run


def fake_espresso(qe_input: str, structure: Structure):
    """
    Returns a list of random forces. Takes the same argument as real ESPRESSO
    in order to allow for substitution.
    """
    noa = len(structure.positions)
    return [np.random.normal(loc=0, scale=1, size=3) for _ in range(noa)]


def fake_predict_on_structure( structure):
    """
        Substitutes in the predict_on_structure method of GaussianProcess
        """

    structure.forces = [np.random.randn(3) for _ in structure.positions]
    structure.stds = [np.random.randn(3) for _ in structure.positions]

    return structure


class Fake_GP(GaussianProcess):
    """
    Fake GP that returns random forces and variances when asked to predict.
    """

    def __init__(self, kernel):
        super(GaussianProcess, self).__init__()

        self.sigma_f = 0
        self.length_scale = 0
        self.sigma_n = 0

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

    def predict(self, chemenv, i):
        """
        Substitutes in the predict method of GaussianProcess
        """
        # structure.forces = [np.random.randn(3) for n in range(structure.nat)]
        # structure.stds = [np.random.randn(3) for n in range(structure.nat)]

        return chemenv


def cleanup_otf_run():
    cleanup_espresso_run()
    os.system('rm otf_run.out')


# ------------------------------------------------------
#          fixtures
# ------------------------------------------------------


@pytest.fixture(scope='module')
def test_params_1():
    params = {'qe_input': './test_files/qe_input_1.in',
              'dt': .01,
              'num_steps': 10,
              'kernel': 'n_body_sc',
              'bodies': 2,
              'cutoff': 5.0}
    yield params


@pytest.fixture(scope='module')
def test_otf_engine_1(test_params_1):
    engine = OTF(qe_input=test_params_1['qe_input'],
                 dt=test_params_1['dt'],
                 number_of_steps=test_params_1['num_steps'],
                 kernel=test_params_1['kernel'],
                 bodies=test_params_1['bodies'],
                 cutoff=test_params_1['cutoff'])

    yield engine

    del engine


# ------------------------------------------------------
#                   test  otf methods
# ------------------------------------------------------

# TODO see if there is a better way to to set up the different input runs
def test_update_1(test_otf_engine_1):
    test_otf_engine_1.structure.prev_positions = [[2.5, 2.5, 2.5],
                                                  [4.5, 2.5, 2.5]]

    test_otf_engine_1.structure.forces = [np.array([.07413986, 0.0, 0.0]),
                                          np.array([-0.07413986, 0.0, 0.0])]

    test_otf_engine_1.update_positions()

    target_positions = [np.array([2.60867409, 2.5, 2.5]),
                        np.array([4.39132591, 2.5, 2.5])
                        ]

    for i, pos in enumerate(test_otf_engine_1.structure.positions):
        assert np.isclose(pos, target_positions[i], rtol=1e-6).all()


# ------------------------------------------------------
#                   test  otf runs
# ------------------------------------------------------

def test_otf_1_1():
    """
    Test that a minimal OTF run can complete after two steps
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in ./pwscf.in')

    otf = OTF(qe_input='./pwscf.in', dt=.0001, number_of_steps=2,
              bodies=2, kernel='n_body_sc',
              cutoff=4, std_tolerance_factor=-.1)
    otf.run()
    cleanup_otf_run()


def test_otf_1_2():
    """
    Test that an otf run can survive going for more steps
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in ./pwscf.in')

    otf = OTF(qe_input='./pwscf.in', dt=.0001, number_of_steps=20,
              bodies=2, kernel='n_body_sc',
              cutoff=5, std_tolerance_factor=-.1)
    otf.run()
    #cleanup_otf_run()


def test_otf_1_3_punchout():
    """
    Test that an otf run will succeed in punchout mode
    with trivial conditions (punch out a cell with one other atom)
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in ./pwscf.in')

    otf = OTF(qe_input='./pwscf.in', dt=.0001, number_of_steps=5,
              bodies=2, punchout_settings={'d': 5}, cutoff=3,
              kernel='n_body_sc', std_tolerance_factor=-.1)
    otf.run()
    cleanup_otf_run()


def test_adapt_from_pwscf_1():
    """
    Load in a pwscf file and test to make sure that hyper parameter
    training is reproduced from the same input; them that a run can be
    performed with no DFT calls
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in ./pwscf.in')

    otf = OTF(qe_input='./pwscf.in', dt=.0001,
              number_of_steps=20,
              bodies=2, cutoff=4,
              kernel='n_body_sc', std_tolerance_factor=0)

    otf.augment_db_from_pwscf(
            pwscf_in_file='./test_files/qe_input_1.in',
            pwscf_out_file='./test_files/qe_output_1.out',
            train = False)

    otf.augment_db_from_pwscf(
        pwscf_in_file='./test_files/qe_input_4.in',
        pwscf_out_file='./test_files/qe_output_4.out',
        train=True, write_hyps=True)

    # Check to see if hyperparameters are as they should be
    # assert np.isclose(otf.gp.hyps, [1.1041008754, 0.9002923226, 1e-05]).all()
    otf.run()
    assert otf.run_stats['dft_calls'] == 0
    cleanup_otf_run()


def test_adapt_from_otf_1():
    """
    Test that the adaptation procedure can work with a simple OTF run
    and reproduce the correct hyperparameters
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in ./pwscf.in')

    otf = OTF(qe_input='./pwscf.in', dt=.0001,
              number_of_steps=2,
              bodies=2, cutoff=4,
              kernel='n_body_sc', std_tolerance_factor=0)

    otf.augment_db_from_otf_run(
        otf_run_output='./test_files/otf_output_1.out',
        train=True,use_prev_hyps=True)

    otf.run()
    assert otf.run_stats['dft_calls'] == 0
    cleanup_otf_run()






