import pytest
import pickle
import os
import json
import numpy as np

from typing import List
from pytest import raises
from scipy.optimize import OptimizeResult

import flare
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
import flare.kernels.sc as en
import flare.kernels.mc_simple as mc_simple
from flare.otf_parser import OtfAnalysis

from .fake_gp import generate_hm, get_tstp, get_random_structure
from copy import deepcopy

multihyps_list = [True, False]


@pytest.fixture(scope='class')
def all_gps() -> GaussianProcess:
    """Returns a GP instance with a two-body numba-based kernel"""

    gp_dict = {True: None, False: None}
    for multihyps in multihyps_list:
        cutoffs = np.ones(2)*0.8
        hyps, hm, _ = generate_hm(1, 1, multihyps=multihyps)
        hl = hm['hyps_label']
        if (multihyps is False):
            hm = None

        # test update_db
        gpname = '2+3+mb_mc'
        hyps = np.hstack([hyps, [1, 1]])
        hl = np.hstack([hl[:-1], ['sigm', 'lsm'], hl[-1]])
        cutoffs = np.ones(3)*0.8

        gp_dict[multihyps] = \
            GaussianProcess(kernel_name=gpname,
                            hyps=hyps,
                            hyp_labels=hl,
                            cutoffs=cutoffs,
                            multihyps=multihyps, hyps_mask=hm,
                            parallel=False, n_cpus=1)

        test_structure, forces = \
            get_random_structure(np.eye(3), [1, 2], 3)
        energy = 3.14

        gp_dict[multihyps].update_db(test_structure, forces, energy=energy)

    yield gp_dict
    del gp_dict


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


@pytest.fixture(scope='module')
def params():
    parameters = {'unique_species': [2, 1],
                  'cutoff': 0.8,
                  'noa': 3,
                  'cell': np.eye(3),
                  'db_pts': 30}
    yield parameters
    del parameters


@pytest.fixture(scope='module')
def validation_env() -> AtomicEnvironment:
    test_pt = get_tstp()
    yield test_pt
    del test_pt


# ------------------------------------------------------
#                   test GP methods
# ------------------------------------------------------

class TestDataUpdating():

    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_update_db(self, all_gps, multihyps, params):

        test_gp = all_gps[multihyps]
        oldsize = len(test_gp.training_data)

        # add structure and forces to db
        test_structure, forces = \
            get_random_structure(params['cell'], params['unique_species'],
                                 params['noa'])
        energy = 3.14
        test_gp.update_db(test_structure, forces, energy=energy)

        assert (len(test_gp.training_data) == params['noa']+oldsize)
        assert (len(test_gp.training_labels_np) == (params['noa']+oldsize)*3)

#


class TestTraining():
    @pytest.mark.parametrize('par, n_cpus', [(False, 1),
                                             (True, 2)])
    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_train(self, all_gps, params, par, n_cpus, multihyps):

        test_gp = all_gps[multihyps]
        test_gp.parallel = par
        test_gp.n_cpus = n_cpus

        test_gp.maxiter = 1

        # train gp
        test_gp.hyps = np.ones(len(test_gp.hyps))
        hyp = list(test_gp.hyps)
        test_gp.train()

        hyp_post = list(test_gp.hyps)

        # check if hyperparams have been updated
        assert (hyp != hyp_post)

    def test_train_failure(self, all_gps, params, mocker):
        """
        Tests the case when 'L-BFGS-B' fails due to a linear algebra error and
        training falls back to BFGS
        """
        # Sets up mocker for scipy minimize. Note that we are mocking
        # 'flare.gp.minimize' because of how the imports are done in gp
        x_result = np.random.rand()
        fun_result = np.random.rand()
        jac_result = np.random.rand()
        train_result = OptimizeResult(x=x_result, fun=fun_result,
                                      jac=jac_result)

        side_effects = [np.linalg.LinAlgError(), train_result]
        mocker.patch('flare.gp.minimize', side_effect=side_effects)
        two_body_gp = all_gps[True]
        two_body_gp.set_L_alpha = mocker.Mock()

        # Executes training
        two_body_gp.algo = 'L-BFGS-B'
        two_body_gp.train()

        # Assert that everything happened as expected
        assert(flare.gp.minimize.call_count == 2)

        calls = flare.gp.minimize.call_args_list
        args, kwargs = calls[0]
        assert(kwargs['method'] == 'L-BFGS-B')

        args, kwargs = calls[1]
        assert(kwargs['method'] == 'BFGS')

        two_body_gp.set_L_alpha.assert_called_once()
        assert(two_body_gp.hyps == x_result)
        assert(two_body_gp.likelihood == -1 * fun_result)
        assert(two_body_gp.likelihood_gradient == -1 * jac_result)


class TestConstraint():

    def test_constrained_optimization_simple(self, all_gps):
        """
        Test constrained optimization with a standard
        number of hyperparameters (3 for a 3-body)
        :return:
        """

        test_gp = all_gps[True]
        orig_hyp_labels = test_gp.hyp_labels
        orig_hyps = np.copy(test_gp.hyps)

        test_gp.hyps_mask['map'] = np.array([1, 3, 5], dtype=int)
        test_gp.hyps_mask['train_noise'] = False
        test_gp.hyps_mask['original'] = orig_hyps

        test_gp.hyp_labels = orig_hyp_labels[test_gp.hyps_mask['map']]
        test_gp.hyps = orig_hyps[test_gp.hyps_mask['map']]

        # Check that the hyperparameters were updated
        test_gp.maxiter = 1
        init_hyps = np.copy(test_gp.hyps)
        results = test_gp.train()
        assert not np.equal(results.x, init_hyps).all()


class TestAlgebra():

    @pytest.mark.parametrize('par, per_atom_par, n_cpus',
                             [(False, False, 1),
                              (True, True, 2),
                              (True, False, 2)])
    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_predict(self, all_gps, validation_env,
                     par, per_atom_par, n_cpus, multihyps):
        test_gp = all_gps[multihyps]
        test_gp.parallel = par
        test_gp.per_atom_par = per_atom_par
        pred = test_gp.predict(x_t=validation_env, d=1)
        assert (len(pred) == 2)
        assert (isinstance(pred[0], float))
        assert (isinstance(pred[1], float))

    @pytest.mark.parametrize('par, per_atom_par, n_cpus',
                             [(False, False, 1),
                              (True, True, 2),
                              (True, False, 2)])
    def test_predict_efs(self, two_plus_three_gp, validation_env,
                         par, per_atom_par, n_cpus):
        """Check that energy, force, and stress prediction matches the other
        GP prediction methods."""

        test_gp = two_plus_three_gp
        test_gp.parallel = par
        test_gp.per_atom_par = per_atom_par
        en_pred, force_pred, stress_pred, en_var, force_var, stress_var = \
            test_gp.predict_efs(validation_env)

        en_pred_2, en_var_2 = \
            test_gp.predict_local_energy_and_var(validation_env)

        force_pred_2, force_var_2 = \
            test_gp.predict(validation_env, 1)

        assert(np.isclose(en_pred, en_pred_2))
        assert(np.isclose(en_var, en_var_2))
        assert(np.isclose(force_pred[0], force_pred_2))
        assert(np.isclose(force_var[0], force_var_2))

    @pytest.mark.parametrize('par, n_cpus', [(True, 2),
                                             (False, 1)])
    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_set_L_alpha(self, all_gps, params, par, n_cpus, multihyps):
        test_gp = all_gps[multihyps]
        test_gp.parallel = par
        test_gp.n_cpus = n_cpus
        test_gp.set_L_alpha()

    @pytest.mark.parametrize('par, n_cpus', [(True, 2),
                                             (False, 1)])
    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_update_L_alpha(self, all_gps, params, par, n_cpus, multihyps):
        # set up gp model
        test_gp = all_gps[multihyps]
        test_gp.parallel = par
        test_gp.n_cpus = n_cpus

        test_structure, forces = \
            get_random_structure(params['cell'], params['unique_species'], 2)
        energy = 3.14                 
        test_gp.check_L_alpha()
        test_gp.update_db(test_structure, forces, energy=energy)
        test_gp.update_L_alpha()

        # compare results with set_L_alpha
        ky_mat_from_update = np.copy(test_gp.ky_mat)
        test_gp.set_L_alpha()
        ky_mat_from_set = np.copy(test_gp.ky_mat)

        assert (np.all(np.absolute(ky_mat_from_update - ky_mat_from_set)) < 1e-6)


class TestIO():
    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_representation_method(self, all_gps, multihyps):
        test_gp = all_gps[multihyps]
        the_str = str(test_gp)
        assert 'GaussianProcess Object' in the_str
        if (multihyps):
            assert 'Kernel: two_three_many_body_mc' in the_str
        else:
            assert 'Kernel: two_plus_three_plus_many_body_mc' in the_str
        assert 'Cutoffs: [0.8 0.8 0.8]' in the_str
        assert 'Model Likelihood: ' in the_str
        if not multihyps:
            assert 'Length: ' in the_str
            assert 'Signal Var.: ' in the_str
            assert "Noise Var.: " in the_str

    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_serialization_method(self, all_gps, validation_env, multihyps):
        """
        Serialize and then un-serialize a GP and ensure that no info was lost.
        Compare one calculation to ensure predictions work correctly.
        :param test_gp:
        :return:
        """
        test_gp = all_gps[multihyps]
        old_gp_dict = test_gp.as_dict()
        new_gp = GaussianProcess.from_dict(old_gp_dict)
        new_gp_dict = new_gp.as_dict()

        assert len(new_gp_dict) == len(old_gp_dict)

        dumpcompare(new_gp_dict, old_gp_dict)

        for d in [1, 2, 3]:
            assert np.all(test_gp.predict(x_t=validation_env, d=d) ==
                          new_gp.predict(x_t=validation_env, d=d))

    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_load_and_reload(self, all_gps, validation_env, multihyps):

        test_gp = all_gps[multihyps]

        test_gp.write_model('test_gp_write', 'pickle')

        new_gp = GaussianProcess.from_file('test_gp_write.pickle')

        for d in [1, 2, 3]:
            assert np.all(test_gp.predict(x_t=validation_env, d=d) ==
                          new_gp.predict(x_t=validation_env, d=d))
        os.remove('test_gp_write.pickle')

        test_gp.write_model('test_gp_write', 'json')

        with open('test_gp_write.json', 'r') as f:
            new_gp = GaussianProcess.from_dict(json.loads(f.readline()))
        for d in [1, 2, 3]:
            assert np.all(test_gp.predict(x_t=validation_env, d=d) ==
                          new_gp.predict(x_t=validation_env, d=d))
        os.remove('test_gp_write.json')

        with raises(ValueError):
            test_gp.write_model('test_gp_write', 'cucumber')


    def test_load_reload_huge(self, all_gps):
        """
        Unit tests that loading and reloading a huge GP works.
        :param all_gps:
        :return:
        """
        test_gp = deepcopy(all_gps[False])
        test_gp.set_L_alpha()
        dummy_gp = deepcopy(test_gp)
        dummy_gp.training_data = [1]*5001

        prev_ky_mat = deepcopy(dummy_gp.ky_mat)
        prev_l_mat = deepcopy(dummy_gp.l_mat)

        dummy_gp.training_data = [1]*5001
        test_gp.write_model('test_gp_write', 'json')
        new_gp = GaussianProcess.from_file('test_gp_write.json')
        assert np.array_equal(prev_ky_mat, new_gp.ky_mat)
        assert np.array_equal(prev_l_mat, new_gp.l_mat)

        os.remove('test_gp_write.json')




def dumpcompare(obj1, obj2):
    '''this source code comes from
    http://stackoverflow.com/questions/15785719/how-to-print-a-dictionary-line-by-line-in-python'''

    if isinstance(obj1, dict):

        assert len(obj1.keys()) == len(
            obj2.keys()), f"key1 {list(obj1.keys())}, \n key2 {list(obj2.keys())}"

        for k1, k2 in zip(sorted(obj1.keys()), sorted(obj2.keys())):

            assert k1 == k2, f"key {k1} is not the same as {k2}"

            print(k1)

            if (k1 != "name"):
                if (obj1[k1] is None):
                    continue
                else:
                    assert dumpcompare(obj1[k1], obj2[k2]
                                   ), f"value {k1} is not the same as {k2}"

    elif isinstance(obj1, (list, tuple)):

        assert len(obj1) == len(obj2)
        for k1, k2 in zip(obj1, obj2):
            assert dumpcompare(k1, k2), f"list elements are different"

    elif isinstance(obj1, np.ndarray):
        if (obj1.size == 0 or obj1.size == 1):  # TODO: address None edge case
            pass
        elif (not isinstance(obj1[0], np.str_)):
            assert np.equal(obj1, obj2).all(), "ndarray is not all the same"
        else:
            for xx, yy in zip(obj1, obj2):
                assert dumpcompare(xx, yy)
    else:
        assert obj1 == obj2

    return True


def test_training_statistics():
    """
    Ensure training statistics are being recorded correctly
    :return:
    """

    test_structure, forces = \
        get_random_structure(np.eye(3), ['H', 'Be'], 10)
    energy = 3.14

    gp = GaussianProcess(kernel_name='2', cutoffs=[10])

    data = gp.training_statistics

    assert data['N'] == 0
    assert len(data['species']) == 0
    assert len(data['envs_by_species']) == 0

    gp.update_db(test_structure, forces, energy=energy)

    data = gp.training_statistics

    assert data['N'] == 10
    assert len(data['species']) == len(set(test_structure.coded_species))
    assert len(data['envs_by_species']) == len(set(
        test_structure.coded_species))


class TestHelper():

    def test_adjust_cutoffs(self, all_gps):

        test_gp = all_gps[False]
        # global training data
        # No need to ajust the other global values since we're not
        # testing on the predictions made, just that the cutoffs in the
        # atomic environments are correctly re-created

        old_cutoffs = np.copy(test_gp.cutoffs)

        test_gp.adjust_cutoffs(np.array(test_gp.cutoffs) + .5, train=False)

        assert np.array_equal(test_gp.cutoffs, old_cutoffs + .5)

        for env in test_gp.training_data:
            assert np.array_equal(env.cutoffs, test_gp.cutoffs)
