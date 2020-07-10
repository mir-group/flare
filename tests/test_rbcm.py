import numpy as np
import os
import pickle
import pytest

from copy import deepcopy
from pytest import raises
from scipy.optimize import OptimizeResult
from typing import List

import flare
from flare.predict import predict_on_structure
from flare.rbcm import RobustBayesianCommitteeMachine
from flare.env import AtomicEnvironment
from flare.struc import Structure

from tests.fake_gp import generate_hm, get_tstp, get_random_structure

multihyps_list = [False]


@pytest.fixture(scope='class')
def all_gps() -> RobustBayesianCommitteeMachine:
    """Returns a GP instance with a two-body numba-based kernel"""

    gp_dict = {True: None, False: None}
    for multihyps in multihyps_list:

        hyps, hm, cutoffs = generate_hm(1, 1, multihyps=multihyps)
        hl = hm['hyp_labels']

        # test update_db

        gp_dict[multihyps] = \
            RobustBayesianCommitteeMachine(n_experts=1, ndata_per_expert=2,
                                           prior_variance=0.1,
                                           per_expert_parallel=False,
                                           kernels=hm['kernels'],
                                           hyps=hyps,
                                           hyp_labels=hl,
                                           cutoffs=cutoffs,
                                           hyps_mask=hm,
                                           parallel=False, n_cpus=1)

        test_structure, forces = \
            get_random_structure(np.eye(3), [1, 2], 3)
        energy = 3.14

        gp_dict[multihyps].update_db(
            test_structure, forces)  # , energy=energy)

    yield gp_dict
    del gp_dict


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
    np.random.seed(0)
    test_pt = get_tstp(None)
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
        test_gp.update_db(test_structure, forces)  # , energy=energy)

        # assert (len(test_gp.training_data) == params['noa']+oldsize)
        # assert (len(test_gp.training_labels_np) == (params['noa']+oldsize)*3)


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
        hyps = tuple(test_gp.hyps)

        test_gp.train()

        hyp_post = tuple(test_gp.hyps)

        # check if hyperparams have been updated
        assert (hyps != hyp_post)

    def test_train_failure(self, all_gps, params, mocker):
        """
        Tests the case when 'L-BFGS-B' fails due to a linear algebra error and
        training falls back to BFGS
        """
        # Sets up mocker for scipy minimize. Note that we are mocking
        # 'flare.rbcm.minimize' because of how the imports are done in gp
        x_result = np.random.rand()
        fun_result = np.random.rand()
        jac_result = np.random.rand()
        train_result = OptimizeResult(x=x_result, fun=fun_result,
                                      jac=jac_result)

        side_effects = [np.linalg.LinAlgError(), train_result]
        mocker.patch('flare.rbcm.minimize', side_effect=side_effects)
        two_body_gp = all_gps[multihyps_list[-1]]
        two_body_gp.set_L_alpha = mocker.Mock()

        # Executes training
        two_body_gp.algo = 'L-BFGS-B'
        two_body_gp.train()

        # Assert that everything happened as expected
        assert(flare.rbcm.minimize.call_count == 2)

        calls = flare.rbcm.minimize.call_args_list
        args, kwargs = calls[0]
        assert(kwargs['method'] == 'L-BFGS-B')

        args, kwargs = calls[1]
        assert(kwargs['method'] == 'BFGS')

        two_body_gp.set_L_alpha.assert_called_once()
        assert(two_body_gp.hyps == x_result)
        # assert(two_body_gp.likelihood == -1 * fun_result)
        # assert(two_body_gp.likelihood_gradient == -1 * jac_result)


class TestConstraint():

    def test_constrained_optimization_simple(self, all_gps):
        """
        Test constrained optimization with a standard
        number of hyperparameters (3 for a 3-body)
        :return:
        """

        test_gp = all_gps[multihyps_list[-1]]

        hyps, hm, cutoffs = generate_hm(1, 1, constraint=True, multihyps=True)

        test_gp.hyps_mask = hm
        test_gp.hyp_labels = hm['hyp_labels']
        test_gp.hyps = hyps
        test_gp.update_kernel(hm['kernel_name'], "mc", hm)
        test_gp.set_L_alpha()

        hyp = list(test_gp.hyps)

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
        pred = test_gp.predict(x_t=validation_env)
        assert (len(pred) == 2)
        assert (isinstance(pred[0][0], float))
        assert (isinstance(pred[1][0], float))

    @pytest.mark.parametrize('par, n_cpus', [(True, 2),
                                             (False, 1)])
    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_set_L_alpha(self, all_gps, params, par, n_cpus, multihyps):
        test_gp = all_gps[multihyps]
        test_gp.parallel = par
        test_gp.n_cpus = n_cpus
        test_gp.set_L_alpha()
        for i in range(test_gp.n_experts):
            print(i, len(test_gp.training_labels[i]), test_gp.ky_mat[i].shape)

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

        i = test_gp.n_experts - 1

        print(len(test_gp.training_labels[i]))
        print(test_gp.ky_mat[i].shape)

        test_gp.check_L_alpha()
        test_gp.update_db(test_structure, forces,  # energy=energy,
                          expert_id=i)
        print(len(test_gp.training_labels[i]))
        print(test_gp.ky_mat[i].shape)

        test_gp.update_L_alpha(i)
        # compare results with set_L_alpha

        ky_mat_from_update = np.copy(test_gp.ky_mat[i])
        test_gp.set_L_alpha()
        test_gp.set_L_alpha_part(i)
        ky_mat_from_set = np.copy(test_gp.ky_mat[i])

        assert (np.all(np.absolute(ky_mat_from_update - ky_mat_from_set)) < 1e-6)


class TestIO():
    # @pytest.mark.parametrize('multihyps', multihyps_list)
    # def test_representation_method(self, all_gps, multihyps):
    #     test_gp = all_gps[multihyps]
    #     the_str = str(test_gp)
    #     assert 'RobustBayesianCommitteeMachine Object' in the_str
    #     assert 'Kernel: [\'twobody\', \'threebody\']' in the_str #, \'manybody\']' in the_str
    #     assert 'Cutoffs: {\'twobody\': 0.8, \'threebody\': 0.8}' in the_str #, \'manybody\': 0.8}' in the_str
    #     assert 'Model Likelihood: ' in the_str
    #     if not multihyps:
    #         assert 'Length ' in the_str
    #         assert 'Signal Var. ' in the_str
    #         assert "Noise Var." in the_str

    # @pytest.mark.parametrize('multihyps', multihyps_list)
    # def test_serialization_method(self, all_gps, validation_env, multihyps):
    #     """
    #     Serialize and then un-serialize a GP and ensure that no info was lost.
    #     Compare one calculation to ensure predictions work correctly.
    #     :param test_gp:
    #     :return:
    #     """
    #     test_gp = all_gps[multihyps]
    #     old_gp_dict = test_gp.as_dict()
    #     new_gp = RobustBayesianCommitteeMachine.from_dict(old_gp_dict)
    #     new_gp_dict = new_gp.as_dict()

    #     assert len(new_gp_dict) == len(old_gp_dict)

    #     dumpcompare(new_gp_dict, old_gp_dict)

    #     test_predict = np.hstack(test_gp.predict(x_t=validation_env))
    #     new_predict = np.hstack(new_gp.predict(x_t=validation_env))
    #     assert np.array_equal(test_predict, new_predict)
    #     assert new_gp.training_data is not test_gp.training_data

    @pytest.mark.parametrize('multihyps', multihyps_list)
    def test_load_and_reload(self, all_gps, validation_env, multihyps):

        test_gp = all_gps[multihyps]

        test_gp.write_model('test_gp_write', 'pickle')

        new_gp = RobustBayesianCommitteeMachine.from_file(
            'test_gp_write.pickle')

        test_predict = np.hstack(test_gp.predict(x_t=validation_env))
        new_predict = np.hstack(new_gp.predict(x_t=validation_env))
        assert np.array_equal(test_predict, new_predict)

        try:
            os.remove('test_gp_write.pickle')
        except:
            pass

        # Test logic for auto-detecting format in write command
        for format in ['pickle']:
            write_string = 'format_write_test.'+format
            if os.path.exists(write_string):
                os.remove(write_string)

            test_gp.write_model(write_string)
            assert os.path.exists(write_string)
            os.remove(write_string)


# def test_training_statistics():
#     """
#     Ensure training statistics are being recorded correctly
#     :return:
#     """
#
#     test_structure, forces = \
#         get_random_structure(np.eye(3), ['H', 'Be'], 10)
#     energy = 3.14
#
#     gp = RobustBayesianCommitteeMachine(n_experts=1, ndata_per_expert=2,
#                             prior_variance=0.1,
#                             per_expert_parallel=False,
#                             kernel_name='2', cutoffs=[10])
#
#     data = gp.training_statistics
#
#     assert data['N'] == 0
#     assert len(data['species']) == 0
#     assert len(data['envs_by_species']) == 0
#
#     gp.update_db(test_structure, forces, energy=energy)
#
#     data = gp.training_statistics
#
#     assert data['N'] == 10
#     assert len(data['species']) == len(set(test_structure.coded_species))
#     assert len(data['envs_by_species']) == len(set(
#         test_structure.coded_species))


# def test_remove_force_data():
#     """
#     Train a GP on one fake structure. Store forces from prediction.
#     Add a new fake structure and ensure predictions change; then remove
#     the structure and ensure predictions go back to normal.
#     :return:
#     """
#
#     test_structure, forces = get_random_structure(5.0*np.eye(3),
#                                                   ['H', 'Be'],
#                                                   5)
#
#     test_structure_2, forces_2 = get_random_structure(5.0*np.eye(3),
#                                                   ['H', 'Be'],
#                                                   5)
#
#     gp = RobustBayesianCommitteeMachine(n_experts=1, ndata_per_expert=2,
#                             prior_variance=0.1,
#                             per_expert_parallel=False,
#                             kernels=['twobody'], cutoffs={'twobody':0.8})
#
#     gp.update_db(test_structure, forces)
#
#     with raises(ValueError):
#         gp.remove_force_data(1000000)
#
#     init_forces, init_stds = predict_on_structure(test_structure, gp,
#                                                  write_to_structure=False)
#     init_forces_2, init_stds_2 = predict_on_structure(test_structure_2, gp,
#                                                  write_to_structure=False)
#
#     # Alternate adding in the entire structure and adding in only one atom.
#     for custom_range in [None, [0]]:
#
#         # Add in data and ensure the predictions change in reponse
#         gp.update_db(test_structure_2, forces_2, custom_range=custom_range)
#
#         new_forces, new_stds = predict_on_structure(test_structure, gp,
#                                                     write_to_structure=False)
#
#         new_forces_2, new_stds_2 = predict_on_structure(test_structure_2, gp,
#                                                      write_to_structure=False)
#
#         assert not np.array_equal(init_forces, new_forces)
#         assert not np.array_equal(init_forces_2, new_forces_2)
#         assert not np.array_equal(init_stds, new_stds)
#         assert not np.array_equal(init_stds_2, new_stds_2)
#
#         # Remove that data and test to see that the predictions revert to
#         # what they were previously
#         if custom_range == [0]:
#             popped_strucs, popped_forces = gp.remove_force_data(5)
#         else:
#             popped_strucs, popped_forces = gp.remove_force_data([5, 6, 7, 8,
#                                                                  9])
#
#         for i in range(len(popped_forces)):
#             assert np.array_equal(popped_forces[i],forces_2[i])
#             assert np.array_equal(popped_strucs[i].structure.positions,
#                               test_structure_2.positions)
#
#         final_forces, final_stds = predict_on_structure(test_structure, gp,
#                                                      write_to_structure=False)
#         final_forces_2, final_stds_2 = predict_on_structure(test_structure_2, gp,
#                                                      write_to_structure=False)
#
#         assert np.array_equal(init_forces, final_forces)
#         assert np.array_equal(init_stds, final_stds)
#
#         assert np.array_equal(init_forces_2, final_forces_2)
#         assert np.array_equal(init_stds_2, final_stds_2)


# class TestHelper():
#
#     def test_adjust_cutoffs(self, all_gps):
#
#         test_gp = all_gps[False]
#         # global training data
#         # No need to ajust the other global values since we're not
#         # testing on the predictions made, just that the cutoffs in the
#         # atomic environments are correctly re-created
#
#         old_cutoffs = {}
#         new_cutoffs = {}
#         for k in test_gp.cutoffs:
#             old_cutoffs[k] = test_gp.cutoffs[k]
#             new_cutoffs[k] = 0.5+old_cutoffs[k]
#         test_gp.hyps_mask['cutoffs']=new_cutoffs
#         test_gp.adjust_cutoffs(new_cutoffs, train=False, new_hyps_mask=test_gp.hyps_mask)
#
#         assert np.array_equal(list(test_gp.cutoffs.values()), np.array(list(old_cutoffs.values()), dtype=float) + .5)
#
#         for env in test_gp.training_data:
#             assert env.cutoffs == test_gp.cutoffs