from flare.rbcm import RobustBayesianCommitteeMachine
from flare.gp import GaussianProcess
import os as os
import numpy as np
from flare.struc import Structure
from flare.env import AtomicEnvironment
from flare.gp_algebra import get_kernel_vector

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


methanol_frames = Structure.from_file(
    os.path.join(TEST_FILE_DIR, "methanol_frames.json")
)

methanol_envs = AtomicEnvironment.from_file(
    os.path.join(TEST_FILE_DIR, "methanol_envs.json")
)


def test_basic():

    rbcm = RobustBayesianCommitteeMachine()

    assert isinstance(rbcm, RobustBayesianCommitteeMachine)


def test_expert_growth():
    """
    Test that as data is added the data is allocated to experts correctly
    :return:
    """

    rbcm = RobustBayesianCommitteeMachine(ndata_per_expert=2)
    a = methanol_envs
    for env in methanol_envs[:5]:
        rbcm.add_one_env(env, env.force)

    assert rbcm.n_experts == 2


def test_prediction():
    """
    Test that prediction functions works.
    The RBCM in the 1-expert case *does not* reduce to a GP's predictions,
    because the way the mean and variance is computed for each expert
    is weighted based on the expert's performance on the entire dataset in a way
    that does not yield 1 in the absence of other experts.

    Hence, perform the relevant transformations on a GP's prediction
    and check it against the RBCM's.
    :return:
    """
    prior_var = 0.1
    rbcm = RobustBayesianCommitteeMachine(
        ndata_per_expert=100,
        prior_variance=prior_var,
    )
    gp = GaussianProcess()

    envs = methanol_envs[:10]

    for env in envs:
        rbcm.add_one_env(env, env.force)
        gp.add_one_env(env, env.force, train=False)

    struc = methanol_frames[-1]
    gp.update_db(struc, forces=struc.forces)
    rbcm.update_db(struc, forces=struc.forces)
    test_env = methanol_envs[-1]

    for d in [1, 2, 3]:
        assert np.array_equal(gp.hyps, rbcm.hyps)
        rbcm_pred = rbcm.predict(test_env, d)
        gp_pred = gp.predict(test_env, d)
        gp_kv = get_kernel_vector(
            gp.name,
            gp.kernel,
            gp.energy_force_kernel,
            test_env,
            d,
            gp.hyps,
            cutoffs=gp.cutoffs,
            hyps_mask=gp.hyps_mask,
            n_cpus=1,
            n_sample=gp.n_sample,
        )
        gp_mean = np.matmul(gp_kv, gp.alpha)
        assert gp_mean == gp_pred[0]
        gp_self_kern = gp.kernel(
            env1=test_env,
            env2=test_env,
            d1=d,
            d2=d,
            hyps=gp.hyps,
            cutoffs=np.array((7, 3.5)),
        )

        gp_var_i = gp_self_kern - np.matmul(np.matmul(gp_kv.T, gp.ky_mat_inv), gp_kv)
        gp_beta = 0.5 * (np.log(prior_var) - np.log(gp_var_i))
        mean = gp_mean * gp_beta / gp_var_i
        var = gp_beta / gp_var_i + (1 - gp_beta) / prior_var
        pred_var = 1.0 / var
        pred_mean = pred_var * mean

        assert pred_mean == rbcm_pred[0]
        assert pred_var == rbcm_pred[1]
