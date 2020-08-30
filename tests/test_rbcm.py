from flare.rbcm import RobustBayesianCommitteeMachine
import os as os
from flare.struc import Structure
from flare.env import AtomicEnvironment

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

    rbcm.add_one_env()

    pass


def test_prediction():
    """
    Test that prediction functions work correctly
    :return:
    """

    pass
