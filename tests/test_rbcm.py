
from flare.rbcm import RobustBayesianCommitteeMachine
import os as os

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, 'test_files')

def test_basic():

    rbcm = RobustBayesianCommitteeMachine()

    assert isinstance(rbcm,RobustBayesianCommitteeMachine)


def test_expert_growth():
    """
    Test that as data is added the data is allocated to experts correctly
    :return:
    """
    pass


def test_prediction():
    """
    Test that prediction functions work correctly
    :return:
    """

    pass