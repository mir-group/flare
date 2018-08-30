import pytest

from gp import GaussianProcess

@pytest.fixture
def rbf_gp():
    """Returns a GP instance with an rbf kernel"""
    return GaussianProcess(kernel='rbf')

@pytest.fixture
def structure():
    """Returns a Structure instance to test on"""
    pass


class TestGP():
    """Test GaussianProcess() class"""

    def test_init(self, rbf_gp):
        pass

    def test_train(self, rbf_gp):
        pass

    def test_opt_hyper(self):
        pass

    # example to show parametrized testing
    @pytest.mark.parametrize("sigma_f", "length_scale", "sigma_n", [
        (1, 1, .1),
        (10, 10, 1)])
    def test_set_kernel(self, rbf_gp, sigma_f, length_scale, sigma_n):
        pass

    def test_predict(self, structure, rbf_gp):
        pass


