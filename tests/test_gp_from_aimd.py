import pytest
import numpy as np
import sys
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.kernels import two_plus_three_body, two_plus_three_body_grad
from flare.gp import GaussianProcess
from flare.gp_from_aimd import Trajectory_Gp



def test_instantiation_of_trajectory_gp():

    fake_gp = GaussianProcess(kernel = two_plus_three_body,
                              kernel_grad=two_plus_three_body_grad,
                              hyps = np.array([]),
                              cutoffs = np.array([]))

    a = Trajectory_Gp(frames=[], gp=fake_gp)

    assert isinstance(a, Trajectory_Gp)

    _ = Trajectory_Gp([], fake_gp, parallel=True, calculate_energy=True)
    _ = Trajectory_Gp([], fake_gp, parallel=True, calculate_energy=False)
    _ = Trajectory_Gp([], fake_gp, parallel=False, calculate_energy=True)
    _ = Trajectory_Gp([], fake_gp, parallel=False, calculate_energy=False)
