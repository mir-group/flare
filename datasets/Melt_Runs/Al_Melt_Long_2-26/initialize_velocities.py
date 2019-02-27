import numpy as np


def get_random_velocities(noa: int, temperature: float, mass: float):
    """Draw velocities from Maxwell-Boltzmann distribution. Assumes mass
    is given in amu."""

    mass_md = mass * 0.000103642695727
    kb = 0.0000861733034
    std = np.sqrt(kb * temperature / mass_md)
    velocities = np.random.normal(scale=std, size=(noa, 3))

    vel_sum = np.sum(velocities, axis=0)
    corrected_velocities = velocities - vel_sum / noa

    return corrected_velocities
