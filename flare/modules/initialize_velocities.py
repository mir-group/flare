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


# TODO: implement velocity function for multicomponent systems
def multicomponent_velocities(temperature, masses):
    noa = len(masses)
    velocities = np.zeros((noa, 3))
    kb = 0.0000861733034
    mom_tot = np.array([0., 0., 0.])

    for n, mass in enumerate(masses):
        mass_md = mass * 0.000103642695727
        std = np.sqrt(kb * temperature / mass_md)
        rand_vel = np.random.normal(scale=std, size=(3))
        velocities[n] = rand_vel
        mom_curr = rand_vel * mass_md
        mom_tot += mom_curr

    # correct momentum
    mom_corr = mom_tot / noa
    for n, mass in enumerate(masses):
        mass_md = mass * 0.000103642695727
        velocities[n] -= mom_corr / mass_md

    return velocities
