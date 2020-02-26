import numpy as np
from typing import List


def update_positions(dt, noa, structure):
    dtdt = dt ** 2
    new_pos = np.zeros((noa, 3))

    for i, pre_pos in enumerate(structure.prev_positions):
        mass = structure.mass_dict[structure.species_labels[i]]
        pos = structure.positions[i]
        forces = structure.forces[i]

        new_pos[i] = 2 * pos - pre_pos + dtdt * forces / mass

    return new_pos


def calculate_temperature(new_pos, structure, dt, noa):
    # set velocity and temperature information
    velocities = (new_pos -
                  structure.prev_positions) / (2 * dt)

    KE = 0
    for i in range(len(structure.positions)):
        for j in range(3):
            KE += 0.5 * \
                structure.mass_dict[structure.species_labels[i]] * \
                velocities[i][j] * velocities[i][j]

    # see conversions.nb for derivation
    kb = 0.0000861733034

    # see p. 61 of "computer simulation of liquids"
    temperature = 2 * KE / ((3 * noa - 3) * kb)

    return KE, temperature, velocities


def get_random_velocities(noa: int, temperature: float, mass: float):
    """Draw velocities from the Maxwell-Boltzmann distribution, assuming a
    fixed mass for all particles in amu.

    Args:
        noa (int): Number of atoms in the system.
        temperature (float): Temperature of the system.
        mass (float): Mass of each particle in amu.

    Returns:
        np.ndarray: Particle velocities, corrected to give zero center of mass motion.
    """
    
    # Use FLARE mass units (time = ps, length = A, energy = eV)
    mass_md = mass * 0.000103642695727
    kb = 0.0000861733034
    std = np.sqrt(kb * temperature / mass_md)
    velocities = np.random.normal(scale=std, size=(noa, 3))
    
    # Remove center-of-mass motion
    vel_sum = np.sum(velocities, axis=0)
    corrected_velocities = velocities - vel_sum / noa

    return corrected_velocities


def multicomponent_velocities(temperature: float, masses: List[float]):
    """Draw velocities from the Maxwell-Boltzmann distribution for particles of
    varying mass.

    Args:
        temperature (float): Temperature of the system.
        masses (List[float]): Particle masses in amu.

    Returns:
        np.ndarray: Particle velocities, corrected to give zero center of mass motion.
    """

    noa = len(masses)
    velocities = np.zeros((noa, 3))
    kb = 0.0000861733034
    mom_tot = np.array([0., 0., 0.])

    for n, mass in enumerate(masses):
        # Convert to FLARE mass units (time = ps, length = A, energy = eV)
        mass_md = mass * 0.000103642695727
        std = np.sqrt(kb * temperature / mass_md)
        rand_vel = np.random.normal(scale=std, size=(3))
        velocities[n] = rand_vel
        mom_curr = rand_vel * mass_md
        mom_tot += mom_curr

    # Correct momentum, remove center of mass motion
    mom_corr = mom_tot / noa
    for n, mass in enumerate(masses):
        mass_md = mass * 0.000103642695727
        velocities[n] -= mom_corr / mass_md

    return velocities
