"""
Utility functions for various tasks.
"""
from warnings import warn
from json import JSONEncoder
from typing import List
from math import inf

import numpy as np


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


def get_supercell_positions(sc_size: int, cell: np.ndarray,
                            positions: np.ndarray):
    """Returns the positions of a supercell of atoms, with the number of cells
    in each direction fixed.

    Args:
        sc_size (int): Size of the supercell.
        cell (np.ndarray): 3x3 array of cell vectors.
        positions (np.ndarray): Positions of atoms in the unit cell.

    Returns:
        np.ndarray: Positions of atoms in the supercell.
    """

    sc_positions = []
    for m in range(sc_size):
        vec1 = m * cell[0]
        for n in range(sc_size):
            vec2 = n * cell[1]
            for p in range(sc_size):
                vec3 = p * cell[2]

                # append translated positions
                for pos in positions:
                    sc_positions.append(pos+vec1+vec2+vec3)

    return np.array(sc_positions)


def supercell_custom(cell: np.ndarray, positions: np.ndarray,
                     size1: int, size2: int, size3: int):
    """Returns the positions of a supercell of atoms with a chosen number of
    cells in each direction.

    Args:
        cell (np.ndarray): 3x3 array of cell vectors.
        positions (np.ndarray): Positions of atoms in the unit cell.
        size1 (int): Number of cells along the first cell vector.
        size2 (int): Number of cells along the second cell vector.
        size3 (int): Number of cells along the third cell vector.

    Returns:
        np.ndarray: Positions of atoms in the supercell.
    """

    sc_positions = []
    for m in range(size1):
        vec1 = m * cell[0]
        for n in range(size2):
            vec2 = n * cell[1]
            for p in range(size3):
                vec3 = p * cell[2]

                # append translated positions
                for pos in positions:
                    sc_positions.append(pos+vec1+vec2+vec3)

    return np.array(sc_positions)
