from typing import List

import numpy as np


def update_positions(dt, noa, structure):
    dtdt = dt ** 2
    new_pos = np.zeros((noa, 3))

    for i, pre_pos in enumerate(structure.prev_positions):
        mass = structure.mass_dict[structure.species_labels[i]]
        pos = structure.positions[i]
        forces = structure.forces[i]

        new_pos[i] = 2 * pos - pre_pos + dtdt * forces / mass

    # Update previous and current positions.
    structure.prev_positions = np.copy(structure.positions)
    structure.positions = new_pos
    structure.positions[:] = structure.wrap_positions()


def calculate_temperature(structure, dt, noa):
    # Set velocity and temperature information.
    velocities = (structure.positions - structure.prev_positions) / dt

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
