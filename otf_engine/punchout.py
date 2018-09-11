#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to facilitate the 'punch-out' scheme of large-scale GP-accelerated
molecular dynamics simulations.

Steven Torrisi
"""

import numpy as np

from env import ChemicalEnvironment

from struc import Structure


def is_within_d_box(pos1: np.array, pos2: np.array, d: float) -> bool:
    """
    Return if pos2 is within a cube of side length d centered around pos1,
    :param pos1: First position
    :type pos1: np.array
    :param pos2: Second position
    :type pos2: np.array
    :param d: Side length of cube
    :type d: float
    :return: bool
    """

    isnear_x = abs(pos1[0] - pos2[0]) <= d / 2
    isnear_y = abs(pos1[1] - pos2[1]) <= d / 2
    isnear_z = abs(pos1[2] - pos2[2]) <= d / 2

    return isnear_x and isnear_y and isnear_z


def punchout(structure: Structure, atom: int, d: float, center: bool = True):
    """
    Punch out a cube of side-length d around the target atom

    :param structure: Structure to punch out a sub-structure from
    :type structure: Structure
    :param atom: Index of atom to punch out around
    :type atom: int
    :param d: Side length of cube around target atom
    :type d: float
    :param center: Return structure centered around 0 or not
    :type center: bool
    """

    # TODO replace this with the env's method for finding neighbors
    a = structure.vec1
    b = structure.vec2
    c = structure.vec3

    shift_vectors = [np.zeros(3),
                     a, b, c,
                     -a, -b, -c,
                     a + b, a - b, a + c, a - c,
                     b + c, b - c,
                     -a + b, -a - b,
                     -a + c, -a - c,
                     -b + c, -b - c,
                     a + b + c,
                     -a + b + c, a - b + c, a + b - c,
                     -a - b + c, -a + b - c, a - b - c,
                     -a - b - c
                     ]

    new_pos = []
    new_prev_pos = []  # Necessary so new structure has non-zero velocity
    new_species = []

    assert 0 <= atom <= len(structure.positions), 'Atom index  is greater ' \
                                                  'than number of atoms ' \
                                                  'in structure'
    target_pos = structure.positions[atom]

    for i, pos in enumerate(structure.positions):
        # Compute all shifted positions to handle edge cases
        shifted_positions = [pos + shift for shift in shift_vectors]

        for j, shifted_pos in enumerate(shifted_positions):
            if is_within_d_box(target_pos, shifted_pos, d):
                new_pos.append(shifted_pos)
                new_prev_pos.append(structure.prev_positions
                                    [i] + shift_vectors[j])
                new_species.append(structure.species[i])

    # Set up other new structural properties
    newlatt = d * np.eye(3)
    new_mass_dict = {}

    for spec in set(new_species):
        new_mass_dict[spec] = structure.mass_dict[spec]

    # Instantiate new structure, and set the previous positions manually

    newstruc = Structure(newlatt, new_species, new_pos,
                         structure.cutoff, new_mass_dict)
    newstruc.prev_positions = list(new_prev_pos)

    # Check if any atoms are unphysically close to each other, add padding
    # accordingly

    # TODO put in the new method here, except applied on the newstructure

    # Sketch of how it will work...

    # BELOW CODE CURRENTLY DOES NOTHING; awaiting update to env.py
    for i, pos in enumerate(newstruc.positions):
        # Compute all shifted positions to handle edge cases
        shifted_positions = [pos + shift for shift in shift_vectors if
                             not np.equal(shift, np.zeros(3)).all()]

        for j, shifted_pos in enumerate(shifted_positions):
            if is_within_d_box(pos, shifted_pos, 1.0):
                # shift box
                pass

    # Center new structure at origin
    if center:
        newstruc.translate_positions(-target_pos)

    return newstruc


if __name__ == '__main__':
    pass
