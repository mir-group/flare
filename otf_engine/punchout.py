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
from typing import List


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



def punchout_structure(structure: Structure, atom: int, d: float, center:
bool = True,neighbor_step = 2,isolated_cluster_d = 0):
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

    assert 0 <= atom <= len(structure.positions), 'Atom index  is greater ' \
                                                  'than number of atoms ' \
                                                  'in structure'
    a = structure.vec1
    b = structure.vec2
    c = structure.vec3

    #TODO determine neighbor_step automatically by determining extent of
    # cell and using d to figure out 'how far' to reach

    shift_vectors=[]

    for i in range(-neighbor_step,neighbor_step+1):
        for j in range(-neighbor_step,neighbor_step+1):
            for k in range(-neighbor_step,neighbor_step+1):
                shift_vectors.append(a*i + b *j + c *k )

    new_pos = []
    new_prev_pos = []
    new_species = []

    target_pos = structure.positions[atom]

    # Loop through all atoms and see which are within d of target position
    for i, pos in enumerate(structure.positions):
        # Compute all shifted positions of atom i
        shifted_positions = [pos + shift for shift in shift_vectors]

        for j, shifted_pos in enumerate(shifted_positions):
            # Record the position and species of atoms which are near the
            # shifted position
            if is_within_d_box(target_pos, shifted_pos, d):
                new_pos.append(shifted_pos)
                new_prev_pos.append(structure.prev_positions
                                    [i] + shift_vectors[j])
                new_species.append(structure.species[i])

    # Set up other new structural properties
    newlatt = d * np.eye(3) + isolated_cluster_d * np.eye(3)
    new_mass_dict = {}

    for spec in set(new_species):
        new_mass_dict[spec] = structure.mass_dict[spec]

    # Instantiate new structure, and set the previous positions manually

    newstruc = Structure(newlatt, species=new_species, positions=new_pos,
                     cutoff=structure.cutoff, mass_dict=new_mass_dict,
                     prev_positions=list(new_prev_pos))

    if center:
        newstruc.translate_positions(-np.array(target_pos))

    return newstruc


def punchout(structure: Structure, atom: int, d: float, center:
bool = True, check_too_close: float = 0, check_stoichiometry: List[str] = [],
             check_bond_break = None,
             adjust_tries = 100,
             isolated_cluster_d : float = 0):
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
    
    :param check_too_close: Check if two atoms are closer than radius of arg
    :type check_too_close: float
    
    :param check_stoichiometry: Check if new structure is modulo a given 
    unit cell
    :type check_stoichiometry: List[str]
    
    :param check_bond_break: Ensure each atom of type A has other elements 
    within a given radius around it
    :type check_bond_break: dict

    """

    newstruc = punchout_structure(structure=structure,atom=atom, d=d,
                                  center=center,
                                  isolated_cluster_d = isolated_cluster_d)

    # Check to see if any two atoms are unphysically close

    passing_structure = False
    newd = float(d)
    adjust_counter = 0
    while passing_structure == False:

        passing_structure = True


        if check_too_close:

            # Loop through atoms structure to find if any are within
            # check_too_close radius

            for i in range(newstruc.nat):
                bond_array, _, _, _ = \
                    ChemicalEnvironment.get_atoms_within_cutoff(newstruc, atom=i)
                for array in bond_array:
                    if array[0]<check_too_close:
                        passing_structure = False


        if check_stoichiometry:

            # Ensure that the ratio of atoms in cell follows prescribed
            # species list

            # Get count of each element from new structure
            spec_dict = newstruc.get_species_count()


            assert set(spec_dict.keys()) == set(check_stoichiometry)

            unique_species = list(set(check_stoichiometry))

            species_ratio = {}


            # Check that ratios of each atom in the new structure
            # corresponds to those in the check_bond_break list

            for element in unique_species:
                n_el = check_stoichiometry.count(element)

                species_ratio[element] = spec_dict[element]/n_el

            for val in species_ratio.values():
                if not np.mod(val, 1) == 0:
                    passing_structure = False

            if len(set(species_ratio.values())) != 1:
                passing_structure = False


        # Check if each atom of interest has at least another atom of that
        # interest nearby
        if check_bond_break:

            pass


        # Terminate loop if the structure passes;
        # else, re-size the box and try again
        if passing_structure:
            break
        else:

            if adjust_counter == adjust_tries:
                raise RuntimeError("Could not punch out cell to satisfy "
                                   "required properties.")

            adjust_counter+=1
            newd = newd * 1.01
            newstruc = punchout_structure(structure,atom=atom, d=newd,
                      center=center, isolated_cluster_d=isolated_cluster_d)

    # TODO put in the new method here, except applied on the newstructure

    return newstruc


if __name__ == '__main__':
    pass
