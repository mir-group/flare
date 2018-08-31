"""
Jon V
"""

import numpy as np
from math import exp
from numba import njit


# get two body kernel between chemical environments
@njit
def two_body_jit(bond_array_1, bond_types_1, bond_array_2,
                 bond_types_2, d1, d2, sig, ls):
    d = sig*sig/(ls*ls*ls*ls)
    e = ls*ls
    f = 1/(2*ls*ls)
    kern = 0

    x1_len = len(bond_types_1)
    x2_len = len(bond_types_2)

    for m in range(x1_len):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]
        typ1 = bond_types_1[m]

        for n in range(x2_len):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]
            typ2 = bond_types_2[n]

            # check that bonds match
            if typ1 == typ2:
                rr = (r1-r2)*(r1-r2)
                kern += d*exp(-f*rr)*coord1*coord2*(e-rr)

    return kern


class TwoBodyEnvironment:

    def __init__(self, positions, species, cell, cutoff, atom):
        # set permanent features if needed
        if not hasattr(self, 'bond_list'):
            TwoBodyEnvironment.bond_list = self.species_to_bonds(species)
            TwoBodyEnvironment.brav_inv = np.linalg.inv(cell)
            TwoBodyEnvironment.vec1 = cell[:, 0]
            TwoBodyEnvironment.vec2 = cell[:, 1]
            TwoBodyEnvironment.vec3 = cell[:, 2]
            TwoBodyEnvironment.cutoff = cutoff

        self.positions = positions
        self.species = species

        bond_array, bond_types, etyps, ctyp =\
            self.get_atoms_within_cutoff(atom)

        self.bond_array = bond_array
        self.bond_types = bond_types
        self.etyps = etyps
        self.ctyp = ctyp

    @staticmethod
    def species_to_bonds(species_list):
        """Converts list of species to a list of bonds.

        :param species_list: all species in the simulation
        :type species_list: list of strings
        :return: all possible bonds
        :rtype: list of lists of two strings
        """

        # get unique species
        unique_species = []
        for species in species_list:
            if species not in unique_species:
                unique_species.append(species)

        # create bond list
        bond_list = []
        for m in range(len(unique_species)):
            species_1 = unique_species[m]

            for n in range(m, len(unique_species)):
                species_2 = unique_species[n]
                bond_list.append([species_1, species_2])

        return bond_list

    @staticmethod
    def is_bond(species1, species2, bond):
        """Check if two species form a specified bond.

        :param species1: first species
        :type species1: str
        :param species2: second species
        :type species2: str
        :param bond: bond to be checked
        :type bond: list<str>
        :return: True or False
        :rtype: bool
        """

        return ([species1, species2] == bond) or ([species2, species1] == bond)

    def species_to_index(self, species1, species2):
        """Given two species, get the corresponding bond index.

        :param species1: first species
        :type species1: string
        :param species2: second species
        :type species2: string
        :param bond_list: all possible bonds
        :type bond_list: list
        :return: bond index
        :rtype: integer
        """

        for bond_index, bond in enumerate(self.bond_list):
            if TwoBodyEnvironment.is_bond(species1, species2, bond):
                return bond_index

    def get_local_atom_images(self, vec):
        """Get periodic images of an atom within the cutoff radius.

        :param vec: atomic position
        :type vec: nparray of shape (3,)
        :return: vectors and distances of atoms within cutoff radius
        :rtype: list of nparrays, list of floats
        """

        # get bravais coefficients
        coeff = np.matmul(self.brav_inv, vec)

        # get bravais coefficients for atoms within one super-super-cell
        coeffs = [[], [], []]
        for n in range(3):
            coeffs[n].append(coeff[n])
            coeffs[n].append(coeff[n]-1)
            coeffs[n].append(coeff[n]+1)
            coeffs[n].append(coeff[n]-2)
            coeffs[n].append(coeff[n]+2)

        # get vectors within cutoff
        vecs = []
        dists = []
        for m in range(len(coeffs[0])):
            for n in range(len(coeffs[1])):
                for p in range(len(coeffs[2])):
                    vec_curr = coeffs[0][m]*self.vec1 +\
                               coeffs[1][n]*self.vec2 +\
                               coeffs[2][p]*self.vec3
                    dist = np.linalg.norm(vec_curr)

                    if dist < self.cutoff:
                        vecs.append(vec_curr)
                        dists.append(dist)

        return vecs, dists

    # return information about atoms inside cutoff region
    def get_atoms_within_cutoff(self, atom):

        pos_atom = self.positions[atom]  # position of central atom
        central_type = self.species[atom]  # type of central atom

        bond_array = []
        bond_types = []
        environment_types = []

        # find all atoms and images in the neighborhood
        for n in range(len(self.positions)):
            diff_curr = self.positions[n] - pos_atom
            typ_curr = self.species[n]
            bond_curr = self.species_to_index(central_type, typ_curr)

            # get images within cutoff
            vecs, dists = self.get_local_atom_images(diff_curr)

            for vec, dist in zip(vecs, dists):
                # ignore self interaction
                if dist != 0:
                    environment_types.append(typ_curr)
                    bond_array.append([dist, vec[0]/dist, vec[1]/dist,
                                       vec[2]/dist])
                    bond_types.append(bond_curr)

        bond_array = np.array(bond_array)
        bond_types = np.array(bond_types)
        return bond_array, bond_types, environment_types, central_type


# testing ground (will be moved to test suite later)
if __name__ == '__main__':
    # create simple test environment
    positions = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    species = ['B', 'A']
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001
    atom = 0

    test_env = TwoBodyEnvironment(positions, species, cell, cutoff, atom)

    # test species_to_bond
    assert(test_env.bond_list == [['B', 'B'], ['B', 'A'], ['A', 'A']])

    # test is_bond (static method)
    assert(TwoBodyEnvironment.is_bond('A', 'B', ['A', 'B']))
    assert(TwoBodyEnvironment.is_bond('B', 'A', ['A', 'B']))
    assert(not TwoBodyEnvironment.is_bond('C', 'A', ['A', 'B']))

    # test species_to_index
    assert(test_env.species_to_index('B', 'B') == 0)
    assert(test_env.species_to_index('B', 'A') == 1)
    assert(test_env.species_to_index('A', 'A') == 2)

    # test get_local_atom_images
    vec = np.array([0.5, 0.5, 0.5])
    vecs, dists = test_env.get_local_atom_images(vec)
    assert(len(dists) == 8)
    assert(len(vecs) == 8)

    # test get_atoms_within_cutoff
    atom = 0
    bond_array, bonds, environment_types, central_type =\
        test_env.get_atoms_within_cutoff(atom)

    assert(bond_array.shape[0] == 8)
