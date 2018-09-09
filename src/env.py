"""
Jon V
"""
import numpy as np
from math import exp
from numba import njit
from struc import Structure
import time


# get three body kernel between two environments
def three_body(env1, env2, d1, d2, sig, ls):
    return ChemicalEnvironment.three_body_jit(env1.bond_array,
                                              env1.bond_types,
                                              env1.cross_bond_dists,
                                              env1.cross_bond_types,
                                              env2.bond_array,
                                              env2.bond_types,
                                              env2.cross_bond_dists,
                                              env2.cross_bond_types,
                                              d1, d2, sig, ls)


def three_body_py(env1, env2, d1, d2, sig, ls):
    return ChemicalEnvironment.three_body_nojit(env1.bond_array,
                                                env1.bond_types,
                                                env1.cross_bond_dists,
                                                env1.cross_bond_types,
                                                env2.bond_array,
                                                env2.bond_types,
                                                env2.cross_bond_dists,
                                                env2.cross_bond_types,
                                                d1, d2, sig, ls)


# get two body kernel between two environments
def two_body(env1, env2, d1, d2, sig, ls):
    return ChemicalEnvironment.two_body_jit(env1.bond_array,
                                            env1.bond_types,
                                            env2.bond_array,
                                            env2.bond_types,
                                            d1, d2, sig, ls)


def two_body_py(env1, env2, d1, d2, sig, ls):
    return ChemicalEnvironment.two_body_nojit(env1.bond_array,
                                              env1.bond_types,
                                              env2.bond_array,
                                              env2.bond_types,
                                              d1, d2, sig, ls)


class ChemicalEnvironment:

    def __init__(self, structure, atom):
        self.structure = structure

        bond_array, bond_types, bond_positions, etyps, ctyp =\
            self.get_atoms_within_cutoff(atom)

        self.bond_array = bond_array
        self.bond_types = bond_types
        self.bond_positions = bond_positions
        self.etyps = etyps
        self.ctyp = ctyp

        cross_bond_dists, cross_bond_types = self.get_cross_bonds()
        self.cross_bond_dists = cross_bond_dists
        self.cross_bond_types = cross_bond_types

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

    @staticmethod
    def is_triplet(species1, species2, species3, triplet):
        return [species1, species2, species3] == triplet

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

        for bond_index, bond in enumerate(self.structure.bond_list):
            if ChemicalEnvironment.is_bond(species1, species2, bond):
                return bond_index

    def triplet_to_index(self, species1, species2, species3):
        for triplet_index, triplet in enumerate(self.structure.triplet_list):
            if ChemicalEnvironment.is_triplet(species1, species2, species3,
                                              triplet):
                return triplet_index

    # TODO: make this static
    def get_local_atom_images(self, vec, super_check=3):
        """Get periodic images of an atom within the cutoff radius.

        :param vec: atomic position
        :type vec: nparray of shape (3,)
        :return: vectors and distances of atoms within cutoff radius
        :rtype: list of nparrays, list of floats
        """

        # get bravais coefficients
        coeff = np.matmul(self.structure.inv_lattice, vec)

        # get bravais coefficients for atoms within supercell
        coeffs = [[], [], []]
        for n in range(3):
            for m in range(super_check):
                if m == 0:
                    coeffs[n].append(coeff[n])
                else:
                    coeffs[n].append(coeff[n]-m)
                    coeffs[n].append(coeff[n]+m)

        # get vectors within cutoff
        vecs = []
        dists = []
        for m in range(len(coeffs[0])):
            for n in range(len(coeffs[1])):
                for p in range(len(coeffs[2])):
                    vec_curr = coeffs[0][m]*self.structure.vec1 +\
                               coeffs[1][n]*self.structure.vec2 +\
                               coeffs[2][p]*self.structure.vec3
                    dist = np.linalg.norm(vec_curr)

                    if dist < self.structure.cutoff:
                        vecs.append(vec_curr)
                        dists.append(dist)

        return vecs, dists

    # return information about atoms inside cutoff region
    def get_atoms_within_cutoff(self, atom):

        pos_atom = self.structure.positions[atom]  # position of central atom
        central_type = self.structure.species[atom]  # type of central atom

        bond_array = []
        bond_types = []
        bond_positions = []
        environment_types = []

        # find all atoms and images in the neighborhood
        for n in range(len(self.structure.positions)):
            diff_curr = self.structure.positions[n] - pos_atom
            typ_curr = self.structure.species[n]
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
                    bond_positions.append([vec[0], vec[1], vec[2]])

        bond_array = np.array(bond_array)
        bond_types = np.array(bond_types)
        bond_positions = np.array(bond_positions)
        return bond_array, bond_types, bond_positions, environment_types,\
            central_type

    # return information about cross bonds
    def get_cross_bonds(self):
        nat = len(self.etyps)
        cross_bond_dists = np.zeros([nat, nat])
        cross_bond_types = np.zeros([nat, nat])

        ctyp = self.ctyp

        for m in range(nat):
            pos1 = self.bond_positions[m]
            etyp1 = self.etyps[m]
            for n in range(nat):
                pos2 = self.bond_positions[n]
                etyp2 = self.etyps[n]

                dist_curr = np.linalg.norm(pos1 - pos2)
                trip_ind = self.triplet_to_index(ctyp, etyp1, etyp2)

                cross_bond_dists[m, n] = dist_curr
                cross_bond_types[m, n] = trip_ind
        return cross_bond_dists, cross_bond_types

    # jit function that computes three body kernel
    @staticmethod
    @njit
    def three_body_jit(bond_array_1, bond_types_1,
                       cross_bond_dists_1, cross_bond_types_1,
                       bond_array_2, bond_types_2,
                       cross_bond_dists_2, cross_bond_types_2,
                       d1, d2, sig, ls):
        d = sig*sig/(ls*ls*ls*ls)
        e = ls*ls
        f = 1/(2*ls*ls)
        kern = 0

        x1_len = len(bond_types_1)
        x2_len = len(bond_types_2)

        # loop over triplets in environment 1
        for m in range(x1_len):
            ri1 = bond_array_1[m, 0]
            ci1 = bond_array_1[m, d1]

            for n in range(m+1, x1_len):
                ri2 = bond_array_1[n, 0]
                ci2 = bond_array_1[n, d1]
                ri3 = cross_bond_dists_1[m, n]
                t1 = cross_bond_types_1[m, n]

                # loop over triplets in environment 2
                for p in range(x2_len):
                    rj1 = bond_array_2[p, 0]
                    cj1 = bond_array_2[p, d2]

                    for q in range(x2_len):
                        if p == q:  # consider distinct bonds
                            continue

                        # get triplet types
                        t2 = cross_bond_types_2[p, q]

                        # proceed if triplet types match
                        if t1 == t2:
                            rj2 = bond_array_2[q, 0]
                            cj2 = bond_array_2[q, d2]
                            rj3 = cross_bond_dists_2[p, q]

                            r11 = ri1-rj1
                            r22 = ri2-rj2
                            r33 = ri3-rj3

                            # add to kernel
                            kern += (e * (ci1 * cj1 + ci2 * cj2) -
                                     (r11 * ci1 + r22 * ci2) *
                                     (r11 * cj1 + r22 * cj2)) *\
                                d * exp(-f * (r11 * r11 + r22 * r22 +
                                              r33 * r33))

        return kern

    # python version of three body kernel for testing purposes
    @staticmethod
    def three_body_nojit(bond_array_1, bond_types_1,
                         cross_bond_dists_1, cross_bond_types_1,
                         bond_array_2, bond_types_2,
                         cross_bond_dists_2, cross_bond_types_2,
                         d1, d2, sig, ls):
        d = sig*sig/(ls*ls*ls*ls)
        e = ls*ls
        f = 1/(2*ls*ls)
        kern = 0

        x1_len = len(bond_types_1)
        x2_len = len(bond_types_2)

        # loop over triplets in environment 1
        for m in range(x1_len):
            ri1 = bond_array_1[m, 0]
            ci1 = bond_array_1[m, d1]

            for n in range(m+1, x1_len):
                ri2 = bond_array_1[n, 0]
                ci2 = bond_array_1[n, d1]
                ri3 = cross_bond_dists_1[m, n]
                t1 = cross_bond_types_1[m, n]

                # loop over triplets in environment 2
                for p in range(x2_len):
                    rj1 = bond_array_2[p, 0]
                    cj1 = bond_array_2[p, d2]

                    for q in range(x2_len):
                        if p == q:  # consider distinct bonds
                            continue

                        # get triplet types
                        t2 = cross_bond_types_2[p, q]

                        # proceed if triplet types match
                        if t1 == t2:
                            rj2 = bond_array_2[q, 0]
                            cj2 = bond_array_2[q, d2]
                            rj3 = cross_bond_dists_2[p, q]

                            r11 = ri1-rj1
                            r22 = ri2-rj2
                            r33 = ri3-rj3

                            # add to kernel
                            kern += (e * (ci1 * cj1 + ci2 * cj2) -
                                     (r11 * ci1 + r22 * ci2) *
                                     (r11 * cj1 + r22 * cj2)) *\
                                d * exp(-f * (r11 * r11 + r22 * r22 +
                                              r33 * r33))

        return kern

    # jit function that computes two body kernel
    @staticmethod
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

    # for testing purposes, define python version of two body kernel
    @staticmethod
    def two_body_nojit(bond_array_1, bond_types_1, bond_array_2,
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


if __name__ == '__main__':
    pass
