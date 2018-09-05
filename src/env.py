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
            for n in range(x1_len):
                if m == n:  # consider distinct bonds
                    continue
                # loop over triplets in environment 2
                for p in range(x2_len):
                    for q in range(x2_len):
                        if p == q:  # consider distinct bonds
                            continue

                        # get triplet types
                        t1 = cross_bond_types_1[m, n]
                        t2 = cross_bond_types_2[p, q]

                        # proceed if triplet types match
                        if t1 == t2:
                            # get triplet 1 details
                            ri1 = bond_array_1[m, 0]
                            ci1 = bond_array_1[m, d1]
                            ri2 = bond_array_1[n, 0]
                            ci2 = bond_array_1[n, d1]
                            ri3 = cross_bond_dists_1[m, n]

                            # get triplet 2 details
                            rj1 = bond_array_2[p, 0]
                            cj1 = bond_array_2[p, d2]
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
            for n in range(x1_len):
                if m == n:  # consider distinct bonds
                    continue
                # loop over triplets in environment 2
                for p in range(x2_len):
                    for q in range(x2_len):
                        if p == q:  # consider distinct bonds
                            continue

                        # get triplet types
                        t1 = cross_bond_types_1[m, n]
                        t2 = cross_bond_types_2[p, q]

                        # proceed if triplet types match
                        if t1 == t2:
                            # get triplet 1 details
                            ri1 = bond_array_1[m, 0]
                            ci1 = bond_array_1[m, d1]
                            ri2 = bond_array_1[n, 0]
                            ci2 = bond_array_1[n, d1]
                            ri3 = cross_bond_dists_1[m, n]

                            # get triplet 2 details
                            rj1 = bond_array_2[p, 0]
                            cj1 = bond_array_2[p, d2]
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


# testing ground (will be moved to test suite later)
if __name__ == '__main__':
    # create test structure
    positions = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    species = ['B', 'A']
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001
    test_structure = Structure(cell, species, positions, cutoff)

    # create environment
    atom = 0
    test_env = ChemicalEnvironment(test_structure, atom)

    # test species_to_bond
    is_bl_right = test_env.structure.bond_list ==\
        [['B', 'B'], ['B', 'A'], ['A', 'A']]
    assert(is_bl_right)

    # test is_bond (static method)
    assert(ChemicalEnvironment.is_bond('A', 'B', ['A', 'B']))
    assert(ChemicalEnvironment.is_bond('B', 'A', ['A', 'B']))
    assert(not ChemicalEnvironment.is_bond('C', 'A', ['A', 'B']))

    # test is_triplet (static method)
    assert(ChemicalEnvironment.is_triplet('A', 'B', 'C', ['A', 'B', 'C']))
    assert(ChemicalEnvironment.is_triplet('A', 'B', 'B', ['A', 'B', 'B']))
    assert(not ChemicalEnvironment.is_triplet('C', 'B', 'B', ['A', 'B', 'B']))

    # test species_to_index
    assert(test_env.species_to_index('B', 'B') == 0)
    assert(test_env.species_to_index('B', 'A') == 1)
    assert(test_env.species_to_index('A', 'A') == 2)

    # test triplet_to_index
    assert(test_structure.triplet_list.index(['A', 'B', 'B']) ==
           test_env.triplet_to_index('A', 'B', 'B'))

    # test get_local_atom_images
    vec = np.array([0.5, 0.5, 0.5])
    vecs, dists = test_env.get_local_atom_images(vec)
    assert(len(dists) == 8)
    assert(len(vecs) == 8)

    # test get_atoms_within_cutoff
    atom = 0
    bond_array, bonds, bond_positions, environment_types, central_type =\
        test_env.get_atoms_within_cutoff(atom)

    assert(bond_array.shape[0] == 8)
    assert(bond_array[0, 1] == bond_positions[0, 0] / bond_array[0, 0])

    # test get_cross_bonds
    nat = len(test_env.etyps)
    mrand = np.random.randint(0, nat)
    nrand = np.random.randint(0, nat)
    pos1 = test_env.bond_positions[mrand]
    pos2 = test_env.bond_positions[nrand]
    assert(test_env.cross_bond_dists[mrand, nrand] ==
           np.linalg.norm(pos1-pos2))

    # test jit and python two body kernels
    def kernel_performance(env1, env2, d1, d2, sig, ls, kernel, its):
        # warm up jit
        time0 = time.time()
        kern_val = kernel(env1, env2, d1, d2, sig, ls)
        time1 = time.time()
        warm_up_time = time1 - time0

        # test run time performance
        time2 = time.time()
        for n in range(its):
            kernel(env1, env2, d1, d2, sig, ls)
        time3 = time.time()
        run_time = (time3 - time2) / its

        return kern_val, run_time, warm_up_time

    def get_jit_speedup(env1, env2, d1, d2, sig, ls, jit_kern, py_kern,
                        its):

        kern_val_jit, run_time_jit, warm_up_time_jit = \
            kernel_performance(env1, env2, d1, d2, sig, ls, jit_kern, its)

        kern_val_py, run_time_py, warm_up_time_py = \
            kernel_performance(env1, env2, d1, d2, sig, ls, py_kern, its)

        speed_up = run_time_py / run_time_jit

        return speed_up, kern_val_jit, kern_val_py, warm_up_time_jit,\
            warm_up_time_py

    # set up two test environments
    positions_1 = [np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])]
    species_1 = ['B', 'A']
    atom_1 = 0
    test_structure_1 = Structure(cell, species_1, positions_1, cutoff)
    env1 = ChemicalEnvironment(test_structure_1, atom_1)

    positions_2 = [np.array([0, 0, 0]), np.array([0.25, 0.3, 0.4])]
    species_2 = ['B', 'A']
    atom_2 = 0
    test_structure_2 = Structure(cell, species_2, positions_2, cutoff)
    env2 = ChemicalEnvironment(test_structure_2, atom_2)

    d1 = 1
    d2 = 1
    sig = 1
    ls = 1

    its = 10

    # compare performance
    speed_up, kern_val_jit, kern_val_py, warm_up_time_jit, warm_up_time_py = \
        get_jit_speedup(env1, env2, d1, d2, sig, ls,
                        two_body, two_body_py, its)

    assert(kern_val_jit == kern_val_py)
    assert(speed_up > 1)

    # test three body function
    def get_random_structure(cell, unique_species, cutoff, noa):
        positions = []
        forces = []
        species = []
        for n in range(noa):
            positions.append(np.random.uniform(-1, 1, 3))
            forces.append(np.random.uniform(-1, 1, 3))
            species.append(unique_species[np.random.randint(0, 2)])

        test_structure = Structure(cell, species, positions, cutoff)

        return test_structure, forces

    cell = np.eye(3)
    unique_species = ['B', 'A']
    cutoff = 0.8
    noa = 15

    # create two test environments
    test_structure_1, _ = \
        get_random_structure(cell, unique_species, cutoff, noa)

    test_structure_2, _ = \
        get_random_structure(cell, unique_species, cutoff, noa)

    test_env_1 = ChemicalEnvironment(test_structure_1, 0)
    test_env_2 = ChemicalEnvironment(test_structure_2, 0)

    print('there are %i atoms in env1.' % len(test_env_1.bond_types))
    print('there are %i atoms in env2.' % len(test_env_2.bond_types))

    two_body(test_env_1, test_env_2, d1, d2, sig, ls)
    three_body(test_env_1, test_env_2, d1, d2, sig, ls)

    its = 10

    two_speed_up, kern_val_jit, kern_val_py, warm_up_time_jit,\
        warm_up_time_py =\
        get_jit_speedup(test_env_1, test_env_2, d1, d2, sig, ls, two_body,
                        two_body_py, its)

    three_speed_up, kern_val_jit, kern_val_py, warm_up_time_jit,\
        warm_up_time_py =\
        get_jit_speedup(test_env_1, test_env_2, d1, d2, sig, ls, three_body,
                        three_body_py, its)

    print('two body speed up is %.3f' % two_speed_up)
    print('three body speed up is %.3f' % three_speed_up)
