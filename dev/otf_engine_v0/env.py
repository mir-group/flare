import numpy as np
from math import exp
from numba import njit
from struc import Structure
import time


class ChemicalEnvironment:

    def __init__(self, structure, atom):
        self.structure = structure

        bond_array, bond_positions, etyps, ctyp = \
            self.get_atoms_within_cutoff(self.structure, atom)

        self.bond_array = bond_array
        self.bond_positions = bond_positions
        self.etyps = etyps
        self.ctyp = ctyp

        self.sort_arrays()

        cross_bond_dists = self.get_cross_bonds()
        self.cross_bond_dists = cross_bond_dists

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

    @staticmethod
    def species_to_index(structure, species1, species2):
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

        for bond_index, bond in enumerate(structure.bond_list):
            if ChemicalEnvironment.is_bond(species1, species2, bond):
                return bond_index

    def triplet_to_index(self, species1, species2, species3):
        for triplet_index, triplet in enumerate(self.structure.triplet_list):
            if ChemicalEnvironment.is_triplet(species1, species2, species3,
                                              triplet):
                return triplet_index

    @staticmethod
    def get_local_atom_images(structure, vec, super_check=1):
        """Get periodic images of an atom within the cutoff radius.

        :param structure: structure defined lattice vectors
        :type structure: Structure
        :param vec: atomic position
        :type vec: nparray of shape (3,)
        :return: vectors and distances of atoms within cutoff radius
        :rtype: list of nparrays, list of floats
        """

        vecs = []
        dists = []

        images = structure.get_periodic_images(vec, super_check)

        for image in images:
            dist = np.linalg.norm(image)

            if dist < structure.cutoff:
                vecs.append(image)
                dists.append(dist)

        return vecs, dists

    # return information about atoms inside cutoff region
    @staticmethod
    def get_atoms_within_cutoff(structure: Structure, atom: int,
                                super_check: int = 1):

        pos_atom = structure.positions[atom]  # position of central atom
        central_type = structure.coded_species[atom]  # type of central atom

        bond_array = []
        bond_positions = []
        environment_types = []

        # find all atoms and images in the neighborhood
        for n in range(len(structure.positions)):
            diff_curr = structure.positions[n] - pos_atom
            typ_curr = structure.coded_species[n]

            # get images within cutoff
            vecs, dists = \
                ChemicalEnvironment.get_local_atom_images(structure,
                                                          diff_curr,
                                                          super_check)

            for vec, dist in zip(vecs, dists):
                # ignore self interaction
                if dist != 0:
                    environment_types.append(typ_curr)
                    bond_array.append([dist, vec[0] / dist, vec[1] / dist,
                                       vec[2] / dist])
                    bond_positions.append([vec[0], vec[1], vec[2]])

        bond_array = np.array(bond_array)
        bond_positions = np.array(bond_positions)
        environment_types = np.array(environment_types)

        return bond_array, bond_positions, environment_types, central_type

    # return information about cross bonds
    def get_cross_bonds(self):
        nat = len(self.etyps)
        cross_bond_dists = np.zeros([nat, nat])

        for m in range(nat):
            pos1 = self.bond_positions[m]
            for n in range(nat):
                pos2 = self.bond_positions[n]
                dist_curr = np.linalg.norm(pos1 - pos2)
                cross_bond_dists[m, n] = dist_curr
        return cross_bond_dists

    def sort_arrays(self):
        sort_inds = self.bond_array[:, 0].argsort()
        self.bond_array = self.bond_array[sort_inds]
        self.bond_positions = self.bond_positions[sort_inds]
        self.etyps = [self.etyps[n] for n in sort_inds]

if __name__ == '__main__':
    pass
