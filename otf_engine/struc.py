""""
OTF engine

Steven Torrisi, Jon Vandermause
"""

from numpy import zeros, ndarray, dot, isclose, mod, ones, equal, matmul, copy
from numpy.linalg import inv
from typing import List


class Structure(object):
    """
        Contains positions, species, cell, cutoff rad, previous positions,
        forces, and stds of forces, computes inv_cell and bond list

        :param lattice: nparray, 3x3 Bravais Lattice
        :param species: list[str], List of elements
        :param positions: list[nparray] list of positions
        :param cutoff: float, Cutoff radius for GP

    """

    def __init__(self, lattice: ndarray, species: List[str],
                 positions: List[ndarray], cutoff: float,
                 mass_dict: dict = None, prev_positions : List[ndarray]=None):
        self.lattice = lattice
        self.vec1 = lattice[0, :]
        self.vec2 = lattice[1, :]
        self.vec3 = lattice[2, :]

        self.species = species
        self.nat = len(species)

        # get unique species
        unique_species = []
        for spec in species:
            if spec not in unique_species:
                unique_species.append(spec)

        self.unique_species = unique_species
        self.bond_list = self.calc_bond_list(self.unique_species)
        self.triplet_list = self.calc_triplet_list(self.unique_species)

        self.inv_lattice = inv(lattice)

        self.positions = positions

        # Default: atoms have no velocity
        if prev_positions is None:
            self.prev_positions = [copy(pos) for pos in self.positions]
        else:
            assert len(positions) == len(prev_pos), 'Previous positions and ' \
                                            'positions are not same length'
            self.prev_positions = prev_positions

        self.forces = [zeros(3) for _ in positions]
        self.stds = [zeros(3) for _ in positions]

        self.cutoff = cutoff

        self.mass_dict = mass_dict

    def translate_positions(self, vector: ndarray = zeros(3)):
        """
        Translate all positions, and previous positions by vector
        :param vector: vector to translate by
        :return:
        """
        for pos in self.positions:
            pos += vector

        for prev_pos in self.prev_positions:
            prev_pos += vector

    def get_periodic_images(self, vec, super_check: int = 2):
        """
        Given vec, find the periodic images of it out to super_check
        neighbors

        :param vec:
        :param super_check:
        :return:
        """
        coeff = matmul(self.inv_lattice, vec)

        # get bravais coefficients for atoms within supercell
        coeffs = [[], [], []]
        for n in range(3):
            for m in range(super_check):
                if m == 0:
                    coeffs[n].append(coeff[n])
                else:
                    coeffs[n].append(coeff[n] - m)
                    coeffs[n].append(coeff[n] + m)

        # get vectors within cutoff
        images = []
        for m in range(len(coeffs[0])):
            for n in range(len(coeffs[1])):
                for p in range(len(coeffs[2])):
                    curr_image = coeffs[0][m] * self.vec1 + \
                                 coeffs[1][n] * self.vec2 + \
                                 coeffs[2][p] * self.vec3
                    images.append(curr_image)

        return images

    def get_index_from_position(self, position):
        """
        Gets the index of an atom from a position, folding back into the
        unit cell
        :param position: Atom to get position of
        :param fold: Attempt to find the index of the 'original' atom in a
        unit cell corresponding to a periodic image of position
        :return:
        """

        for i in range(self.nat):
            if isclose(position, self.positions[i], atol=1e-6).all():
                return i

        raise Exception("Position does not correspond to atom in structure")

    @staticmethod
    def calc_bond_list(unique_species):
        """Converts unique species to a list of bonds.

        :param unique_species: unique species in the simulation
        :type unique_species: list of strings
        :return: all possible bonds
        :rtype: list of lists of two strings
        """

        # create bond list
        bond_list = []
        for m in range(len(unique_species)):
            species_1 = unique_species[m]

            for n in range(m, len(unique_species)):
                species_2 = unique_species[n]
                bond_list.append([species_1, species_2])

        return bond_list

    @staticmethod
    # given list of unique species, return list of possible triplets
    def calc_triplet_list(unique_species):
        triplet_list = []
        for m in unique_species:
            for n in unique_species:
                for p in unique_species:
                    triplet_list.append([m, n, p])
        return triplet_list


if __name__ == '__main__':
    # create simple structure
    pass
