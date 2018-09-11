from numpy import zeros, ndarray
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
                 mass_dict: dict = None):
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
        self.prev_positions = list(positions)
        self.forces = [zeros(3) for _ in positions]
        self.stds = [zeros(3) for _ in positions]

        self.cutoff = cutoff

        self.mass_dict = mass_dict

    def translate_positions(self, vector: ndarray=zeros(3)):
        """
        Translate all positions, and previous positions by vector
        :param vector: vector to translate by
        :return:
        """
        for pos in self.positions:
            pos += vector

        for prev_pos in self.prev_positions:
            prev_pos += vector


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
