from numpy import zeros
from numpy.linalg import inv
import numpy as np


class Structure(object):
    """
        Contains positions, species, cell, cutoff rad, previous positions,
        forces, and stds of forces, computes inv_cell and bond list

        :param lattice: nparray, 3x3 Bravais Lattice
        :param species: list[str], List of elements
        :param positions: list[nparray] list of positions
        :param cutoff: float, Cutoff radius for GP

    """

    def __init__(self, lattice, species, positions, cutoff):
        self.lattice = lattice
        self.vec1 = lattice[0, :]
        self.vec2 = lattice[1, :]
        self.vec3 = lattice[2, :]

        self.species = species
        self.elements = list(set(species))
        self.bond_list = self.calc_bond_list(self.elements)

        self.inv_lattice = inv(lattice)

        self.positions = positions
        self.prev_positions = list(positions)
        self.forces = [zeros(3) for pos in positions]
        self.stds = [zeros(3) for pos in positions]

        self.cutoff = cutoff

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

if __name__ == '__main__':
    # create simple structure
    lattice = np.eye(3)
    species = ['B', 'A']
    positions = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    test_structure = Structure(lattice, species, positions, cutoff)

    # test species_to_bond
    print(test_structure.bond_list)
    assert(test_structure.bond_list == [['B', 'B'], ['B', 'A'], ['A', 'A']])
