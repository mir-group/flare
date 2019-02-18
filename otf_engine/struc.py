from numpy import zeros, ndarray, dot, isclose, mod, ones, equal, matmul, \
    copy, arccos, array, arange
from numpy import abs as npabs
from numpy.linalg import inv
from numpy.random import uniform, normal
from typing import List
from math import sin, cos


class Structure(object):
    """
        Contains positions, species, cell, cutoff rad, previous positions,
        forces, and stds of forces, computes inv_cell and bond list

        :param cell: nparray, 3x3 Bravais cell
        :param species: list[str], List of elements
        :param positions: list[nparray] list of positions
        :param cutoff: float, Cutoff radius for GP

    """

    def __init__(self, cell: ndarray, species: List[str],
                 positions: ndarray, mass_dict: dict = None,
                 prev_positions: ndarray=None):
        self.cell = cell
        self.vec1 = cell[0, :]
        self.vec2 = cell[1, :]
        self.vec3 = cell[2, :]

        self.species = species
        self.nat = len(species)

        # get unique species
        unique_species, coded_species = self.get_unique_species(species)

        self.unique_species = unique_species
        self.coded_species = coded_species
        self.nos = len(unique_species)

        self.inv_cell = inv(cell)

        self.positions = array(positions)

        # Default: atoms have no velocity
        if prev_positions is None:
            self.prev_positions = copy(self.positions)
        else:
            assert len(positions) == len(prev_positions), 'Previous ' \
                                                          'positions and ' \
                                            'positions are not same length'
            self.prev_positions = prev_positions

        self.forces = zeros((3, len(positions)))
        self.stds = zeros((3, len(positions)))
        self.mass_dict = mass_dict
        self.dft_forces = False

    @staticmethod
    def get_unique_species(species):
        unique_species = []
        coded_species = []
        for spec in species:
            if spec in unique_species:
                coded_species.append(unique_species.index(spec))
            else:
                coded_species.append(len(unique_species))
                unique_species.append(spec)

        return unique_species, coded_species


if __name__ == '__main__':
    pass
