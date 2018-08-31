from numpy import zeros
from numpy.linalg import inv


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
        self.elements = set(species)
        self.bond_list = self.calc_bond_list(self.elements)

        self.inv_lattice = inv(lattice)

        self.positions = positions
        self.prev_positions = list(positions)
        self.forces = [zeros(3) for pos in positions]
        self.stds = [zeros(3) for pos in positions]

        self.cutoff = cutoff

    def calc_bond_list(self, elements):
        bond_list = []
        pass
        return bond_list
