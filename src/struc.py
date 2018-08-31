from numpy import zeros
from numpy.linalg import inv

class Structure(object):
    """
        Contains positions, species, cell, cutoff rad, previous positions,
        forces, and stds of forces, computes inv_cell an

    """

    def __init__(self, alat,lattice,species,positions,cutoff):

        self.alat = alat
        self.lattice = lattice

        self.species = (species)
        self.elements = set(species)

        self.inv_lattice = inv(lattice)


        self.positions=positions
        self.prev_positions = list(positions)
        self.forces = [np.zeros(3) for pos in positions]
        self.stds = [np.zeros(3) for pos in positions]


