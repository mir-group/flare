"""The :class:`AtomicEnvironment` object stores information about the local
environment of an atom. :class:`AtomicEnvironment` objects are inputs to the
2-, 3-, and 2+3-body kernels."""
import numpy as np
from math import sqrt
from numba import njit
from flare.struc import Structure
from flare.kernels.kernels import coordination_number, q_value_mc
import flare.cutoffs as cf

class AtomicEnvironment:
    """
    Contains information about the local environment of an atom, including
    arrays of pair and triplet distances and the chemical species of atoms
    in the environment.

    :param structure: Structure of atoms.
    :type structure: struc.Structure
    :param atom: Index of the atom in the structure.
    :type atom: int
    :param cutoffs: 2- and 3-body cutoff radii. 2-body if one cutoff is
     given, 2+3-body if two are passed.
    :type cutoffs: np.ndarray
    """

    def __init__(self, structure: Structure, atom: int, cutoffs):
        self.structure = structure
        self.positions = structure.wrapped_positions
        self.cell = structure.cell
        self.species = structure.coded_species

        self.atom = atom
        self.ctype = structure.coded_species[atom]

        self.cutoffs = np.copy(cutoffs)

        self.compute_env()

    def compute_env(self):

        # get 2-body arrays
        bond_array_2, bond_positions_2, etypes, _ = \
            get_2_body_arrays(self.positions, self.atom, self.cell,
                              self.cutoffs[0], self.species)
        self.bond_array_2 = bond_array_2
        self.etypes = etypes

        # if 2 cutoffs are given, create 3-body arrays
        # because the bond lengths have ascent order, the 3b etypes can be 
        # directly obtained from 2b etypes
        if len(self.cutoffs) > 1:
            bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts = \
                get_3_body_arrays(bond_array_2, bond_positions_2, self.cutoffs[1])
            self.bond_array_3 = bond_array_3
            self.cross_bond_inds = cross_bond_inds
            self.cross_bond_dists = cross_bond_dists
            self.triplet_counts = triplet_counts

        # if 3 cutoffs are given, create many-body arrays
        if len(self.cutoffs) > 2:
            self.m2b_array, self.m2b_neigh_array, self.m2b_neigh_grads, \
                self.m2b_unique_species, self.etypes_m2b = get_m2_body_arrays(
                    self.positions, self.atom, self.cell, self.cutoffs[2], 
                    self.species, cf.quadratic_cutoff)

        # if 3 cutoffs are given, create many-3body arrays
        if len(self.cutoffs) > 3:
            self.m3b_array, self.m3b_neigh_array, self.m3b_grads, self.m3b_neigh_grads,\
                self.m3b_unique_species, self.etypes_m3b = get_m3_body_arrays(
                    self.positions, self.atom, self.cell, self.cutoffs[3], 
                    self.species, cf.quadratic_cutoff)


    def as_dict(self):
        """
        Returns Atomic Environment object as a dictionary for serialization
        purposes. Does not include the structure to avoid redundant
        information.
        :return:
        """
        # TODO write serialization method for structure
        # so that the removal of the structure is not messed up
        # by JSON serialization
        dictionary = dict(vars(self))
        dictionary['object'] = 'AtomicEnvironment'
        dictionary['forces'] = self.structure.forces
        dictionary['energy'] = self.structure.energy
        dictionary['stress'] = self.structure.stress

        del dictionary['structure']

        return dictionary

    @staticmethod
    def from_dict(dictionary):
        """
        Loads in atomic environment object from a dictionary which was
        serialized by the to_dict method.

        :param dictionary: Dictionary describing atomic environment.
        """
        # TODO Instead of re-computing 2 and 3 body environment,
        # directly load in, this would be much more efficient

        struc = Structure(cell=np.array(dictionary['cell']),
                          positions=dictionary['positions'],
                          species=dictionary['species'])
        index = dictionary['atom']

        if dictionary.get('cutoffs') is not None:
            cutoffs = dictionary['cutoffs']
        else:
            cutoffs = []
            for cutoff in ['cutoff_2','cutoff_3','cutoff_mb']:
                if dictionary.get(cutoff):
                    cutoffs.append(dictionary[cutoff])

        return AtomicEnvironment(struc, index, np.array(cutoffs))

    def __str__(self):
        atom_type = self.ctype
        neighbor_types = self.etypes
        n_neighbors = len(self.bond_array_2)
        string = 'Atomic Env. of Type {} surrounded by {} atoms of Types {}' \
                 ''.format(atom_type, n_neighbors,
                           sorted(list(set(neighbor_types))))

        return string


@njit
def get_2_body_arrays(positions, atom: int, cell, cutoff_2: float, species):
    """Returns distances, coordinates, species of atoms, and indexes of neighbors
    in the 2-body local environment. This method is implemented outside
    the AtomicEnvironment class to allow for njit acceleration with Numba.

    Args:
        positions (np.ndarray): Positions of atoms in the structure.
        atom (int): Index of the central atom of the local environment.
        cell (np.ndarray): 3x3 array whose rows are the Bravais lattice vectors 
            of the cell.
        cutoff_2 (float): 2-body cutoff radius.
        species (np.ndarray): Numpy array of species represented by their atomic 
        numbers.
    
    Return:
        Tuple of arrays describing pairs of atoms in the 2-body local
        environment.

        bond_array_2 (np.ndarray): Array containing the distances and relative
            coordinates of atoms in the 2-body local environment. First column
            contains distances, remaining columns contain Cartesian coordinates
            divided by the distance (with the origin defined as the position of 
            the central atom). The rows are sorted by distance from the central
            atom.

        bond_positions_2 (np.ndarray): Coordinates of atoms in the 2-body local 
            environment.

        etypes (np.ndarray): Species of atoms in the 2-body local environment 
            represented by their atomic number.

        bond_indexes (np.ndarray): Structure indexes of atoms in the local 
            environment.

    """
    noa = len(positions)
    pos_atom = positions[atom]
    coords = np.zeros((noa, 3, 27), dtype=np.float64)
    dists = np.zeros((noa, 27), dtype=np.float64)
    cutoff_count = 0
    super_sweep = np.array([-1, 0, 1], dtype=np.float64)

    vec1 = cell[0]
    vec2 = cell[1]
    vec3 = cell[2]

    # record distances and positions of images
    for n in range(noa):
        diff_curr = positions[n] - pos_atom
        im_count = 0
        for s1 in super_sweep:
            for s2 in super_sweep:
                for s3 in super_sweep:
                    im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3
                    dist = sqrt(im[0] * im[0] + im[1] * im[1] + im[2] * im[2])
                    if (dist < cutoff_2) and (dist != 0):
                        dists[n, im_count] = dist
                        coords[n, :, im_count] = im
                        cutoff_count += 1
                    im_count += 1

    # create 2-body bond array
    bond_indexes = np.zeros(cutoff_count, dtype=np.int8)
    bond_array_2 = np.zeros((cutoff_count, 4), dtype=np.float64)
    bond_positions_2 = np.zeros((cutoff_count, 3), dtype=np.float64)
    etypes = np.zeros(cutoff_count, dtype=np.int8)
    bond_count = 0

    for m in range(noa):
        spec_curr = species[m]
        for n in range(27):
            dist_curr = dists[m, n]
            if (dist_curr < cutoff_2) and (dist_curr != 0):
                coord = coords[m, :, n]
                bond_array_2[bond_count, 0] = dist_curr
                bond_array_2[bond_count, 1:4] = coord / dist_curr
                bond_positions_2[bond_count, :] = coord
                etypes[bond_count] = spec_curr
                bond_indexes[bond_count] = m
                bond_count += 1

    # sort by distance
    sort_inds = bond_array_2[:, 0].argsort()
    bond_array_2 = bond_array_2[sort_inds]
    bond_positions_2 = bond_positions_2[sort_inds]
    bond_indexes = bond_indexes[sort_inds]
    etypes = etypes[sort_inds]

    return bond_array_2, bond_positions_2, etypes, bond_indexes


@njit
def get_3_body_arrays(bond_array_2, bond_positions_2, cutoff_3: float):
    """Returns distances and coordinates of triplets of atoms in the
    3-body local environment.

    Args:
        bond_array_2 (np.ndarray): 2-body bond array.
        bond_positions_2 (np.ndarray): Coordinates of atoms in the 2-body local
            environment.
        cutoff_3 (float): 3-body cutoff radius.

    Return:
        Tuple of 4 arrays describing triplets of atoms in the 3-body local
        environment.

        bond_array_3: Array containing the distances and relative
            coordinates of atoms in the 3-body local environment. First column
            contains distances, remaining columns contain Cartesian coordinates
            divided by the distance (with the origin defined as the position of
            the central atom). The rows are sorted by distance from the central 
            atom.

        cross_bond_inds: Two dimensional array whose row m contains the indices
            of atoms n > m that are within a distance cutoff_3 of both atom n 
            and the central atom.

        cross_bond_dists: Two dimensional array whose row m contains the
            distances from atom m of atoms n > m that are within a distance 
            cutoff_3 of both atom n and the central atom.

        triplet_counts: One dimensional array of integers whose entry m is the
            number of atoms that are within a distance cutoff_3 of atom m.

    """

    # get 3-body bond array
    ind_3 = -1
    noa = bond_array_2.shape[0]
    for count, dist in enumerate(bond_array_2[:, 0]):
        if dist > cutoff_3:
            ind_3 = count
            break
    if ind_3 == -1:
        ind_3 = noa

    bond_array_3 = bond_array_2[0:ind_3, :]
    bond_positions_3 = bond_positions_2[0:ind_3, :]

    # get cross bond array
    cross_bond_inds = np.zeros((ind_3, ind_3), dtype=np.int8) - 1
    cross_bond_dists = np.zeros((ind_3, ind_3), dtype=np.float64)
    triplet_counts = np.zeros(ind_3, dtype=np.int8)
    for m in range(ind_3):
        pos1 = bond_positions_3[m]
        count = m + 1
        trips = 0
        for n in range(m + 1, ind_3):
            pos2 = bond_positions_3[n]
            diff = pos2 - pos1
            dist_curr = sqrt(
                diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

            if dist_curr < cutoff_3:
                cross_bond_inds[m, count] = n
                cross_bond_dists[m, count] = dist_curr
                count += 1
                trips += 1
        triplet_counts[m] = trips

    return bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts


@njit
def get_m2_body_arrays(positions, atom: int, cell, cutoff: float, species, 
                       cutoff_func=cf.quadratic_cutoff):
    # TODO: 
    # 1. need to deal with the conflict of cutoff functions if other funcs are used
    # 2. complete the docs of "Return"
    """Returns distances, and species of atoms in the many-body
    local environment, and returns distances and numbers of neighbours for atoms in the one
    many-body local environment. This method is implemented outside the AtomicEnvironment
    class to allow for njit acceleration with Numba.
    
    Note: here we assume the cutoff is not too large, i.e., 2 * cutoff < cell_size 

    Args:
        positions (np.ndarray): Positions of atoms in the structure.
        atom (int): Index of the central atom of the local environment.
        cell (np.ndarray): 3x3 array whose rows are the Bravais lattice vectors of the
            cell.
        cutoff (float): 2-body cutoff radius.
        species (np.ndarray): Numpy array of species represented by their atomic numbers.

    Return:
        Tuple of arrays describing pairs of atoms in the 2-body local
        environment.
    """
    # Get distances, positions, species and indexes of neighbouring atoms
    bond_array_mb, __, etypes, bond_inds = get_2_body_arrays(
        positions, atom, cell, cutoff, species)

    species_list = np.array(list(set(species)), dtype=np.int8)
    n_bonds = len(bond_inds)
    n_specs = len(species_list)
    qs = np.zeros(n_specs, dtype=np.float64)
    qs_neigh = np.zeros((n_bonds, n_specs), dtype=np.float64)
    q_grads = np.zeros((n_bonds, 3), dtype=np.float64)

    q_func = coordination_number
    # get coordination number of center atom for each species
    for s in range(n_specs):
        qs[s] = q_value_mc(bond_array_mb[:, 0], cutoff, species_list[s], 
            etypes, cutoff_func, q_func)

    # get coordination number of all neighbor atoms for each species
    for i in range(n_bonds):
        neigh_bond_array, _, neigh_etypes, _ = get_2_body_arrays(positions, 
            bond_inds[i], cell, cutoff, species)
        for s in range(n_specs):
            qs_neigh[i, s] = q_value_mc(neigh_bond_array[:, 0], cutoff,
                species_list[s], neigh_etypes, cutoff_func, q_func)

        # get grad from each neighbor atom, assume the cutoff is not too large
        # such that 2 * cutoff < cell_size 
        ri = bond_array_mb[i, 0]
        for d in range(3):
            ci = bond_array_mb[i, d+1]
            _, q_grads[i, d] = q_func(ri, ci, cutoff, cutoff_func)

    return qs, qs_neigh, q_grads, species_list, etypes 

@njit
def get_m3_body_arrays(positions, atom: int, cell, cutoff: float, species, 
                       cutoff_func=cf.quadratic_cutoff):
    """
    Note: here we assume the cutoff is not too large, 
    i.e., 2 * cutoff < cell_size 
    """
    species_list = np.array(list(set(species)), dtype=np.int8)

    q_func = coordination_number

    bond_array, bond_positions, etypes, bond_inds = \
        get_2_body_arrays(positions, atom, cell, cutoff, species)

    bond_array_m3b, cross_bond_inds, cross_bond_dists, triplets = \
        get_3_body_arrays(bond_array, bond_positions, cutoff)

    # get descriptor of center atom for each species
    m3b_array = q3_value_mc(bond_array_m3b[:, 0], cross_bond_inds, 
        cross_bond_dists, triplets, cutoff, species_list, etypes, 
        cutoff_func, q_func)


    # get descriptor of all neighbor atoms for each species
    n_bonds = len(bond_array_m3b)
    n_specs = len(species_list)
    m3b_neigh_array = np.zeros((n_bonds, n_specs, n_specs))
    for i in range(n_bonds):
        neigh_bond_array, neigh_positions, neigh_etypes, _ = \
            get_2_body_arrays(positions, bond_inds[i], cell, cutoff, species)

        neigh_array_m3b, neigh_cross_inds, neigh_cross_dists, neigh_triplets = \
            get_3_body_arrays(neigh_bond_array, neigh_positions, cutoff)

        m3b_neigh_array[i, :, :] = q3_value_mc(neigh_array_m3b[:, 0],
            neigh_cross_inds, neigh_cross_dists, neigh_triplets, 
            cutoff, species_list, neigh_etypes, cutoff_func, q_func)

    # get grad from each neighbor atom, assume the cutoff is not too large
    # such that 2 * cutoff < cell_size 
    m3b_neigh_grads = q3_neigh_grads_mc(bond_array_m3b, cross_bond_inds, 
        cross_bond_dists, triplets, cutoff, species_list, etypes, 
        cutoff_func, q_func)

    m3b_grads = q3_grads_mc(m3b_neigh_grads, species_list, etypes)

    return m3b_array, m3b_neigh_array, m3b_grads, m3b_neigh_grads, species_list, etypes

@njit
def q3_grads_mc(neigh_grads, species_list, etypes):
    n_specs = len(species_list)
    n_neigh = neigh_grads.shape[0]
    grads = np.zeros((n_specs, n_specs, 3))
    for i in range(n_neigh):
        si = np.where(species_list==etypes[i])[0][0]
        for spec_j in species_list:
            sj = np.where(species_list==spec_j)[0][0]
            if si == sj:
                grads[si, sj, :] += neigh_grads[i, sj, :] / 2
            else:
                grads[si, sj, :] += neigh_grads[i, sj, :]

    return grads

@njit
def q3_neigh_grads_mc(bond_array_m3b, cross_bond_inds, cross_bond_dists, 
    triplets, r_cut, species_list, etypes, cutoff_func, 
    q_func=coordination_number):

    n_bonds = len(bond_array_m3b)
    n_specs = len(species_list)
    m3b_grads = np.zeros((n_bonds, n_specs, 3))

    # get grad from each neighbor atom
    for i in range(n_bonds):

        # get grad of q_func
        ri = bond_array_m3b[i, 0]
        si = np.where(species_list==etypes[i])[0][0]
        qi, _ = q_func(ri, 0, r_cut, cutoff_func)

        qi_grads = np.zeros(3)
        for d in range(3):
            ci = bond_array_m3b[i, d + 1]
            _, qi_grads[d] = q_func(ri, ci, r_cut, cutoff_func)

        # go through all triplets with "atom" and "i"
        for ind in range(triplets[i]): 
            j = cross_bond_inds[i, i + ind + 1]
            rj = bond_array_m3b[j, 0]
            sj = np.where(species_list==etypes[j])[0][0]
            qj, _ = q_func(rj, 0, r_cut, cutoff_func)
            
            qj_grads = np.zeros(3)
            for d in range(3):
                cj = bond_array_m3b[j, d + 1]
                _, qj_grads[d] = q_func(rj, cj, r_cut, cutoff_func)

            rij = cross_bond_dists[i, i + ind + 1] 
            qij, _ = q_func(rij, 0, r_cut, cutoff_func)

            q_grad = (qi_grads * qj + qi * qj_grads) * qij

            # remove duplicant
    #        if si == sj:
    #            q_grad /= 2
            m3b_grads[i, sj, :] += q_grad
            m3b_grads[j, si, :] += q_grad

    return m3b_grads


@njit
def q3_value_mc(distances, cross_bond_inds, cross_bond_dists, triplets,
    r_cut, species_list, etypes, cutoff_func, q_func=coordination_number):
    """Compute value of many-body many components descriptor based
    on distances of atoms in the local many-body environment.

    Args:
        distances (np.ndarray): distances between atoms i and j
        r_cut (float): cutoff hyperparameter
        ref_species (int): species to consider to compute the contribution
        etypes (np.ndarray): atomic species of neighbours
        cutoff_func (callable): cutoff function
        q_func (callable): many-body pairwise descrptor function

    Return:
        float: the value of the many-body descriptor
    """
    n_specs = len(species_list)
    mb3_array = np.zeros((n_specs, n_specs))
    n_bonds = len(distances)

    for m in range(n_bonds):
        q1, _ = q_func(distances[m], 0, r_cut, cutoff_func)
        s1 = np.where(species_list==etypes[m])[0][0]

        for n in range(triplets[m]):
            ind = cross_bond_inds[m, m + n + 1]
            s2 = np.where(species_list==etypes[ind])[0][0] 
            q2, _ = q_func(distances[ind], 0, r_cut, cutoff_func)
    
            r3 = cross_bond_dists[m, m + n + 1]
            q3, _ = q_func(r3, 0, r_cut, cutoff_func)

            mb3_array[s1, s2] += q1 * q2 * q3
            if s1 != s2:
                mb3_array[s2, s1] += q1 * q2 * q3

    return mb3_array 



if __name__ == '__main__':
    pass
