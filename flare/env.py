"""The :class:`AtomicEnvironment` object stores information about the local
environment of an atom. :class:`AtomicEnvironment` objects are inputs to the
2-, 3-, and 2+3-body kernels."""
import numpy as np
from copy import deepcopy
from math import sqrt
from numba import njit
from flare.struc import Structure
from flare.parameters import Parameters
from flare.kernels.kernels import coordination_number, q_value_mc
import flare.kernels.cutoffs as cf


class AtomicEnvironment:
    """Contains information about the local environment of an atom,
    including arrays of pair and triplet distances and the chemical
    species of atoms in the environment.

    :param structure: Structure of atoms.
    :type structure: struc.Structure
    :param atom: Index of the atom in the structure.
    :type atom: int
    :param cutoffs: 2- and 3-body cutoff radii. 2-body if one cutoff is
    given, 2+3-body if two are passed.
    :type cutoffs: np.ndarray
    :param cutoffs_mask: a dictionary to store multiple cutoffs if neede
                         it should be exactly the same as the hyps mask
    :type cutoffs_mask: dict

    The cutoffs_mask allows the user to define multiple cutoffs for different
    bonds, triples, and many body interaction. This dictionary should be
    consistent with the hyps_mask used in the GuassianProcess object.

    * specie_mask: 118-long integer array descirbing which elements belong to
                 like groups for determining which bond hyperparameters to use.
                 For instance, [0,0,1,1,0 ...] assigns H to group 0, He and
                 Li to group 1, and Be to group 0 (the 0th register is ignored).
    * nspecie: Integer, number of different species groups (equal to number of
             unique values in specie_mask).
    * ntwobody: Integer, number of different hyperparameter/cutoff sets to associate with
             different 2-body pairings of atoms in groups defined in specie_mask.
    * twobody_mask: Array of length nspecie^2, which describes the cutoff to
                 associate with different pairings of species types. For example, if there
                 are atoms of type 0 and 1, then twobody_mask defines which cutoff
                 to use for parings [0-0, 0-1, 1-0, 1-1]: if we wanted cutoff0 for
                 0-0 parings and set 1 for 0-1 and 1-1 pairings, then we would make
                 twobody_mask [0, 1, 1, 1].
    * twobody_cutoff_list: Array of length ntwobody, which stores the cutoff used for different
                 types of bonds defined in twobody_mask
    * ncut3b:    Integer, number of different cutoffs sets to associate
                 with different 3-body pariings of atoms in groups defined in specie_mask.
    * cut3b_mask: Array of length nspecie^2, which describes the cutoff to
                 associate with different bond types in triplets. For example, in a triplet
                 (C, O, H) , there are three cutoffs. Cutoffs for CH bond, CO bond and OH bond.
                 If C and O are associate with atom group 1 in specie_mask and H are associate with
                 group 0 in specie_mask, the cut3b_mask[1*nspecie+0] determines the C/O-H bond cutoff,
                 and cut3b_mask[1*nspecie+1] determines the C-O bond cutoff. If we want the
                 former one to use the 1st cutoff in threebody_cutoff_list and the later to use the 2nd cutoff
                 in threebody_cutoff_list, the cut3b_mask should be [0, 0, 0, 1]
    * threebody_cutoff_list: Array of length ncut3b, which stores the cutoff used for different
                 types of bonds in triplets.
    * nmanybody :      Integer, number of different cutoffs set to associate with different coordination
                 numbers
    * manybody_mask:   similar to twobody_mask and cut3b_mask.
    * manybody_cutoff_list: Array of length nmanybody, stores the cutoff used for different many body terms

    Examples can be found at the end of in tests/test_env.py

    """

    all_kernel_types = ['twobody', 'threebody', 'manybody']
    ndim = {'twobody': 2, 'threebody': 3, 'manybody': 2, 'cut3b': 2}

    def __init__(self, structure: Structure, atom: int, cutoffs, sweep=1, cutoffs_mask=None):
        self.structure = structure
        self.positions = structure.wrapped_positions
        self.cell = structure.cell
        self.species = structure.coded_species
        self.sweep_array = np.arange(-sweep, sweep+1, 1)

        self.atom = atom
        self.ctype = structure.coded_species[atom]

        # backward compatability
        if not isinstance(cutoffs, dict):
            newcutoffs = {'twobody':cutoffs[0]}
            if len(cutoffs)>1:
                newcutoffs['threebody'] = cutoffs[1]
            if len(cutoffs)>2:
                newcutoffs['manybody'] = cutoffs[2]
            cutoffs = newcutoffs

        if cutoffs_mask is None:
            cutoffs_mask = {'cutoffs': cutoffs}
        elif cutoffs is not None:
            cutoffs_mask['cutoffs'] = deepcopy(cutoffs)

        self.twobody_cutoff = 0
        self.threebody_cutoff = 0
        self.manybody_cutoff = 0

        self.ntwobody = 1
        self.ncut3b = 0
        self.nmanybody = 0

        self.nspecie = 1
        self.specie_mask = None
        self.twobody_mask = None
        self.threebody_mask = None
        self.manybody_mask = None
        self.twobody_cutoff_list = None
        self.threebody_cutoff_list = None
        self.manybody_cutoff_list = None

        self.setup_mask(cutoffs_mask)

        assert self.threebody_cutoff <= self.twobody_cutoff, \
            "2b cutoff has to be larger than 3b cutoff"
        # # TO DO, once the mb function is updated to use the bond_array_2
        # # this block should be activated.
        # assert self.manybody_cutoff <= self.twobody_cutoff, \
        #         "mb cutoff has to be larger than mb cutoff"

        self.compute_env()

    def setup_mask(self, cutoffs_mask):

        self.cutoffs_mask = deepcopy(cutoffs_mask)
        self.cutoffs = cutoffs_mask['cutoffs']

        for kernel in AtomicEnvironment.all_kernel_types:
            ndim = AtomicEnvironment.ndim[kernel]
            if kernel in self.cutoffs:
                setattr(self, kernel+'_cutoff', self.cutoffs[kernel])

        if (self.twobody_cutoff == 0):
            self.twobody_cutoff = np.max([self.threebody_cutoff, self.manybody_cutoff])
            self.cutoffs['twobody'] = self.twobody_cutoff

        self.nspecie = cutoffs_mask.get('nspecie', 1)
        if 'specie_mask' in cutoffs_mask:
            self.specie_mask = np.array(cutoffs_mask['specie_mask'], dtype=np.int)

        for kernel in AtomicEnvironment.all_kernel_types:
            ndim = AtomicEnvironment.ndim[kernel]
            if kernel in self.cutoffs:
                setattr(self, kernel+'_cutoff', self.cutoffs[kernel])
                setattr(self, 'n'+kernel, 1)
                if kernel != 'threebody':
                    name_list = [kernel+'_cutoff_list',
                                 'n'+kernel, kernel+'_mask']
                    for name in name_list:
                        if name in cutoffs_mask:
                            setattr(self, name, cutoffs_mask[name])
                else:
                    self.ncut3b = cutoffs_mask.get('ncut3b', 1)
                    self.cut3b_mask = cutoffs_mask.get('cut3b_mask', None)
                    if 'threebody_cutoff_list' in cutoffs_mask:
                        self.threebody_cutoff_list = np.array(cutoffs_mask['threebody_cutoff_list'], dtype=np.float)

    def compute_env(self):

        # get 2-body arrays
        if (self.ntwobody >= 1):
            bond_array_2, bond_positions_2, etypes, bond_inds = \
                get_2_body_arrays(self.positions, self.atom, self.cell, self.twobody_cutoff,
                                  self.twobody_cutoff_list, self.species, self.sweep_array,
                                  self.nspecie, self.specie_mask, self.twobody_mask)

            self.bond_array_2 = bond_array_2
            self.etypes = etypes
            self.bond_inds = bond_inds

        # if 2 cutoffs are given, create 3-body arrays
        if self.ncut3b > 0:
            bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts = \
                get_3_body_arrays(bond_array_2, bond_positions_2,
                                  self.species[self.atom], etypes, self.threebody_cutoff,
                                  self.threebody_cutoff_list,
                                  self.nspecie, self.specie_mask, self.cut3b_mask)
            self.bond_array_3 = bond_array_3
            self.cross_bond_inds = cross_bond_inds
            self.cross_bond_dists = cross_bond_dists
            self.triplet_counts = triplet_counts

        # if 3 cutoffs are given, create many-body arrays
        if self.nmanybody > 0:
            self.q_array, self.q_neigh_array, self.q_neigh_grads, \
                self.unique_species, self.etypes_mb = \
                get_m_body_arrays(self.positions, self.atom, self.cell,
                                  self.manybody_cutoff, self.manybody_cutoff_list,
                                  self.species, self.sweep_array,
                                  self.nspecie, self.specie_mask,
                                  self.manybody_mask, cf.quadratic_cutoff)

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
        dictionary['cutoffs_mask'] = self.cutoffs_mask

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
            cutoffs = {}

        cutoffs_mask = dictionary.get('cutoffs_mask', None)

        return AtomicEnvironment(struc, index, cutoffs, cutoffs_mask=cutoffs_mask)

    def __str__(self):
        atom_type = self.ctype
        neighbor_types = self.etypes
        n_neighbors = len(self.bond_array_2)
        string = 'Atomic Env. of Type {} surrounded by {} atoms '\
                 'of Types {}'.format(atom_type, n_neighbors,
                                      sorted(list(set(neighbor_types))))

        return string


@njit
def get_2_body_arrays(positions, atom: int, cell, r_cut, cutoff_2, species, sweep,
                      nspecie, specie_mask, twobody_mask):
    """Returns distances, coordinates, species of atoms, and indices of neighbors
    in the 2-body local environment. This method is implemented outside
    the AtomicEnvironment class to allow for njit acceleration with Numba.

    :param positions: Positions of atoms in the structure.
    :type positions: np.ndarray
    :param atom: Index of the central atom of the local environment.
    :type atom: int
    :param cell: 3x3 array whose rows are the Bravais lattice vectors of the
        cell.
    :type cell: np.ndarray
    :param cutoff_2: 2-body cutoff radius.
    :type cutoff_2: np.ndarray
    :param species: Numpy array of species represented by their atomic numbers.
    :type species: np.ndarray
    :param nspecie: number of atom types to define bonds
    :type: int
    :param specie_mask: mapping from atomic number to atom types
    :type: np.ndarray
    :param twobody_mask: mapping from the types of end atoms to bond types
    :type: np.ndarray
    :return: Tuple of arrays describing pairs of atoms in the 2-body local
     environment.

     bond_array_2: Array containing the distances and relative
     coordinates of atoms in the 2-body local environment. First column
     contains distances, remaining columns contain Cartesian coordinates
     divided by the distance (with the origin defined as the position of the
     central atom). The rows are sorted by distance from the central atom.

     bond_positions_2: Coordinates of atoms in the 2-body local environment.

     etypes: Species of atoms in the 2-body local environment represented by
     their atomic number.

     bond_indices: Structure indices of atoms in the local environment.

    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    noa = len(positions)
    pos_atom = positions[atom]
    coords = np.zeros((noa, 3, 27), dtype=np.float64)
    dists = np.zeros((noa, 27), dtype=np.float64)
    cutoff_count = 0

    vec1 = cell[0]
    vec2 = cell[1]
    vec3 = cell[2]

    sepcut = False
    if nspecie > 1 and cutoff_2 is not None:
        sepcut = True
        bc = specie_mask[species[atom]]
        bcn = nspecie * bc

    # record distances and positions of images
    for n in range(noa):
        diff_curr = positions[n] - pos_atom
        im_count = 0
        if sepcut:
            bn = specie_mask[species[n]]
            r_cut = cutoff_2[twobody_mask[bn+bcn]]

        for s1 in sweep:
            for s2 in sweep:
                for s3 in sweep:
                    im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3
                    dist = sqrt(im[0] * im[0] + im[1] * im[1] + im[2] * im[2])
                    if (dist < r_cut) and (dist != 0):
                        dists[n, im_count] = dist
                        coords[n, :, im_count] = im
                        cutoff_count += 1
                    im_count += 1

    # create 2-body bond array
    bond_indices = np.zeros(cutoff_count, dtype=np.int8)
    bond_array_2 = np.zeros((cutoff_count, 4), dtype=np.float64)
    bond_positions_2 = np.zeros((cutoff_count, 3), dtype=np.float64)
    etypes = np.zeros(cutoff_count, dtype=np.int8)
    bond_count = 0

    for m in range(noa):
        spec_curr = species[m]
        if sepcut:
            bm = specie_mask[species[m]]
            r_cut = cutoff_2[twobody_mask[bm+bcn]]
        for n in range(27):
            dist_curr = dists[m, n]
            if (dist_curr < r_cut) and (dist_curr != 0):
                coord = coords[m, :, n]
                bond_array_2[bond_count, 0] = dist_curr
                bond_array_2[bond_count, 1:4] = coord / dist_curr
                bond_positions_2[bond_count, :] = coord
                etypes[bond_count] = spec_curr
                bond_indices[bond_count] = m
                bond_count += 1

    # sort by distance
    sort_inds = bond_array_2[:, 0].argsort()
    bond_array_2 = bond_array_2[sort_inds]
    bond_positions_2 = bond_positions_2[sort_inds]
    bond_indices = bond_indices[sort_inds]
    etypes = etypes[sort_inds]

    return bond_array_2, bond_positions_2, etypes, bond_indices


@njit
def get_3_body_arrays(bond_array_2, bond_positions_2, ctype,
                      etypes, r_cut, cutoff_3,
                      nspecie, specie_mask, cut3b_mask):
    """Returns distances and coordinates of triplets of atoms in the
    3-body local environment.

    :param bond_array_2: 2-body bond array.
    :type bond_array_2: np.ndarray
    :param bond_positions_2: Coordinates of atoms in the 2-body local
     environment.
    :type bond_positions_2: np.ndarray
    :param ctype: atomic number of the center atom
    :type: int
    :param cutoff_3: 3-body cutoff radius.
    :type cutoff_3: np.ndarray
    :param nspecie: number of atom types to define bonds
    :type: int
    :param specie_mask: mapping from atomic number to atom types
    :type: np.ndarray
    :param cut3b_mask: mapping from the types of end atoms to bond types
    :type: np.ndarray
    :return: Tuple of 4 arrays describing triplets of atoms in the 3-body local
     environment.

     bond_array_3: Array containing the distances and relative
     coordinates of atoms in the 3-body local environment. First column
     contains distances, remaining columns contain Cartesian coordinates
     divided by the distance (with the origin defined as the position of the
     central atom). The rows are sorted by distance from the central atom.

     cross_bond_inds: Two dimensional array whose row m contains the indices
     of atoms n > m that are within a distance cutoff_3 of both atom n and the
     central atom.

     cross_bond_dists: Two dimensional array whose row m contains the
     distances from atom m of atoms n > m that are within a distance cutoff_3
     of both atom n and the central atom.

     triplet_counts: One dimensional array of integers whose entry m is the
     number of atoms that are within a distance cutoff_3 of atom m.

    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    sepcut = False
    if nspecie > 1 and cutoff_3 is not None:
        bc = specie_mask[ctype]
        bcn = nspecie * bc
        r_cut = np.max(cutoff_3)
        sepcut = True

    # get 3-body bond array
    ind_3_l = np.where(bond_array_2[:, 0] > r_cut)[0]
    if (ind_3_l.shape[0] > 0):
        ind_3 = ind_3_l[0]
    else:
        ind_3 = bond_array_2.shape[0]

    bond_array_3 = bond_array_2[0:ind_3, :]
    bond_positions_3 = bond_positions_2[0:ind_3, :]

    cut_m = r_cut
    cut_n = r_cut
    cut_mn = r_cut

    # get cross bond array
    cross_bond_inds = np.zeros((ind_3, ind_3), dtype=np.int8) - 1
    cross_bond_dists = np.zeros((ind_3, ind_3), dtype=np.float64)
    triplet_counts = np.zeros(ind_3, dtype=np.int8)
    for m in range(ind_3):
        pos1 = bond_positions_3[m]
        count = m + 1
        trips = 0

        if sepcut:
            # choose bond dependent bond
            bm = specie_mask[etypes[m]]
            btype_m = cut3b_mask[bm + bcn]  # (m, c)
            cut_m = cutoff_3[btype_m]
            bmn = nspecie * bm  # for cross_dist usage

        for n in range(m + 1, ind_3):

            if sepcut:
                bn = specie_mask[etypes[n]]
                btype_n = cut3b_mask[bn + bcn]  # (n, c)
                cut_n = cutoff_3[btype_n]

                # for cross_dist (m,n) pair
                btype_mn = cut3b_mask[bn + bmn]
                cut_mn = cutoff_3[btype_mn]

            pos2 = bond_positions_3[n]
            diff = pos2 - pos1
            dist_curr = sqrt(
                diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

            if dist_curr < cut_mn \
                    and bond_array_2[m, 0] < cut_m \
                    and bond_array_2[n, 0] < cut_n:
                cross_bond_inds[m, count] = n
                cross_bond_dists[m, count] = dist_curr
                count += 1
                trips += 1

        triplet_counts[m] = trips

    return bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts


@njit
def get_m_body_arrays(positions, atom: int, cell, r_cut, manybody_cutoff_list,
                      species, sweep: np.ndarray, nspec, spec_mask, manybody_mask,
                      cutoff_func=cf.quadratic_cutoff):
    # TODO:
    # 1. need to deal with the conflict of cutoff functions if other funcs are used
    # 2. complete the docs of "Return"
    # TODO: this can be probably improved using stored arrays, redundant calls to get_2_body_arrays
    # Get distances, positions, species and indices of neighbouring atoms
    """
    Args:
        positions (np.ndarray): Positions of atoms in the structure.
        atom (int): Index of the central atom of the local environment.
        cell (np.ndarray): 3x3 array whose rows are the Bravais lattice vectors of the
            cell.
        manybody_cutoff_list (float): 2-body cutoff radius.
        species (np.ndarray): Numpy array of species represented by their atomic numbers.

    Return:
        Tuple of arrays describing pairs of atoms in the 2-body local
        environment.
    """
    # Get distances, positions, species and indexes of neighbouring atoms
    bond_array_mb, __, etypes, bond_inds = get_2_body_arrays(
        positions, atom, cell, r_cut, manybody_cutoff_list, species, sweep,
        nspec, spec_mask, manybody_mask)

    sepcut = False
    if nspec > 1 and manybody_cutoff_list is not None:
        bc = spec_mask[species[atom]]
        bcn = bc * nspec
        sepcut = True

    species_list = np.array(list(set(species)), dtype=np.int8)
    n_bonds = len(bond_inds)
    n_specs = len(species_list)
    qs = np.zeros(n_specs, dtype=np.float64)
    qs_neigh = np.zeros((n_bonds, n_specs), dtype=np.float64)
    q_grads = np.zeros((n_bonds, 3), dtype=np.float64)

    # get coordination number of center atom for each species
    for s in range(n_specs):
        if sepcut:
            bs = spec_mask[species_list[s]]
            mbtype = manybody_mask[bcn + bs]
            r_cut = manybody_cutoff_list[mbtype]

        qs[s] = q_value_mc(bond_array_mb[:, 0], r_cut, species_list[s],
                           etypes, cutoff_func)

    # get coordination number of all neighbor atoms for each species
    for i in range(n_bonds):
        if sepcut:
            be = spec_mask[etypes[i]]
            ben = be * nspec

        neigh_bond_array, _, neigh_etypes, _ = \
            get_2_body_arrays(positions, bond_inds[i], cell, r_cut,
                              manybody_cutoff_list, species, sweep, nspec, spec_mask, manybody_mask)
        for s in range(n_specs):
            if sepcut:
                bs = spec_mask[species_list[s]]
                mbtype = manybody_mask[bs + ben]
                r_cut = manybody_cutoff_list[mbtype]

            qs_neigh[i, s] = q_value_mc(neigh_bond_array[:, 0], r_cut,
                                        species_list[s], neigh_etypes, cutoff_func)

    # get grad from each neighbor atom
    for i in range(n_bonds):
        if sepcut:
            be = spec_mask[etypes[i]]
            mbtype = manybody_mask[bcn + be]
            r_cut = manybody_cutoff_list[mbtype]

        ri = bond_array_mb[i, 0]
        for d in range(3):
            ci = bond_array_mb[i, d+1]

            _, q_grads[i, d] = coordination_number(ri, ci, r_cut,
                                                   cutoff_func)

    return qs, qs_neigh, q_grads, species_list, etypes


if __name__ == '__main__':
    pass
