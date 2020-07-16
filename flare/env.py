"""The :class:`AtomicEnvironment` object stores information about the local
environment of an atom. :class:`AtomicEnvironment` objects are inputs to the
2-, 3-, and 2+3-body kernels."""
import numpy as np
from copy import deepcopy
from math import ceil
from flare.struc import Structure
import flare.kernels.cutoffs as cf
from flare.utils.env_getarray import get_2_body_arrays, get_3_body_arrays, \
    get_m2_body_arrays



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
    * ntwobody: Integer, number of different hyperparameter/cutoff sets to
        associate with different 2-body pairings of atoms in groups defined in
        specie_mask.
    * twobody_mask: Array of length nspecie^2, which describes the cutoff to
        associate with different pairings of species types. For example, if
        there are atoms of type 0 and 1, then twobody_mask defines which cutoff
        to use for parings [0-0, 0-1, 1-0, 1-1]: if we wanted cutoff0 for
        0-0 parings and set 1 for 0-1 and 1-1 pairings, then we would make
        twobody_mask [0, 1, 1, 1].
    * twobody_cutoff_list: Array of length ntwobody, which stores the cutoff
        used for different types of bonds defined in twobody_mask
    * ncut3b: Integer, number of different cutoffs sets to associate
        with different 3-body pariings of atoms in groups defined in
        specie_mask.
    * cut3b_mask: Array of length nspecie^2, which describes the cutoff to
        associate with different bond types in triplets. For example, in a
        triplet (C, O, H) , there are three cutoffs. Cutoffs for CH bond, CO
        bond and OH bond. If C and O are associate with atom group 1 in
        specie_mask and H are associate with group 0 in specie_mask, the
        cut3b_mask[1*nspecie+0] determines the C/O-H bond cutoff, and
        cut3b_mask[1*nspecie+1] determines the C-O bond cutoff. If we want the
        former one to use the 1st cutoff in threebody_cutoff_list and the later
        to use the 2nd cutoff in threebody_cutoff_list, the cut3b_mask should
        be [0, 0, 0, 1].
    * threebody_cutoff_list: Array of length ncut3b, which stores the cutoff
        used for different types of bonds in triplets.
    * nmanybody: Integer, number of different cutoffs set to associate with
        different coordination numbers.
    * manybody_mask: Similar to twobody_mask and cut3b_mask.
    * manybody_cutoff_list: Array of length nmanybody, stores the cutoff used
        for different many body terms

    Examples can be found at the end of in tests/test_env.py

    """

    all_kernel_types = ['twobody', 'threebody', 'manybody']
    ndim = {'twobody': 2, 'threebody': 3, 'manybody': 2, 'cut3b': 2}

    def __init__(self, structure: Structure, atom: int, cutoffs,
                 cutoffs_mask=None):

        self.structure = structure
        self.positions = structure.wrapped_positions
        self.cell = np.array(structure.cell)
        self.species = structure.coded_species

        # backward compatability
        if not isinstance(cutoffs, dict):
            newcutoffs = {'twobody': cutoffs[0]}
            if len(cutoffs) > 1:
                newcutoffs['threebody'] = cutoffs[1]
            if len(cutoffs) > 2:
                newcutoffs['manybody'] = cutoffs[2]
            cutoffs = newcutoffs

        if cutoffs_mask is None:
            cutoffs_mask = {'cutoffs': cutoffs}
        elif cutoffs is not None:
            cutoffs_mask['cutoffs'] = deepcopy(cutoffs)

        # Set the sweep array based on the max cutoff.
        sweep_val = ceil(np.max(list(cutoffs.values())) / structure.max_cutoff)
        self.sweep_val = sweep_val
        self.sweep_array = np.arange(-sweep_val, sweep_val + 1, 1)

        self.atom = atom
        self.ctype = structure.coded_species[atom]

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

        self.bond_array_2 = None
        self.etypes = None
        self.bond_inds = None
        self.bond_array_3 = None
        self.cross_bond_inds = None
        self.cross_bond_dists = None
        self.triplet_counts = None
        self.q_array = None
        self.q_neigh_array = None
        self.q_grads = None
        self.q_neigh_grads = None
        self.unique_species = None
        self.etypes_mb = None

        self.compute_env()

    def setup_mask(self, cutoffs_mask):

        self.cutoffs_mask = deepcopy(cutoffs_mask)
        self.cutoffs = cutoffs_mask['cutoffs']

        for kernel in AtomicEnvironment.all_kernel_types:
            ndim = AtomicEnvironment.ndim[kernel]
            if kernel in self.cutoffs:
                setattr(self, kernel + '_cutoff', self.cutoffs[kernel])

        if (self.twobody_cutoff == 0):
            self.twobody_cutoff = \
                np.max([self.threebody_cutoff, self.manybody_cutoff])

            self.cutoffs['twobody'] = self.twobody_cutoff

        self.nspecie = cutoffs_mask.get('nspecie', 1)
        if 'specie_mask' in cutoffs_mask:
            self.specie_mask = np.array(cutoffs_mask['specie_mask'],
                                        dtype=np.int)

        for kernel in AtomicEnvironment.all_kernel_types:
            ndim = AtomicEnvironment.ndim[kernel]
            if kernel in self.cutoffs:
                setattr(self, kernel + '_cutoff', self.cutoffs[kernel])
                setattr(self, 'n' + kernel, 1)
                if kernel != 'threebody':
                    name_list = [kernel + '_cutoff_list',
                                 'n' + kernel, kernel + '_mask']
                    for name in name_list:
                        if name in cutoffs_mask:
                            setattr(self, name, cutoffs_mask[name])
                else:
                    self.ncut3b = cutoffs_mask.get('ncut3b', 1)
                    self.cut3b_mask = cutoffs_mask.get('cut3b_mask', None)
                    if 'threebody_cutoff_list' in cutoffs_mask:
                        self.threebody_cutoff_list = \
                            np.array(cutoffs_mask['threebody_cutoff_list'],
                                     dtype=np.float)


    def compute_env(self):

        # get 2-body arrays
        if self.ntwobody >= 1:
            bond_array_2, bond_positions_2, etypes, bond_inds = \
                get_2_body_arrays(
                    self.positions, self.atom, self.cell, self.twobody_cutoff,
                    self.twobody_cutoff_list, self.species, self.sweep_array,
                    self.nspecie, self.specie_mask, self.twobody_mask)

            self.bond_array_2 = bond_array_2
            self.etypes = etypes
            self.bond_inds = bond_inds

        # if 2 cutoffs are given, create 3-body arrays
        if self.ncut3b > 0:
            bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts = \
                get_3_body_arrays(
                    bond_array_2, bond_positions_2, self.species[self.atom],
                    etypes, self.threebody_cutoff, self.threebody_cutoff_list,
                    self.nspecie, self.specie_mask, self.cut3b_mask)

            self.bond_array_3 = bond_array_3
            self.cross_bond_inds = cross_bond_inds
            self.cross_bond_dists = cross_bond_dists
            self.triplet_counts = triplet_counts

        # if 3 cutoffs are given, create many-body arrays
        if self.nmanybody > 0:
            self.q_array, self.q_neigh_array, self.q_grads, \
                self.q_neigh_grads, self.unique_species, self.etypes_mb = \
                get_m2_body_arrays(
                    self.positions, self.atom, self.cell, self.manybody_cutoff,
                    self.manybody_cutoff_list, self.species, self.sweep_array,
                    self.nspecie, self.specie_mask, self.manybody_mask,
                    cf.quadratic_cutoff)

    def as_dict(self, include_structure: bool = False):
        """
        Returns Atomic Environment object as a dictionary for serialization
        purposes. Optional to not include the structure to avoid redundant
        information.
        :return:
        """

        dictionary = dict(vars(self))
        dictionary['object'] = 'AtomicEnvironment'
        dictionary['forces'] = self.structure.forces

        # Backward compatibility for older models: Cutoffs mask.
        # Can be deleted one day if support is dropped for older (Pre June
        # 2020) pickled environment objects.
        cutoffs_mask = getattr(self, 'cutoffs_mask', {'cutoffs': self.cutoffs})
        if not hasattr(self, 'cutoffs_mask'):
            self.cutoffs_mask = cutoffs_mask
        dictionary['cutoffs_mask'] = cutoffs_mask

        if not include_structure:
            del dictionary['structure']
        else:
            dictionary['structure'] = dictionary['structure'].as_dict()

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

        return AtomicEnvironment(
            struc, index, cutoffs, cutoffs_mask=cutoffs_mask)

    def __str__(self):
        atom_type = self.ctype
        neighbor_types = self.etypes
        n_neighbors = len(self.bond_array_2)
        string = 'Atomic Env. of Type {} surrounded by {} atoms ' \
                 'of Types {}'.format(atom_type, n_neighbors,
                                      sorted(list(set(neighbor_types))))

        return string
