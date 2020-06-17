"""
Multicomponent kernels (simple) restrict all signal variance and length scale of hyperparameters
to a single value. The kernels in this module allow you to have different sets of hyperparameters
and cutoffs for different interactions, and have flexible groupings of elements. It also allows
you to do partial hyper-parameter training, keeping some components fixed.

To use this set of kernels, we need a hyps_mask dictionary for GaussianProcess, MappedGaussianProcess,
and AtomicEnvironment (if you also set up different cutoffs).  A simple example is shown below.

Examples:

    >>> from flare.util.parameter_helper import ParameterHelper
    >>> from flare.gp import GaussianProcess

    >>> pm = ParameterHelper(species=['O', 'C', 'H'],
    ...                      kernels={'twobody':[['*', '*'], ['O','O']],
    ...                               'threebody':[['*', '*', '*'], ['O','O', 'O']]},
    ...                      parameters={'twobody0':[1, 0.5, 1], 'twobody1':[2, 0.2, 2],
    ...                            'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
    ...                            'cutoff_twobody':2, 'cutoff_threebody':1, 'noise': 0.05},
    ...                      constraints={'twobody0':[False, True]})
    >>> hyps_mask = pm1.as_dict()
    >>> hyps = hyps_mask.pop('hyps')
    >>> cutoffs = hyps_mask.pop('cutoffs')
    >>> hyp_labels = hyps_mask.pop('hyp_labels')
    >>> kernels = hyps_mask['kernels']
    >>> gp_model = GaussianProcess(kernels=kernels,
    ...                            hyps=hyps, cutoffs=cutoffs,
    ...                            hyp_labels=hyp_labels,
    ...                            parallel=True, per_atom_par=False,
    ...                            n_cpus=n_cpus,
    ...                            multihyps=True, hyps_mask=hm)


In the example above, Parameters class generates the arrays needed
for these kernels and store all the grouping and mapping information in the
hyps_mask dictionary.  It stores following keys and values:

* spec_mask: 118-long integer array descirbing which elements belong to
             like groups for determining which bond hyperparameters to use. For
             instance, [0,0,1,1,0 ...] assigns H to group 0, He and Li to group 1,
             and Be to group 0 (the 0th register is ignored).
* nspec: Integer, number of different species groups (equal to number of
         unique values in spec_mask).
* nbond: Integer, number of different hyperparameter sets to associate with
         different 2-body pairings of atoms in groups defined in spec_mask.
* bond_mask: Array of length nspec^2, which describes the hyperparameter sets to
             associate with different pairings of species types. For example, if there
             are atoms of type 0 and 1, then bond_mask defines which hyperparameters
             to use for parings [0-0, 0-1, 1-0, 1-1]: if we wanted hyperparameter set 0 for
             0-0 parings and set 1 for 0-1 and 1-1 pairings, then we would make
             bond_mask [0, 1, 1, 1].
* ntriplet: Integer, number of different hyperparameter sets to associate
            with different 3-body pariings of atoms in groups defined in spec_mask.
* triplet_mask: Similar to bond mask: Triplet pairings of type 0 and 1 atoms
                would go {0-0-0, 0-0-1, 0-1-0, 0-1-1, 1-0-0, 1-0-1, 1-1-0, 1-1-1},
                and if we wanted hyp. set 0 for triplets with only atoms of type 0
                and hyp. set 1 for all the rest, then the triplet_mask array would
                read [0,1,1,1,1,1,1,1]. The user should make sure that the mask has
                a permutational symmetry.
* cutoff_2b: Array of length nbond, which stores the cutoff used for different
             types of bonds defined in bond_mask
* ncut3b:    Integer, number of different cutoffs sets to associate
             with different 3-body pariings of atoms in groups defined in spec_mask.
* cut3b_mask: Array of length nspec^2, which describes the cutoff to
             associate with different bond types in triplets. For example, in a triplet
             (C, O, H) , there are three cutoffs. Cutoffs for CH bond, CO bond and OH bond.
             If C and O are associate with atom group 1 in spec_mask and H are associate with
             group 0 in spec_mask, the cut3b_mask[1*nspec+0] determines the C/O-H bond cutoff,
             and cut3b_mask[1*nspec+1] determines the C-O bond cutoff. If we want the
             former one to use the 1st cutoff in cutoff_3b and the later to use the 2nd cutoff
             in cutoff_3b, the cut3b_mask should be [0, 0, 0, 1]
* cutoff_3b: Array of length ncut3b, which stores the cutoff used for different
             types of bonds in triplets.
* nmb :      Integer, number of different cutoffs set to associate with different coordination
             numbers
* mb_mask:   similar to bond_mask and cut3b_mask.
* cutoff_mb: Array of length nmb, stores the cutoff used for different many body terms

For selective optimization. one can define 'map', 'train_noise' and 'original'
to identify which element to be optimized. All three have to be defined.
train_noise = Bool (True/False), whether the noise parameter can be optimized
original: np.array. Full set of initial values for hyperparmeters
map: np.array, array to map the hyper parameter back to the full set.
map[i]=j means the i-th element in hyps should be the j-th element in
hyps_mask['original']

For example, the full set of hyper parmeters
may include [ls21, ls22, sig21, sig22, ls3
sg3, noise] but suppose you wanted only the set 21 optimized.
The full set of hyperparameters is defined in 'original'; include all those
you want to leave static, and set initial guesses for those you want to vary.
Have the 'map' list contain the indices of the hyperparameters in 'original'
that correspond to the hyperparameters you want to vary.
Have a hyps list which contain those which you want to vary. Below,
ls21, ls22 etc... represent floating-point variables which correspond
to the initial guesses / static values.
You would then pass in:

hyps = [ls21, sig21]
hyps_mask = { ..., 'train_noise': False, 'map':[0, 2],
                   'original': [ls21, ls22, sig21, sig22, ls3, sg3, noise]}
the hyps argument should only contain the values that need to be optimized.
If you want noise to be trained as well include noise as the
final hyperparameter value in hyps.

"""

from math import exp
import numpy as np

from numba import njit

import flare.kernels.cutoffs as cf

from flare.kernels.kernels import force_helper, grad_constants, grad_helper, \
    force_energy_helper, three_body_en_helper, three_body_helper_1, \
    three_body_helper_2, three_body_grad_helper_1, three_body_grad_helper_2
from flare.kernels.mc_3b_sepcut import three_body_mc_sepcut_jit, \
    three_body_mc_grad_sepcut_jit, three_body_mc_force_en_sepcut_jit, \
    three_body_mc_en_sepcut_jit
from flare.kernels.mc_mb_sepcut import \
    many_body_mc_sepcut_jit, many_body_mc_grad_sepcut_jit, \
    many_body_mc_force_en_sepcut_jit, many_body_mc_en_sepcut_jit

# -----------------------------------------------------------------------------
#                        two plus three plus many body kernels
# -----------------------------------------------------------------------------


def two_three_many_body_mc(env1, env2, d1, d2, cutoff_2b, cutoff_3b, cutoff_mb,
                           nspec, spec_mask,
                           nbond, bond_mask, ntriplet, triplet_mask,
                           ncut3b, cut3b_mask,
                           nmb, mb_mask,
                           sig2, ls2, sig3, ls3, sigm, lsm,
                           cutoff_func=cf.quadratic_cutoff):
    """2+3+manybody multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        nmb (int): number of different hyperparameter sets to associate with manybody pairings
        mb_mask (np.ndarray): nspec^2 long integer array
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        sigm (np.ndarray): signal variances associates with manybody term
        lsm (np.ndarray): length scales associates with manybody term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3+many-body kernel.
    """

    two_term = two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                               env2.bond_array_2, env2.ctype, env2.etypes,
                               d1, d2, sig2, ls2, cutoff_2b, cutoff_func,
                               nspec, spec_mask, bond_mask)

    if (ncut3b == 0):
        tbmcj = three_body_mc_jit
    else:
        tbmcj = three_body_mc_sepcut_jit

    three_term = \
        tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
              env2.bond_array_3, env2.ctype, env2.etypes,
              env1.cross_bond_inds, env2.cross_bond_inds,
              env1.cross_bond_dists, env2.cross_bond_dists,
              env1.triplet_counts, env2.triplet_counts,
              d1, d2, sig3, ls3, cutoff_3b, cutoff_func,
              nspec, spec_mask, triplet_mask, cut3b_mask)

    mbmcj = many_body_mc_sepcut_jit
    many_term = mbmcj(env1.q_array, env2.q_array,
                      env1.q_neigh_array, env2.q_neigh_array,
                      env1.q_neigh_grads, env2.q_neigh_grads,
                      env1.ctype, env2.ctype,
                      env1.etypes_mb, env2.etypes_mb,
                      env1.unique_species, env2.unique_species,
                      d1, d2, sigm, lsm,
                      nspec, spec_mask, mb_mask)

    return two_term + three_term + many_term


def two_three_many_body_mc_grad(env1, env2, d1, d2, cutoff_2b, cutoff_3b, cutoff_mb,
                                nspec, spec_mask,
                                nbond, bond_mask, ntriplet, triplet_mask,
                                ncut3b, cut3b_mask,
                                nmb, mb_mask,
                                sig2, ls2, sig3, ls3, sigm, lsm,
                                cutoff_func=cf.quadratic_cutoff):
    """2+3+manybody multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        nmb (int): number of different hyperparameter sets to associate with manybody pairings
        mb_mask (np.ndarray): nspec^2 long integer array
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        sigm (np.ndarray): signal variances associates with manybody term
        lsm (np.ndarray): length scales associates with manybody term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        (float, np.ndarray):
            Value of the 2+3+manybody kernel and its gradient
            with respect to the hyperparameters.
    """

    kern2, grad2 = \
        two_body_mc_grad_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                             env2.bond_array_2, env2.ctype, env2.etypes,
                             d1, d2, sig2, ls2, cutoff_2b, cutoff_func,
                             nspec, spec_mask,
                             nbond, bond_mask)

    if (ncut3b == 0):
        tbmcj = three_body_mc_grad_jit
    else:
        tbmcj = three_body_mc_grad_sepcut_jit

    kern3, grad3 = \
        tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
              env2.bond_array_3, env2.ctype, env2.etypes,
              env1.cross_bond_inds, env2.cross_bond_inds,
              env1.cross_bond_dists, env2.cross_bond_dists,
              env1.triplet_counts, env2.triplet_counts,
              d1, d2, sig3, ls3, cutoff_3b,
              cutoff_func,
              nspec, spec_mask,
              ntriplet, triplet_mask, cut3b_mask)

    mbmcj = many_body_mc_grad_sepcut_jit
    kern_many, gradm = mbmcj(env1.q_array, env2.q_array,
                             env1.q_neigh_array, env2.q_neigh_array,
                             env1.q_neigh_grads, env2.q_neigh_grads,
                             env1.ctype, env2.ctype,
                             env1.etypes_mb, env2.etypes_mb,
                             env1.unique_species, env2.unique_species,
                             d1, d2, sigm, lsm,
                             nspec, spec_mask, nmb, mb_mask)

    return kern2 + kern3 + kern_many, np.hstack([grad2, grad3, gradm])


def two_three_many_mc_force_en(env1, env2, d1, cutoff_2b, cutoff_3b, cutoff_mb,
                               nspec, spec_mask,
                               nbond, bond_mask, ntriplet, triplet_mask,
                               ncut3b, cut3b_mask,
                               nmb, mb_mask,
                               sig2, ls2, sig3, ls3, sigm, lsm,
                               cutoff_func=cf.quadratic_cutoff):
    """2+3+manybody multi-element kernel between a force component and a local
    energy.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        nmb (int): number of different hyperparameter sets to associate with manybody pairings
        mb_mask (np.ndarray): nspec^2 long integer array
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        sigm (np.ndarray): signal variances associates with manybody term
        lsm (np.ndarray): length scales associates with manybody term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3+many-body force/energy kernel.
    """

    two_term = \
        two_body_mc_force_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                 env2.bond_array_2, env2.ctype, env2.etypes,
                                 d1, sig2, ls2, cutoff_2b, cutoff_func,
                                 nspec, spec_mask,
                                 bond_mask) / 2

    if (ncut3b == 0):
        tbmcj = three_body_mc_force_en_jit
    else:
        tbmcj = three_body_mc_force_en_sepcut_jit

    three_term = \
        tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
              env2.bond_array_3, env2.ctype, env2.etypes,
              env1.cross_bond_inds, env2.cross_bond_inds,
              env1.cross_bond_dists,
              env2.cross_bond_dists,
              env1.triplet_counts, env2.triplet_counts,
              d1, sig3, ls3, cutoff_3b, cutoff_func,
              nspec, spec_mask, triplet_mask,
              cut3b_mask) / 3

    mbmcj = many_body_mc_force_en_sepcut_jit
    many_term = mbmcj(env1.q_array, env2.q_array,
                      env1.q_neigh_array, env1.q_neigh_grads,
                      env1.ctype, env2.ctype, env1.etypes_mb,
                      env1.unique_species, env2.unique_species,
                      d1, sigm, lsm,
                      nspec, spec_mask, mb_mask)

    return two_term + three_term + many_term


def two_three_many_mc_en(env1, env2, cutoff_2b, cutoff_3b, cutoff_mb,
                         nspec, spec_mask,
                         nbond, bond_mask, ntriplet, triplet_mask,
                         ncut3b, cut3b_mask,
                         nmb, mb_mask,
                         sig2, ls2, sig3, ls3, sigm, lsm,
                         cutoff_func=cf.quadratic_cutoff):
    """2+3+many-body multi-element kernel between two local energies.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        nmb (int): number of different hyperparameter sets to associate with manybody pairings
        mb_mask (np.ndarray): nspec^2 long integer array
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        sigm (np.ndarray): signal variances associates with manybody term
        lsm (np.ndarray): length scales associates with manybody term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3+many-body energy/energy kernel.
    """

    two_term = two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                  env2.bond_array_2, env2.ctype, env2.etypes,
                                  sig2, ls2, cutoff_2b, cutoff_func,
                                  nspec,
                                  spec_mask,
                                  bond_mask)/4

    if (ncut3b == 0):
        tbmcj = three_body_mc_en_jit
    else:
        tbmcj = three_body_mc_en_sepcut_jit

    three_term = \
        tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
              env2.bond_array_3, env2.ctype, env2.etypes,
              env1.cross_bond_inds, env2.cross_bond_inds,
              env1.cross_bond_dists, env2.cross_bond_dists,
              env1.triplet_counts, env2.triplet_counts,
              sig3, ls3, cutoff_3b, cutoff_func,
              nspec, spec_mask,
              triplet_mask, cut3b_mask)/9.

    mbmcj = many_body_mc_en_sepcut_jit
    many_term = mbmcj(env1.q_array, env2.q_array,
                      env1.ctype, env2.ctype,
                      env1.unique_species, env2.unique_species,
                      sigm, lsm,
                      nspec, spec_mask, mb_mask)

    return two_term + three_term + many_term


# -----------------------------------------------------------------------------
#                        two plus three body kernels
# -----------------------------------------------------------------------------


def two_plus_three_body_mc(env1, env2, d1, d2, cutoff_2b, cutoff_3b, cutoff_mb,
                           nspec, spec_mask,
                           nbond, bond_mask, ntriplet, triplet_mask,
                           ncut3b, cut3b_mask,
                           nmb, mb_mask,
                           sig2, ls2, sig3, ls3, sigm, lsm,
                           cutoff_func=cf.quadratic_cutoff):
    """2+3-body multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body force/force kernel.
    """

    two_term = two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                               env2.bond_array_2, env2.ctype, env2.etypes,
                               d1, d2, sig2, ls2, cutoff_2b, cutoff_func,
                               nspec, spec_mask, bond_mask)

    if (ncut3b <= 1):
        tbmcj = three_body_mc_jit
    else:
        tbmcj = three_body_mc_sepcut_jit

    three_term = \
        tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
              env2.bond_array_3, env2.ctype, env2.etypes,
              env1.cross_bond_inds, env2.cross_bond_inds,
              env1.cross_bond_dists, env2.cross_bond_dists,
              env1.triplet_counts, env2.triplet_counts,
              d1, d2, sig3, ls3, cutoff_3b, cutoff_func,
              nspec, spec_mask, triplet_mask, cut3b_mask)

    return two_term + three_term


def two_plus_three_body_mc_grad(env1, env2, d1, d2, cutoff_2b, cutoff_3b, cutoff_mb,
                                nspec, spec_mask,
                                nbond, bond_mask, ntriplet, triplet_mask,
                                ncut3b, cut3b_mask,
                                nmb, mb_mask,
                                sig2, ls2, sig3, ls3, sigm, lsm,
                                cutoff_func=cf.quadratic_cutoff):
    """2+3-body multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        (float, np.ndarray):
            Value of the 2+3-body kernel and its gradient
            with respect to the hyperparameters.
    """

    kern2, grad2 = \
        two_body_mc_grad_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                             env2.bond_array_2, env2.ctype, env2.etypes,
                             d1, d2, sig2, ls2, cutoff_2b, cutoff_func,
                             nspec, spec_mask,
                             nbond, bond_mask)

    if (ncut3b == 0):
        tbmcj = three_body_mc_grad_jit
    else:
        tbmcj = three_body_mc_grad_sepcut_jit

    kern3, grad3 = \
        tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
              env2.bond_array_3, env2.ctype, env2.etypes,
              env1.cross_bond_inds, env2.cross_bond_inds,
              env1.cross_bond_dists, env2.cross_bond_dists,
              env1.triplet_counts, env2.triplet_counts,
              d1, d2, sig3, ls3, cutoff_3b,
              cutoff_func,
              nspec, spec_mask,
              ntriplet, triplet_mask, cut3b_mask)

    g = np.hstack([grad2, grad3])

    return kern2 + kern3, g


def two_plus_three_mc_force_en(env1, env2, d1, cutoff_2b, cutoff_3b, cutoff_mb,
                               nspec, spec_mask, nbond, bond_mask,
                               ntriplet, triplet_mask, ncut3b, cut3b_mask,
                               nmb, mb_mask,
                               sig2, ls2, sig3, ls3, sigm, lsm,
                               cutoff_func=cf.quadratic_cutoff):
    """2+3-body multi-element kernel between force and local energy

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body force/energy kernel.
    """

    two_term = \
        two_body_mc_force_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                 env2.bond_array_2, env2.ctype, env2.etypes,
                                 d1, sig2, ls2, cutoff_2b, cutoff_func,
                                 nspec, spec_mask,
                                 bond_mask) / 2

    if (ncut3b == 0):
        tbmcj = three_body_mc_force_en_jit
    else:
        tbmcj = three_body_mc_force_en_sepcut_jit

    three_term = \
        tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
              env2.bond_array_3, env2.ctype, env2.etypes,
              env1.cross_bond_inds, env2.cross_bond_inds,
              env1.cross_bond_dists,
              env2.cross_bond_dists,
              env1.triplet_counts, env2.triplet_counts,
              d1, sig3, ls3, cutoff_3b, cutoff_func,
              nspec, spec_mask,
              triplet_mask, cut3b_mask) / 3

    return two_term + three_term


def two_plus_three_mc_en(env1, env2, cutoff_2b, cutoff_3b, cutoff_mb,
                         nspec, spec_mask, nbond, bond_mask,
                         ntriplet, triplet_mask, ncut3b, cut3b_mask,
                         nmb, mb_mask,
                         sig2, ls2, sig3, ls3, sigm, lsm,
                         cutoff_func=cf.quadratic_cutoff):
    """2+3-body multi-element kernel between two local energies

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body energy/energy kernel.
    """

    two_term = two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                  env2.bond_array_2, env2.ctype, env2.etypes,
                                  sig2, ls2, cutoff_2b, cutoff_func,
                                  nspec,
                                  spec_mask,
                                  bond_mask) / 4

    if (ncut3b == 0):
        tbmcj = three_body_mc_en_jit
    else:
        tbmcj = three_body_mc_en_sepcut_jit

    three_term = \
        tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
              env2.bond_array_3, env2.ctype, env2.etypes,
              env1.cross_bond_inds, env2.cross_bond_inds,
              env1.cross_bond_dists, env2.cross_bond_dists,
              env1.triplet_counts, env2.triplet_counts,
              sig3, ls3, cutoff_3b, cutoff_func,
              nspec, spec_mask,
              triplet_mask, cut3b_mask)/9

    return two_term + three_term


# -----------------------------------------------------------------------------
#                      three body multicomponent kernel
# -----------------------------------------------------------------------------


def three_body_mc(env1, env2, d1, d2, cutoff_2b, cutoff_3b, cutoff_mb,
                  nspec, spec_mask, nbond, bond_mask,
                  ntriplet, triplet_mask, ncut3b, cut3b_mask,
                  nmb, mb_mask,
                  sig2, ls2, sig3, ls3, sigm, lsm,
                  cutoff_func=cf.quadratic_cutoff):
    """3-body multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b: dummy
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond: dummy
        bond_mask: dummy
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2: dummy
        ls2: dummy
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body force/force kernel.
    """

    if (ncut3b == 0):
        tbmcj = three_body_mc_jit
    else:
        tbmcj = three_body_mc_sepcut_jit

    return tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
                 env2.bond_array_3, env2.ctype, env2.etypes,
                 env1.cross_bond_inds, env2.cross_bond_inds,
                 env1.cross_bond_dists, env2.cross_bond_dists,
                 env1.triplet_counts, env2.triplet_counts,
                 d1, d2, sig3, ls3, cutoff_3b, cutoff_func,
                 nspec, spec_mask,
                 triplet_mask, cut3b_mask)


def three_body_mc_grad(env1, env2, d1, d2, cutoff_2b, cutoff_3b, cutoff_mb,
                       nspec, spec_mask, nbond, bond_mask,
                       ntriplet, triplet_mask, ncut3b, cut3b_mask,
                       nmb, mb_mask,
                       sig2, ls2, sig3, ls3, sigm, lsm,
                       cutoff_func=cf.quadratic_cutoff):
    """3-body multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b: dummy
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond: dummy
        bond_mask: dummy
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2: dummy
        ls2: dummy
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        (float, np.ndarray):
            Value of the 2+3+manybody kernel and its gradient
            with respect to the hyperparameters.
    """

    if (ncut3b == 0):
        tbmcj = three_body_mc_grad_jit
    else:
        tbmcj = three_body_mc_grad_sepcut_jit

    return tbmcj(
        env1.bond_array_3, env1.ctype, env1.etypes,
        env2.bond_array_3, env2.ctype, env2.etypes,
        env1.cross_bond_inds, env2.cross_bond_inds,
        env1.cross_bond_dists, env2.cross_bond_dists,
        env1.triplet_counts, env2.triplet_counts,
        d1, d2, sig3, ls3, cutoff_3b, cutoff_func,
        nspec, spec_mask, ntriplet, triplet_mask, cut3b_mask)


def three_body_mc_force_en(env1, env2, d1, cutoff_2b, cutoff_3b, cutoff_mb,
                           nspec, spec_mask, nbond, bond_mask, ntriplet, triplet_mask,
                           ncut3b, cut3b_mask, nmb, mb_mask,
                           sig2, ls2, sig3, ls3, sigm, lsm,
                           cutoff_func=cf.quadratic_cutoff):
    """3-body multi-element kernel between a force component and local energies

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b: dummy
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond: dummy
        bond_mask: dummy
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2: dummy
        ls2: dummy
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body force/energy kernel.
    """

    if (ncut3b == 0):
        tbmcj = three_body_mc_force_en_jit
    else:
        tbmcj = three_body_mc_force_en_sepcut_jit

    return tbmcj(env1.bond_array_3, env1.ctype,
                 env1.etypes,
                 env2.bond_array_3, env2.ctype,
                 env2.etypes,
                 env1.cross_bond_inds,
                 env2.cross_bond_inds,
                 env1.cross_bond_dists,
                 env2.cross_bond_dists,
                 env1.triplet_counts,
                 env2.triplet_counts,
                 d1, sig3, ls3, cutoff_3b,
                 cutoff_func,
                 nspec,
                 spec_mask,
                 triplet_mask, cut3b_mask) / 3


def three_body_mc_en(env1, env2, cutoff_2b, cutoff_3b,  cutoff_mb,  nspec, spec_mask,
                     nbond, bond_mask, ntriplet, triplet_mask,
                     ncut3b, cut3b_mask, nmb, mb_mask,
                     sig2, ls2, sig3, ls3, sigm, lsm,
                     cutoff_func=cf.quadratic_cutoff):
    """3-body multi-element kernel between two local energies

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b: dummy
        cutoff_3b (float, np.ndarray): cutoff(s) for three-body interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond: dummy
        bond_mask: dummy
        ntriplet (int): number of different hyperparameter sets to associate with 3-body pairings
        triplet_mask (np.ndarray): nspec^3 long integer array
        ncut3b (int): number of different 3-body cutoff sets to associate with 3-body pairings
        cut3b_mask (np.ndarray): nspec^2 long integer array
        sig2: dummy
        ls2: dummy
        sig3 (np.ndarray): signal variances associates with three-body term
        ls3 (np.ndarray): length scales associates with three-body term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body energy/energy kernel.
    """

    if (ncut3b == 0):
        tbmcj = three_body_mc_en_jit
    else:
        tbmcj = three_body_mc_en_sepcut_jit

    return tbmcj(env1.bond_array_3, env1.ctype, env1.etypes,
                 env2.bond_array_3, env2.ctype, env2.etypes,
                 env1.cross_bond_inds, env2.cross_bond_inds,
                 env1.cross_bond_dists, env2.cross_bond_dists,
                 env1.triplet_counts, env2.triplet_counts,
                 sig3, ls3, cutoff_3b, cutoff_func,
                 nspec, spec_mask,
                 triplet_mask, cut3b_mask)/9


# -----------------------------------------------------------------------------
#                       two body multicomponent kernel
# -----------------------------------------------------------------------------


def two_body_mc(
        env1, env2, d1, d2, cutoff_2b, cutoff_3b,  cutoff_mb,  nspec, spec_mask,
        nbond, bond_mask, ntriplet, triplet_mask, ncut3b, cut3b_mask,
        nmb, mb_mask, sig2, ls2, sig3, ls3, sigm, lsm,
        cutoff_func=cf.quadratic_cutoff):
    """2-body multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b: dummy
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet: dummy
        triplet_mask: dummy
        ncut3b: dummy
        cut3b_mask: dummy
        sig2: dummy
        ls2: dummy
        sig3: dummy
        ls3: dummy
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2-body force/force kernel.
    """

    return two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                           env2.bond_array_2, env2.ctype, env2.etypes,
                           d1, d2, sig2, ls2, cutoff_2b, cutoff_func,
                           nspec, spec_mask, bond_mask)


def two_body_mc_grad(
        env1, env2, d1, d2, cutoff_2b, cutoff_3b,  cutoff_mb,  nspec, spec_mask,
        nbond, bond_mask, ntriplet, triplet_mask,
        ncut3b, cut3b_mask, nmb, mb_mask,
        sig2, ls2, sig3, ls3, sigm, lsm,
        cutoff_func=cf.quadratic_cutoff):
    """2-body multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b: dummy
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet: dummy
        triplet_mask: dummy
        ncut3b: dummy
        cut3b_mask: dummy
        sig2 (np.ndarray): signal variances associates with two-body term
        ls2 (np.ndarray): length scales associates with two-body term
        sig3: dummy
        ls3: dummy
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        (float, np.ndarray):
            Value of the 2-body kernel and its gradient
            with respect to the hyperparameters.
    """

    return two_body_mc_grad_jit(
        env1.bond_array_2, env1.ctype, env1.etypes,
        env2.bond_array_2, env2.ctype, env2.etypes,
        d1, d2, sig2, ls2, cutoff_2b, cutoff_func,
        nspec, spec_mask, nbond, bond_mask)


def two_body_mc_force_en(env1, env2, d1, cutoff_2b, cutoff_3b, cutoff_mb,
                         nspec, spec_mask, nbond, bond_mask, ntriplet, triplet_mask,
                         ncut3b, cut3b_mask, nmb, mb_mask,
                         sig2, ls2, sig3, ls3, sigm, lsm,
                         cutoff_func=cf.quadratic_cutoff):
    """2-body multi-element kernel between a force components and local energy

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b: dummy
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet: dummy
        triplet_mask: dummy
        ncut3b: dummy
        cut3b_mask: dummy
        sig2: dummy
        ls2: dummy
        sig3: dummy
        ls3: dummy
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2-body force/energy kernel.
    """

    return two_body_mc_force_en_jit(
        env1.bond_array_2, env1.ctype, env1.etypes,
        env2.bond_array_2, env2.ctype, env2.etypes,
        d1, sig2, ls2, cutoff_2b, cutoff_func,
        nspec, spec_mask, bond_mask) / 2


def two_body_mc_en(env1, env2, cutoff_2b, cutoff_3b, cutoff_mb,
                   nspec, spec_mask,
                   nbond, bond_mask, ntriplet, triplet_mask,
                   ncut3b, cut3b_mask, nmb, mb_mask,
                   sig2, ls2, sig3, ls3, sigm, lsm,
                   cutoff_func=cf.quadratic_cutoff):
    """2-body multi-element kernel between two local energies

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b (float, np.ndarray): cutoff(s) for two-body interaction
        cutoff_3b: dummy
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond (int): number of different hyperparameter sets to associate with two-body pairings
        bond_mask (np.ndarray): nspec^2 long integer array
        ntriplet: dummy
        triplet_mask: dummy
        ncut3b: dummy
        cut3b_mask: dummy
        sig2: dummy
        ls2: dummy
        sig3: dummy
        ls3: dummy
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2-body energy/energy kernel.
    """

    return two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                              env2.bond_array_2, env2.ctype, env2.etypes,
                              sig2, ls2, cutoff_2b, cutoff_func,
                              nspec, spec_mask, bond_mask)/4


# -----------------------------------------------------------------------------
#                 three body multicomponent kernel (numba)
# -----------------------------------------------------------------------------

@njit
def three_body_mc_jit(bond_array_1, c1, etypes1,
                      bond_array_2, c2, etypes2,
                      cross_bond_inds_1, cross_bond_inds_2,
                      cross_bond_dists_1, cross_bond_dists_2,
                      triplets_1, triplets_2,
                      d1, d2, sig, ls, r_cut, cutoff_func,
                      nspec, spec_mask, triplet_mask, cut3b_mask):
    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec * nspec

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = nspec * bei1

        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ei2 = etypes1[ind1]

            tr_spec = [c1, ei1, ei2]
            c2_ind = tr_spec
            if c2 in tr_spec:
                tr_spec.remove(c2)

                ri2 = bond_array_1[ind1, 0]
                ci2 = bond_array_1[ind1, d1]
                fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)

                bei2 = spec_mask[ei2]

                ri3 = cross_bond_dists_1[m, m + n + 1]
                fi3, _ = cutoff_func(r_cut, ri3, 0)

                fi = fi1 * fi2 * fi3
                fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

                ttypei = triplet_mask[bc1n + bei1n + bei2]

                tls1 = ls1[ttypei]
                tls2 = ls2[ttypei]
                tls3 = ls3[ttypei]
                tsig2 = sig2[ttypei]

                for p in range(bond_array_2.shape[0]):
                    ej1 = etypes2[p]
                    tr_spec1 = [tr_spec[0], tr_spec[1]]
                    if ej1 in tr_spec1:
                        tr_spec1.remove(ej1)
                        rj1 = bond_array_2[p, 0]
                        cj1 = bond_array_2[p, d2]
                        fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)

                        for q in range(triplets_2[p]):
                            ind2 = cross_bond_inds_2[p, p + 1 + q]
                            ej2 = etypes2[ind2]
                            if ej2 == tr_spec1[0]:
                                ind2 = cross_bond_inds_2[p, p + 1 + q]
                                rj2 = bond_array_2[ind2, 0]
                                cj2 = bond_array_2[ind2, d2]
                                fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                                ej2 = etypes2[ind2]

                                rj3 = cross_bond_dists_2[p, p + 1 + q]
                                fj3, _ = cutoff_func(r_cut, rj3, 0)

                                fj = fj1 * fj2 * fj3
                                fdj = fdj1 * fj2 * fj3 + fj1 * fdj2 * fj3

                                r11 = ri1 - rj1
                                r12 = ri1 - rj2
                                r13 = ri1 - rj3
                                r21 = ri2 - rj1
                                r22 = ri2 - rj2
                                r23 = ri2 - rj3
                                r31 = ri3 - rj1
                                r32 = ri3 - rj2
                                r33 = ri3 - rj3

                                if (c1 == c2):
                                    if (ei1 == ej1) and (ei2 == ej2):
                                        kern += \
                                            three_body_helper_1(ci1, ci2, cj1, cj2, r11,
                                                                r22, r33, fi, fj, fdi, fdj,
                                                                tls1, tls2, tls3,
                                                                tsig2)
                                    if (ei1 == ej2) and (ei2 == ej1):
                                        kern += \
                                            three_body_helper_1(ci1, ci2, cj2, cj1, r12,
                                                                r21, r33, fi, fj, fdi, fdj,
                                                                tls1, tls2, tls3,
                                                                tsig2)
                                if (c1 == ej1):
                                    if (ei1 == ej2) and (ei2 == c2):
                                        kern += \
                                            three_body_helper_2(ci2, ci1, cj2, cj1, r21,
                                                                r13, r32, fi, fj, fdi,
                                                                fdj,
                                                                tls1, tls2, tls3,
                                                                tsig2)
                                    if (ei1 == c2) and (ei2 == ej2):
                                        kern += \
                                            three_body_helper_2(ci1, ci2, cj2, cj1, r11,
                                                                r23, r32, fi, fj, fdi,
                                                                fdj,
                                                                tls1, tls2, tls3,
                                                                tsig2)
                                if (c1 == ej2):
                                    if (ei1 == ej1) and (ei2 == c2):
                                        kern += \
                                            three_body_helper_2(ci2, ci1, cj1, cj2, r22,
                                                                r13, r31, fi, fj, fdi,
                                                                fdj,
                                                                tls1, tls2, tls3,
                                                                tsig2)
                                    if (ei1 == c2) and (ei2 == ej1):
                                        kern += \
                                            three_body_helper_2(ci1, ci2, cj1, cj2, r12,
                                                                r23, r31, fi, fj, fdi,
                                                                fdj,
                                                                tls1, tls2, tls3,
                                                                tsig2)

    return kern


@njit
def three_body_mc_grad_jit(bond_array_1, c1, etypes1,
                           bond_array_2, c2, etypes2,
                           cross_bond_inds_1, cross_bond_inds_2,
                           cross_bond_dists_1, cross_bond_dists_2,
                           triplets_1, triplets_2,
                           d1, d2, sig, ls, r_cut, cutoff_func,
                           nspec, spec_mask, ntriplet, triplet_mask,
                           cut3b_mask):
    """Kernel gradient for 3-body force comparisons."""

    kern = 0
    sig_derv = np.zeros(ntriplet, dtype=np.float64)
    ls_derv = np.zeros(ntriplet, dtype=np.float64)

    # pre-compute constants that appear in the inner loop
    sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6 = grad_constants(sig, ls)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec * nspec

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = bei1 * nspec

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ei2 = etypes1[ind1]
            tr_spec = [c1, ei1, ei2]
            c2_ind = tr_spec
            if c2 in tr_spec:
                tr_spec.remove(c2)

                ri3 = cross_bond_dists_1[m, m + n + 1]
                ri2 = bond_array_1[ind1, 0]
                ci2 = bond_array_1[ind1, d1]

                bei2 = spec_mask[ei2]

                ttypei = triplet_mask[bc1n + bei1n + bei2]

                tls1 = ls1[ttypei]
                tls2 = ls2[ttypei]
                tls3 = ls3[ttypei]
                tls4 = ls4[ttypei]
                tls5 = ls5[ttypei]
                tls6 = ls6[ttypei]
                tsig2 = sig2[ttypei]
                tsig3 = sig3[ttypei]

                fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                fi3, _ = cutoff_func(r_cut, ri3, 0)

                fi = fi1 * fi2 * fi3
                fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

                for p in range(bond_array_2.shape[0]):
                    ej1 = etypes2[p]
                    tr_spec1 = [tr_spec[0], tr_spec[1]]
                    if ej1 in tr_spec1:
                        tr_spec1.remove(ej1)
                        rj1 = bond_array_2[p, 0]
                        cj1 = bond_array_2[p, d2]
                        fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)

                        for q in range(triplets_2[p]):
                            ind2 = cross_bond_inds_2[p, p + 1 + q]
                            ej2 = etypes2[ind2]
                            if ej2 == tr_spec1[0]:
                                ind2 = cross_bond_inds_2[p, p + q + 1]
                                rj3 = cross_bond_dists_2[p, p + q + 1]
                                rj2 = bond_array_2[ind2, 0]
                                cj2 = bond_array_2[ind2, d2]
                                ej2 = etypes2[ind2]

                                fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                                fj3, _ = cutoff_func(r_cut, rj3, 0)

                                fj = fj1 * fj2 * fj3
                                fdj = fdj1 * fj2 * fj3 + fj1 * fdj2 * fj3

                                r11 = ri1 - rj1
                                r12 = ri1 - rj2
                                r13 = ri1 - rj3
                                r21 = ri2 - rj1
                                r22 = ri2 - rj2
                                r23 = ri2 - rj3
                                r31 = ri3 - rj1
                                r32 = ri3 - rj2
                                r33 = ri3 - rj3

                                if (c1 == c2):
                                    if (ei1 == ej1) and (ei2 == ej2):
                                        kern_term, sig_term, ls_term = \
                                            three_body_grad_helper_1(ci1, ci2, cj1, cj2,
                                                                     r11, r22, r33, fi, fj,
                                                                     fdi, fdj, tls1, tls2,
                                                                     tls3, tls4, tls5,
                                                                     tls6,
                                                                     tsig2, tsig3)
                                        kern += kern_term
                                        sig_derv[ttypei] += sig_term
                                        ls_derv[ttypei] += ls_term

                                    if (ei1 == ej2) and (ei2 == ej1):
                                        kern_term, sig_term, ls_term = \
                                            three_body_grad_helper_1(ci1, ci2, cj2, cj1,
                                                                     r12, r21, r33, fi, fj,
                                                                     fdi, fdj, tls1, tls2,
                                                                     tls3, tls4, tls5,
                                                                     tls6,
                                                                     tsig2, tsig3)
                                        kern += kern_term
                                        sig_derv[ttypei] += sig_term
                                        ls_derv[ttypei] += ls_term

                                if (c1 == ej1):
                                    if (ei1 == ej2) and (ei2 == c2):
                                        kern_term, sig_term, ls_term = \
                                            three_body_grad_helper_2(ci2, ci1, cj2, cj1,
                                                                     r21, r13, r32, fi, fj,
                                                                     fdi, fdj, tls1, tls2,
                                                                     tls3, tls4, tls5,
                                                                     tls6,
                                                                     tsig2, tsig3)
                                        kern += kern_term
                                        sig_derv[ttypei] += sig_term
                                        ls_derv[ttypei] += ls_term

                                    if (ei1 == c2) and (ei2 == ej2):
                                        kern_term, sig_term, ls_term = \
                                            three_body_grad_helper_2(ci1, ci2, cj2, cj1,
                                                                     r11, r23, r32, fi, fj,
                                                                     fdi, fdj, tls1, tls2,
                                                                     tls3, tls4, tls5,
                                                                     tls6,
                                                                     tsig2, tsig3)
                                        kern += kern_term
                                        sig_derv[ttypei] += sig_term
                                        ls_derv[ttypei] += ls_term

                                if (c1 == ej2):
                                    if (ei1 == ej1) and (ei2 == c2):
                                        kern_term, sig_term, ls_term = \
                                            three_body_grad_helper_2(ci2, ci1, cj1, cj2,
                                                                     r22, r13, r31, fi, fj,
                                                                     fdi, fdj, tls1, tls2,
                                                                     tls3, tls4, tls5,
                                                                     tls6,
                                                                     tsig2, tsig3)
                                        kern += kern_term
                                        sig_derv[ttypei] += sig_term
                                        ls_derv[ttypei] += ls_term

                                    if (ei1 == c2) and (ei2 == ej1):
                                        kern_term, sig_term, ls_term = \
                                            three_body_grad_helper_2(ci1, ci2, cj1, cj2,
                                                                     r12, r23, r31, fi, fj,
                                                                     fdi, fdj, tls1, tls2,
                                                                     tls3, tls4, tls5,
                                                                     tls6,
                                                                     tsig2, tsig3)

                                        kern += kern_term
                                        sig_derv[ttypei] += sig_term
                                        ls_derv[ttypei] += ls_term

    return kern, np.hstack((sig_derv, ls_derv))


@njit
def three_body_mc_force_en_jit(bond_array_1, c1, etypes1,
                               bond_array_2, c2, etypes2,
                               cross_bond_inds_1, cross_bond_inds_2,
                               cross_bond_dists_1, cross_bond_dists_2,
                               triplets_1, triplets_2,
                               d1, sig, ls, r_cut, cutoff_func,
                               nspec, spec_mask, triplet_mask, cut3b_mask):
    """Kernel for 3-body force/energy comparisons."""

    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

    bc1 = spec_mask[c1]
    bc1n = nspec * nspec * bc1

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = nspec * bei1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ei2 = etypes1[ind1]
            tr_spec = [c1, ei1, ei2]
            c2_ind = tr_spec
            if c2 in tr_spec:
                tr_spec.remove(c2)
                ri2 = bond_array_1[ind1, 0]
                ci2 = bond_array_1[ind1, d1]
                fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)

                bei2 = spec_mask[ei2]

                ttypei = triplet_mask[bc1n + bei1n + bei2]

                tls1 = ls1[ttypei]
                tls2 = ls2[ttypei]
                tsig2 = sig2[ttypei]

                ri3 = cross_bond_dists_1[m, m + n + 1]
                fi3, _ = cutoff_func(r_cut, ri3, 0)

                fi = fi1 * fi2 * fi3
                fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

                for p in range(bond_array_2.shape[0]):

                    ej1 = etypes2[p]
                    tr_spec1 = [tr_spec[0], tr_spec[1]]
                    if ej1 in tr_spec1:
                        tr_spec1.remove(ej1)
                        rj1 = bond_array_2[p, 0]
                        fj1, _ = cutoff_func(r_cut, rj1, 0)

                        for q in range(triplets_2[p]):

                            ind2 = cross_bond_inds_2[p, p + 1 + q]
                            ej2 = etypes2[ind2]
                            if ej2 == tr_spec1[0]:

                                ind2 = cross_bond_inds_2[p, p + q + 1]
                                rj2 = bond_array_2[ind2, 0]
                                fj2, _ = cutoff_func(r_cut, rj2, 0)
                                ej2 = etypes2[ind2]
                                rj3 = cross_bond_dists_2[p, p + q + 1]
                                fj3, _ = cutoff_func(r_cut, rj3, 0)
                                fj = fj1 * fj2 * fj3

                                r11 = ri1 - rj1
                                r12 = ri1 - rj2
                                r13 = ri1 - rj3
                                r21 = ri2 - rj1
                                r22 = ri2 - rj2
                                r23 = ri2 - rj3
                                r31 = ri3 - rj1
                                r32 = ri3 - rj2
                                r33 = ri3 - rj3

                                if (c1 == c2):
                                    if (ei1 == ej1) and (ei2 == ej2):
                                        kern += three_body_en_helper(ci1, ci2, r11, r22,
                                                                     r33, fi, fj, fdi,
                                                                     tls1,
                                                                     tls2, tsig2)
                                    if (ei1 == ej2) and (ei2 == ej1):
                                        kern += three_body_en_helper(ci1, ci2, r12, r21,
                                                                     r33, fi, fj, fdi,
                                                                     tls1,
                                                                     tls2, tsig2)
                                if (c1 == ej1):
                                    if (ei1 == ej2) and (ei2 == c2):
                                        kern += three_body_en_helper(ci1, ci2, r13, r21,
                                                                     r32, fi, fj, fdi,
                                                                     tls1,
                                                                     tls2, tsig2)
                                    if (ei1 == c2) and (ei2 == ej2):
                                        kern += three_body_en_helper(ci1, ci2, r11, r23,
                                                                     r32, fi, fj, fdi,
                                                                     tls1,
                                                                     tls2, tsig2)
                                if (c1 == ej2):
                                    if (ei1 == ej1) and (ei2 == c2):
                                        kern += three_body_en_helper(ci1, ci2, r13, r22,
                                                                     r31, fi, fj, fdi,
                                                                     tls1,
                                                                     tls2, tsig2)
                                    if (ei1 == c2) and (ei2 == ej1):
                                        kern += three_body_en_helper(ci1, ci2, r12, r23,
                                                                     r31, fi, fj, fdi,
                                                                     tls1,
                                                                     tls2, tsig2)

    return kern


@njit
def three_body_mc_en_jit(bond_array_1, c1, etypes1,
                         bond_array_2, c2, etypes2,
                         cross_bond_inds_1, cross_bond_inds_2,
                         cross_bond_dists_1, cross_bond_dists_2,
                         triplets_1, triplets_2,
                         sig, ls, r_cut, cutoff_func,
                         nspec, spec_mask, triplet_mask, cut3b_mask):
    kern = 0

    sig2 = sig * sig
    ls2 = 1 / (2 * ls * ls)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec * nspec

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1, _ = cutoff_func(r_cut, ri1, 0)
        ei1 = etypes1[m]

        bei1 = spec_mask[ei1]
        bei1n = nspec * bei1

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ei2 = etypes1[ind1]
            tr_spec = [c1, ei1, ei2]
            c2_ind = tr_spec
            if c2 in tr_spec:
                tr_spec.remove(c2)
                ri2 = bond_array_1[ind1, 0]
                fi2, _ = cutoff_func(r_cut, ri2, 0)

                bei2 = spec_mask[ei2]

                ttypei = triplet_mask[bc1n + bei1n + bei2]

                tls2 = ls2[ttypei]
                tsig2 = sig2[ttypei]

                ri3 = cross_bond_dists_1[m, m + n + 1]
                fi3, _ = cutoff_func(r_cut, ri3, 0)
                fi = fi1 * fi2 * fi3

                for p in range(bond_array_2.shape[0]):
                    ej1 = etypes2[p]
                    tr_spec1 = [tr_spec[0], tr_spec[1]]
                    if ej1 in tr_spec1:
                        tr_spec1.remove(ej1)
                        rj1 = bond_array_2[p, 0]
                        fj1, _ = cutoff_func(r_cut, rj1, 0)

                        for q in range(triplets_2[p]):
                            ind2 = cross_bond_inds_2[p, p + 1 + q]
                            ej2 = etypes2[ind2]
                            if ej2 == tr_spec1[0]:
                                rj2 = bond_array_2[ind2, 0]
                                fj2, _ = cutoff_func(r_cut, rj2, 0)

                                rj3 = cross_bond_dists_2[p, p + q + 1]
                                fj3, _ = cutoff_func(r_cut, rj3, 0)
                                fj = fj1 * fj2 * fj3

                                r11 = ri1 - rj1
                                r12 = ri1 - rj2
                                r13 = ri1 - rj3
                                r21 = ri2 - rj1
                                r22 = ri2 - rj2
                                r23 = ri2 - rj3
                                r31 = ri3 - rj1
                                r32 = ri3 - rj2
                                r33 = ri3 - rj3

                                if (c1 == c2):
                                    if (ei1 == ej1) and (ei2 == ej2):
                                        C1 = r11 * r11 + r22 * r22 + r33 * r33
                                        kern += tsig2 * \
                                            exp(-C1 * tls2) * fi * fj
                                    if (ei1 == ej2) and (ei2 == ej1):
                                        C3 = r12 * r12 + r21 * r21 + r33 * r33
                                        kern += tsig2 * \
                                            exp(-C3 * tls2) * fi * fj
                                if (c1 == ej1):
                                    if (ei1 == ej2) and (ei2 == c2):
                                        C5 = r13 * r13 + r21 * r21 + r32 * r32
                                        kern += tsig2 * \
                                            exp(-C5 * tls2) * fi * fj
                                    if (ei1 == c2) and (ei2 == ej2):
                                        C2 = r11 * r11 + r23 * r23 + r32 * r32
                                        kern += tsig2 * \
                                            exp(-C2 * tls2) * fi * fj
                                if (c1 == ej2):
                                    if (ei1 == ej1) and (ei2 == c2):
                                        C6 = r13 * r13 + r22 * r22 + r31 * r31
                                        kern += tsig2 * \
                                            exp(-C6 * tls2) * fi * fj
                                    if (ei1 == c2) and (ei2 == ej1):
                                        C4 = r12 * r12 + r23 * r23 + r31 * r31
                                        kern += tsig2 * \
                                            exp(-C4 * tls2) * fi * fj

    return kern


# -----------------------------------------------------------------------------
#                 two body multicomponent kernel (numba)
# -----------------------------------------------------------------------------


@njit
def two_body_mc_jit(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                    d1, d2, sig, ls, r_cut, cutoff_func,
                    nspec, spec_mask, bond_mask):
    """Multicomponent two-body force/force kernel accelerated with Numba's
    njit decorator.
    Loops over bonds in two environments and adds to the kernel if bonds are
    of the same type.
    """

    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec

    for m in range(bond_array_1.shape[0]):
        e1 = etypes1[m]

        if ((c2 == e1) or (c2 == c1)):
            ri = bond_array_1[m, 0]
            ci = bond_array_1[m, d1]

            be1 = spec_mask[e1]
            btype = bond_mask[bc1n + be1]

            tls1 = ls1[btype]
            tls2 = ls2[btype]
            tls3 = ls3[btype]
            tsig2 = sig2[btype]
            tr_cut = r_cut[btype]

            fi, fdi = cutoff_func(tr_cut, ri, ci)

            for n in range(bond_array_2.shape[0]):
                e2 = etypes2[n]

                # check if bonds agree
                if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                    rj = bond_array_2[n, 0]
                    cj = bond_array_2[n, d2]
                    fj, fdj = cutoff_func(tr_cut, rj, cj)
                    r11 = ri - rj

                    A = ci * cj
                    B = r11 * ci
                    C = r11 * cj
                    D = r11 * r11

                    kern += force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                         tls1, tls2, tls3, tsig2)

    return kern


@njit
def two_body_mc_grad_jit(bond_array_1, c1, etypes1,
                         bond_array_2, c2, etypes2,
                         d1, d2, sig, ls, r_cut, cutoff_func,
                         nspec, spec_mask, nbond, bond_mask):
    """Multicomponent two-body force/force kernel gradient accelerated with
    Numba's njit decorator."""

    kern = 0
    sig_derv = np.zeros(nbond, dtype=np.float64)
    ls_derv = np.zeros(nbond, dtype=np.float64)

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    ls4 = 1 / (ls * ls * ls)
    ls5 = ls * ls
    ls6 = ls2 * ls4

    sig2 = sig * sig
    sig3 = 2 * sig

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec

    for m in range(bond_array_1.shape[0]):
        e1 = etypes1[m]
        if ((c2 == e1) or (c2 == c1)):
            ri = bond_array_1[m, 0]
            ci = bond_array_1[m, d1]

            be1 = spec_mask[e1]
            btype = bond_mask[bc1n + be1]

            tls1 = ls1[btype]
            tls2 = ls2[btype]
            tls3 = ls3[btype]
            tls4 = ls4[btype]
            tls5 = ls5[btype]
            tls6 = ls6[btype]
            tsig2 = sig2[btype]
            tsig3 = sig3[btype]
            tr_cut = r_cut[btype]

            fi, fdi = cutoff_func(tr_cut, ri, ci)

            for n in range(bond_array_2.shape[0]):
                e2 = etypes2[n]

                # check if bonds agree
                if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                    rj = bond_array_2[n, 0]
                    cj = bond_array_2[n, d2]
                    fj, fdj = cutoff_func(tr_cut, rj, cj)

                    r11 = ri - rj

                    A = ci * cj
                    B = r11 * ci
                    C = r11 * cj
                    D = r11 * r11

                    kern_term, sig_term, ls_term = \
                        grad_helper(A, B, C, D, fi, fj, fdi, fdj,
                                    tls1, tls2, tls3,
                                    tls4, tls5, tls6,
                                    tsig2, tsig3)

                    kern += kern_term
                    sig_derv[btype] += sig_term
                    ls_derv[btype] += ls_term

    kern_grad = np.hstack((sig_derv, ls_derv))

    return kern, kern_grad


@njit
def two_body_mc_force_en_jit(bond_array_1, c1, etypes1,
                             bond_array_2, c2, etypes2,
                             d1, sig, ls, r_cut, cutoff_func,
                             nspec, spec_mask, bond_mask):
    """Multicomponent two-body force/energy kernel accelerated with
    Numba's njit decorator."""

    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    sig2 = sig * sig

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec

    for m in range(bond_array_1.shape[0]):
        e1 = etypes1[m]
        if ((c2 == e1) or (c2 == c1)):
            ri = bond_array_1[m, 0]
            ci = bond_array_1[m, d1]

            be1 = spec_mask[e1]
            btype = bond_mask[bc1n + be1]

            tls1 = ls1[btype]
            tls2 = ls2[btype]
            tsig2 = sig2[btype]
            tr_cut = r_cut[btype]

            fi, fdi = cutoff_func(tr_cut, ri, ci)

            for n in range(bond_array_2.shape[0]):
                e2 = etypes2[n]

                # check if bonds agree
                if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                    rj = bond_array_2[n, 0]
                    fj, _ = cutoff_func(tr_cut, rj, 0)

                    r11 = ri - rj
                    B = r11 * ci
                    D = r11 * r11
                    kern += force_energy_helper(B, D, fi, fj, fdi,
                                                tls1, tls2,
                                                tsig2)

    return kern


@njit
def two_body_mc_en_jit(bond_array_1, c1, etypes1,
                       bond_array_2, c2, etypes2,
                       sig, ls, r_cut, cutoff_func,
                       nspec, spec_mask, bond_mask):
    """Multicomponent two-body energy/energy kernel accelerated with
    Numba's njit decorator."""

    kern = 0

    ls1 = 1 / (2 * ls * ls)
    sig2 = sig * sig

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec

    for m in range(bond_array_1.shape[0]):
        e1 = etypes1[m]
        if ((c2 == e1) or (c2 == c1)):
            ri = bond_array_1[m, 0]

            be1 = spec_mask[e1]
            btype = bond_mask[bc1n + be1]

            tls1 = ls1[btype]
            tsig2 = sig2[btype]
            tr_cut = r_cut[btype]
            fi, _ = cutoff_func(tr_cut, ri, 0)

            for n in range(bond_array_2.shape[0]):
                e2 = etypes2[n]

                if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                    rj = bond_array_2[n, 0]
                    fj, _ = cutoff_func(tr_cut, rj, 0)
                    r11 = ri - rj
                    kern += fi * fj * tsig2 * exp(-r11 * r11 * tls1)

    return kern


def many_body_mc(env1, env2, d1, d2, cutoff_2b, cutoff_3b, cutoff_mb,
                 nspec, spec_mask,
                 nbond, bond_mask, ntriplet, triplet_mask,
                 ncut3b, cut3b_mask,
                 nmb, mb_mask,
                 sig2, ls2, sig3, ls3, sigm, lsm,
                 cutoff_func=cf.quadratic_cutoff):
    """many-body multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b: dummy
        cutoff_3b: dummy
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond: dummy
        bond_mask: dummy
        ntriplet: dummy
        triplet_mask: dummy
        ncut3b: dummy
        cut3b_mask: dummy
        nmb (int): number of different hyperparameter sets to associate with manybody pairings
        mb_mask (np.ndarray): nspec^2 long integer array
        sig2: dummy
        ls2: dummy
        sig3: dummy
        ls3: dummy
        sigm (np.ndarray): signal variances associates with manybody term
        lsm (np.ndarray): length scales associates with manybody term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3+many-body kernel.
    """
    return many_body_mc_sepcut_jit(env1.q_array, env2.q_array,
                                   env1.q_neigh_array, env2.q_neigh_array,
                                   env1.q_neigh_grads, env2.q_neigh_grads,
                                   env1.ctype, env2.ctype,
                                   env1.etypes_mb, env2.etypes_mb,
                                   env1.unique_species, env2.unique_species,
                                   d1, d2, sigm, lsm,
                                   nspec, spec_mask, mb_mask)


def many_body_mc_grad(env1, env2, d1, d2, cutoff_2b, cutoff_3b, cutoff_mb,
                      nspec, spec_mask,
                      nbond, bond_mask, ntriplet, triplet_mask,
                      ncut3b, cut3b_mask,
                      nmb, mb_mask,
                      sig2, ls2, sig3, ls3, sigm, lsm,
                      cutoff_func=cf.quadratic_cutoff):
    """manybody multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        cutoff_2b: dummy
        cutoff_3b: dummy
        cutoff_mb (float, np.ndarray): cutoff(s) for coordination-based manybody interaction
        nspec (int): number of different species groups
        spec_mask (np.ndarray): 118-long integer array that determines specie group
        nbond: dummy
        bond_mask: dummy
        ntriplet: dummy
        triplet_mask: dummy
        ncut3b: dummy
        cut3b_mask: dummy
        nmb (int): number of different hyperparameter sets to associate with manybody pairings
        mb_mask (np.ndarray): nspec^2 long integer array
        sig2: dummy
        ls2: dummy
        sig3: dummy
        ls3: dummy
        sigm (np.ndarray): signal variances associates with manybody term
        lsm (np.ndarray): length scales associates with manybody term
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        (float, np.ndarray):
            Value of the 2+3+manybody kernel and its gradient
            with respect to the hyperparameters.
    """

    return many_body_mc_grad_sepcut_jit(env1.q_array, env2.q_array,
                                        env1.q_neigh_array, env2.q_neigh_array,
                                        env1.q_neigh_grads, env2.q_neigh_grads,
                                        env1.ctype, env2.ctype,
                                        env1.etypes_mb, env2.etypes_mb,
                                        env1.unique_species, env2.unique_species,
                                        d1, d2, sigm, lsm,
                                        nspec, spec_mask, nmb, mb_mask)


def many_body_mc_force_en(env1, env2, d1, cutoff_2b, cutoff_3b, cutoff_mb,
                          nspec, spec_mask,
                          nbond, bond_mask, ntriplet, triplet_mask,
                          ncut3b, cut3b_mask,
                          nmb, mb_mask,
                          sig2, ls2, sig3, ls3, sigm, lsm,
                          cutoff_func=cf.quadratic_cutoff):
    """many-body single-element kernel between two local energies.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): Two-element array containing the 2-, 3-, and
            many-body cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the many-body force/energy kernel.
    """

    return many_body_mc_force_en_sepcut_jit(env1.q_array, env2.q_array,
                                            env1.q_neigh_array,
                                            env1.q_neigh_grads,
                                            env1.ctype, env2.ctype,
                                            env1.etypes_mb,
                                            env1.unique_species, env2.unique_species,
                                            d1, sigm, lsm,
                                            nspec, spec_mask, mb_mask)


def many_body_mc_en(env1, env2, cutoff_2b, cutoff_3b, cutoff_mb,
                    nspec, spec_mask,
                    nbond, bond_mask, ntriplet, triplet_mask,
                    ncut3b, cut3b_mask,
                    nmb, mb_mask,
                    sig2, ls2, sig3, ls3, sigm, lsm,
                    cutoff_func=cf.quadratic_cutoff):
    """many-body multi-element kernel between two local energies.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): One-element array containing the 2-body
            cutoff.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2-body force/energy kernel.
    """

    return many_body_mc_en_sepcut_jit(env1.q_array, env2.q_array,
                                      env1.ctype, env2.ctype,
                                      env1.unique_species, env2.unique_species,
                                      sigm, lsm,
                                      nspec, spec_mask, mb_mask)


_str_to_kernel = {'2': two_body_mc,
                  '2_en': two_body_mc_en,
                  '2_grad': two_body_mc_grad,
                  '2_force_en': two_body_mc_force_en,
                  '2_efs_energy': 'not implemented',
                  '2_efs_force': 'not implemented',
                  '2_efs_self': 'not implemented',
                  '3': three_body_mc,
                  '3_grad': three_body_mc_grad,
                  '3_en': three_body_mc_en,
                  '3_force_en': three_body_mc_force_en,
                  '3_efs_energy': 'not implemented',
                  '3_efs_force': 'not implemented',
                  '3_efs_self': 'not implemented',
                  'many': many_body_mc,
                  'many_grad': many_body_mc_grad,
                  'many_en': many_body_mc_en,
                  'many_force_en': many_body_mc_force_en,
                  'many_efs_energy': 'not implemented',
                  'many_efs_force': 'not implemented',
                  'many_efs_self': 'not implemented',
                  '2+3': two_plus_three_body_mc,
                  '2+3_grad': two_plus_three_body_mc_grad,
                  '2+3_en': two_plus_three_mc_en,
                  '2+3_force_en': two_plus_three_mc_force_en,
                  '2+3_efs_energy': 'not implemented',
                  '2+3_efs_force': 'not implemented',
                  '2+3_efs_self': 'not implemented',
                  '2+3+many': two_three_many_body_mc,
                  '2+3+many_grad': two_three_many_body_mc_grad,
                  '2+3+many_en': two_three_many_mc_en,
                  '2+3+many_force_en': two_three_many_mc_force_en,
                  '2+3+many_efs_energy': 'not implemented',
                  '2+3+many_efs_force': 'not implemented',
                  '2+3+many_efs_self': 'not implemented'
                  }
