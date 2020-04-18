"""Multi-element 2-, 3-, and 2+3-body kernels that restrict all signal
variance hyperparameters to a single value."""
import numpy as np
# from numba import njit
from math import exp
import sys
import os
from flare.env import AtomicEnvironment
import flare.cutoffs as cf
from flare.kernels.kernels import force_helper, grad_constants, grad_helper, \
    force_energy_helper, three_body_en_helper, three_body_helper_1, \
    three_body_helper_2, three_body_grad_helper_1, three_body_grad_helper_2, \
    k_sq_exp_double_dev, k_sq_exp_dev, coordination_number, q_value, \
    mb_grad_helper_ls_, mb_grad_helper_ls_
from typing import Callable


# -----------------------------------------------------------------------------
#                        two plus three body kernels
# -----------------------------------------------------------------------------


def two_plus_three_body_mc(env1: AtomicEnvironment, env2: AtomicEnvironment,
                           d1: int, d2: int, hyps: 'ndarray',
                           cutoffs: 'ndarray',
                           cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """2+3-body multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2, sig_n).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body kernel.
    """

    sig2 = hyps[0]
    ls2 = hyps[1]
    sig3 = hyps[2]
    ls3 = hyps[3]
    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]

    two_term = two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                               env2.bond_array_2, env2.ctype, env2.etypes,
                               d1, d2, sig2, ls2, r_cut_2, cutoff_func)

    three_term = \
        three_body_mc_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                          env2.bond_array_3, env2.ctype, env2.etypes,
                          env1.cross_bond_inds, env2.cross_bond_inds,
                          env1.cross_bond_dists, env2.cross_bond_dists,
                          env1.triplet_counts, env2.triplet_counts,
                          d1, d2, sig3, ls3, r_cut_3, cutoff_func)

    return two_term + three_term


def two_plus_three_body_mc_grad(env1: AtomicEnvironment,
                                env2: AtomicEnvironment,
                                d1: int, d2: int, hyps: 'ndarray',
                                cutoffs: 'ndarray',
                                cutoff_func: Callable = cf.quadratic_cutoff) \
        -> ('float', 'ndarray'):
    """2+3-body multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2, sig_n).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        (float, np.ndarray):
            Value of the 2+3-body kernel and its gradient
            with respect to the hyperparameters.
    """

    sig2 = hyps[0]
    ls2 = hyps[1]
    sig3 = hyps[2]
    ls3 = hyps[3]
    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]

    kern2, grad2 = \
        two_body_mc_grad_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                             env2.bond_array_2, env2.ctype, env2.etypes,
                             d1, d2, sig2, ls2, r_cut_2, cutoff_func)

    kern3, grad3 = \
        three_body_mc_grad_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                               env2.bond_array_3, env2.ctype, env2.etypes,
                               env1.cross_bond_inds, env2.cross_bond_inds,
                               env1.cross_bond_dists, env2.cross_bond_dists,
                               env1.triplet_counts, env2.triplet_counts,
                               d1, d2, sig3, ls3, r_cut_3,
                               cutoff_func)

    return kern2 + kern3, np.array([grad2[0], grad2[1], grad3[0], grad3[1]])


def two_plus_three_mc_force_en(env1: AtomicEnvironment,
                               env2: AtomicEnvironment,
                               d1: int, hyps: 'ndarray', cutoffs: 'ndarray',
                               cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """2+3-body multi-element kernel between a force component and a local
    energy.

    Args:
        env1 (AtomicEnvironment): Local environment associated with the
            force component.
        env2 (AtomicEnvironment): Local environment associated with the
            local energy.
        d1 (int): Force component of the first environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body force/energy kernel.
    """

    sig2 = hyps[0]
    ls2 = hyps[1]
    sig3 = hyps[2]
    ls3 = hyps[3]
    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]

    two_term = \
        two_body_mc_force_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                 env2.bond_array_2, env2.ctype, env2.etypes,
                                 d1, sig2, ls2, r_cut_2, cutoff_func) / 2

    three_term = \
        three_body_mc_force_en_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                                   env2.bond_array_3, env2.ctype, env2.etypes,
                                   env1.cross_bond_inds, env2.cross_bond_inds,
                                   env1.cross_bond_dists,
                                   env2.cross_bond_dists,
                                   env1.triplet_counts, env2.triplet_counts,
                                   d1, sig3, ls3, r_cut_3, cutoff_func) / 3

    return two_term + three_term


def two_plus_three_mc_en(env1: AtomicEnvironment, env2: AtomicEnvironment,
                         hyps: 'ndarray', cutoffs: 'ndarray',
                         cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """2+3-body multi-element kernel between two local energies.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3-body force/energy kernel.
    """

    sig2 = hyps[0]
    ls2 = hyps[1]
    sig3 = hyps[2]
    ls3 = hyps[3]
    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]

    two_term = two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                  env2.bond_array_2, env2.ctype, env2.etypes,
                                  sig2, ls2, r_cut_2, cutoff_func)

    three_term = \
        three_body_mc_en_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                             env2.bond_array_3, env2.ctype, env2.etypes,
                             env1.cross_bond_inds, env2.cross_bond_inds,
                             env1.cross_bond_dists, env2.cross_bond_dists,
                             env1.triplet_counts, env2.triplet_counts,
                             sig3, ls3, r_cut_3, cutoff_func)

    return two_term + three_term


# -----------------------------------------------------------------------------
#                     two plus three plus many body kernels
# -----------------------------------------------------------------------------

def two_plus_three_plus_many_body_mc(env1: AtomicEnvironment, env2: AtomicEnvironment,
                                     d1: int, d2: int, hyps, cutoffs,
                                     cutoff_func=cf.quadratic_cutoff):
    """2+3-body single-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2, sig3, ls3, sig_n).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3+many-body kernel.
    """

    sig2 = hyps[0]
    ls2 = hyps[1]
    sig3 = hyps[2]
    ls3 = hyps[3]
    sigm = hyps[4]
    lsm = hyps[5]

    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]
    r_cut_m = cutoffs[2]

    two_term = two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                               env2.bond_array_2, env2.ctype, env2.etypes,
                               d1, d2, sig2, ls2, r_cut_2, cutoff_func)

    three_term = \
        three_body_mc_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                          env2.bond_array_3, env2.ctype, env2.etypes,
                          env1.cross_bond_inds, env2.cross_bond_inds,
                          env1.cross_bond_dists, env2.cross_bond_dists,
                          env1.triplet_counts, env2.triplet_counts,
                          d1, d2, sig3, ls3, r_cut_3, cutoff_func)

    many_term = many_body_mc_jit(env1.bond_array_mb, env2.bond_array_mb, env1.neigh_dists_mb,
                                 env2.neigh_dists_mb, env1.num_neighs_mb, env2.num_neighs_mb,
                                 env1.ctype, env2.ctype, env1.etypes, env2.etypes,
                                 env1.etype_mb, env2.etype_mb, env1.species, env2.species,
                                 d1, d2, sigm, lsm, r_cut_m, cutoff_func)

    return two_term + three_term + many_term


def two_plus_three_plus_many_body_mc_grad(env1: AtomicEnvironment, env2: AtomicEnvironment,
                                          d1: int, d2: int, hyps, cutoffs,
                                          cutoff_func=cf.quadratic_cutoff):
    """2+3+many-body single-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2, sig3, ls3, sig_n).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3+many-body kernel.
    """

    sig2 = hyps[0]
    ls2 = hyps[1]
    sig3 = hyps[2]
    ls3 = hyps[3]
    sigm = hyps[4]
    lsm = hyps[5]

    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]
    r_cut_m = cutoffs[2]

    kern2, grad2 = two_body_mc_grad_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                        env2.bond_array_2, env2.ctype, env2.etypes,
                                        d1, d2, sig2, ls2, r_cut_2, cutoff_func)

    kern3, grad3 = \
        three_body_mc_grad_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                               env2.bond_array_3, env2.ctype, env2.etypes,
                               env1.cross_bond_inds, env2.cross_bond_inds,
                               env1.cross_bond_dists, env2.cross_bond_dists,
                               env1.triplet_counts, env2.triplet_counts,
                               d1, d2, sig3, ls3, r_cut_3, cutoff_func)

    kern_many, gradm = many_body_mc_grad_jit(env1.bond_array_mb, env2.bond_array_mb,
                                             env1.neigh_dists_mb, env2.neigh_dists_mb,
                                             env1.num_neighs_mb, env2.num_neighs_mb, env1.ctype,
                                             env2.ctype, env1.etypes, env2.etypes,
                                             env1.etype_mb, env2.etype_mb,
                                             env1.species, env2.species, d1, d2, sigm,
                                             lsm, r_cut_m, cutoff_func)

    return kern2 + kern3 + kern_many, np.array(
        [grad2[0], grad2[1], grad3[0], grad3[1], gradm[0], gradm[1]])


def two_plus_three_plus_many_body_mc_force_en(env1: AtomicEnvironment, env2: AtomicEnvironment,
                                              d1: int, hyps, cutoffs,
                                              cutoff_func=cf.quadratic_cutoff):
    """2+3+many-body single-element kernel between two force and energy components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2, sig3, ls3, sig_n).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3+many-body kernel.
    """

    sig2 = hyps[0]
    ls2 = hyps[1]
    sig3 = hyps[2]
    ls3 = hyps[3]
    sigm = hyps[4]
    lsm = hyps[5]

    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]
    r_cut_m = cutoffs[2]

    two_term = \
        two_body_mc_force_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                 env2.bond_array_2, env2.ctype, env2.etypes,
                                 d1, sig2, ls2, r_cut_2, cutoff_func) / 2

    three_term = \
        three_body_mc_force_en_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                                   env2.bond_array_3, env2.ctype, env2.etypes,
                                   env1.cross_bond_inds, env2.cross_bond_inds,
                                   env1.cross_bond_dists,
                                   env2.cross_bond_dists,
                                   env1.triplet_counts, env2.triplet_counts,
                                   d1, sig3, ls3, r_cut_3, cutoff_func) / 3

    many_term = many_body_mc_force_en_jit(env1.bond_array_mb, env2.bond_array_mb,
                                          env1.neigh_dists_mb, env1.num_neighs_mb,
                                          env1.ctype, env2.ctype, env1.etypes, env2.etypes,
                                          env1.etype_mb,
                                          env1.species, env2.species, d1, sigm, lsm, r_cut_m,
                                          cutoff_func)

    return two_term + three_term + many_term


def two_plus_three_plus_many_body_mc_en(env1: AtomicEnvironment, env2: AtomicEnvironment,
                                        hyps, cutoffs, cutoff_func=cf.quadratic_cutoff):
    """2+3+many-body single-element energy kernel.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig1, ls1,
            sig2, ls2, sig3, ls3, sig_n).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2+3+many-body kernel.
    """

    sig2 = hyps[0]
    ls2 = hyps[1]
    sig3 = hyps[2]
    ls3 = hyps[3]
    sigm = hyps[4]
    lsm = hyps[5]

    r_cut_2 = cutoffs[0]
    r_cut_3 = cutoffs[1]
    r_cut_m = cutoffs[2]

    two_term = two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                  env2.bond_array_2, env2.ctype, env2.etypes,
                                  sig2, ls2, r_cut_2, cutoff_func)

    three_term = \
        three_body_mc_en_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                             env2.bond_array_3, env2.ctype, env2.etypes,
                             env1.cross_bond_inds, env2.cross_bond_inds,
                             env1.cross_bond_dists, env2.cross_bond_dists,
                             env1.triplet_counts, env2.triplet_counts,
                             sig3, ls3, r_cut_3, cutoff_func)

    many_term = many_body_mc_en_jit(env1.bond_array_2, env2.bond_array_2, env1.ctype,
                                    env2.ctype, env1.etypes, env2.etypes, env1.species,
                                    env2.species,
                                    sigm, lsm, r_cut_m, cutoff_func)

    return two_term + three_term + many_term


# -----------------------------------------------------------------------------
#                      three body multicomponent kernel
# -----------------------------------------------------------------------------


def three_body_mc(env1: AtomicEnvironment, env2: AtomicEnvironment,
                  d1: int, d2: int, hyps: 'ndarray', cutoffs: 'ndarray',
                  cutoff_func: Callable = cf.quadratic_cutoff) -> float:
    """3-body multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 3-body kernel.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_mc_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                             env2.bond_array_3, env2.ctype, env2.etypes,
                             env1.cross_bond_inds, env2.cross_bond_inds,
                             env1.cross_bond_dists, env2.cross_bond_dists,
                             env1.triplet_counts, env2.triplet_counts,
                             d1, d2, sig, ls, r_cut, cutoff_func)


def three_body_mc_grad(env1: AtomicEnvironment, env2: AtomicEnvironment,
                       d1: int, d2: int, hyps: 'ndarray', cutoffs: 'ndarray',
                       cutoff_func: Callable = cf.quadratic_cutoff) \
        -> ('float', 'ndarray'):
    """3-body multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        (float, np.ndarray):
            Value of the 3-body kernel and its gradient with respect to the
            hyperparameters.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_mc_grad_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                                  env2.bond_array_3, env2.ctype, env2.etypes,
                                  env1.cross_bond_inds, env2.cross_bond_inds,
                                  env1.cross_bond_dists, env2.cross_bond_dists,
                                  env1.triplet_counts, env2.triplet_counts,
                                  d1, d2, sig, ls, r_cut, cutoff_func)


def three_body_mc_force_en(env1: AtomicEnvironment, env2: AtomicEnvironment,
                           d1: int, hyps: 'ndarray', cutoffs: 'ndarray',
                           cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """3-body multi-element kernel between a force component and a local
    energy.

    Args:
        env1 (AtomicEnvironment): Local environment associated with the
            force component.
        env2 (AtomicEnvironment): Local environment associated with the
            local energy.
        d1 (int): Force component of the first environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 3-body force/energy kernel.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_mc_force_en_jit(env1.bond_array_3, env1.ctype,
                                      env1.etypes,
                                      env2.bond_array_3, env2.ctype,
                                      env2.etypes,
                                      env1.cross_bond_inds,
                                      env2.cross_bond_inds,
                                      env1.cross_bond_dists,
                                      env2.cross_bond_dists,
                                      env1.triplet_counts, env2.triplet_counts,
                                      d1, sig, ls, r_cut, cutoff_func) / 3


def three_body_mc_en(env1: AtomicEnvironment, env2: AtomicEnvironment,
                     hyps: 'ndarray', cutoffs: 'ndarray',
                     cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """3-body multi-element kernel between two local energies.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 3-body force/energy kernel.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    return three_body_mc_en_jit(env1.bond_array_3, env1.ctype, env1.etypes,
                                env2.bond_array_3, env2.ctype, env2.etypes,
                                env1.cross_bond_inds, env2.cross_bond_inds,
                                env1.cross_bond_dists, env2.cross_bond_dists,
                                env1.triplet_counts, env2.triplet_counts,
                                sig, ls, r_cut, cutoff_func)


# -----------------------------------------------------------------------------
#                       two body multicomponent kernel
# -----------------------------------------------------------------------------


def two_body_mc(env1: AtomicEnvironment, env2: AtomicEnvironment,
                d1: float, d2: float, hyps: 'ndarray', cutoffs: 'ndarray',
                cutoff_func: Callable = cf.quadratic_cutoff) -> float:
    """2-body multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): One-element array containing the 2-body
            cutoff.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2-body kernel.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    return two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                           env2.bond_array_2, env2.ctype, env2.etypes,
                           d1, d2, sig, ls, r_cut, cutoff_func)


def two_body_mc_grad(env1: AtomicEnvironment, env2: AtomicEnvironment,
                     d1: int, d2: int, hyps: 'ndarray', cutoffs: 'ndarray',
                     cutoff_func: Callable = cf.quadratic_cutoff) \
        -> (float, 'ndarray'):
    """2-body multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): One-element array containing the 2-body
            cutoff.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        (float, np.ndarray):
            Value of the 2-body kernel and its gradient with respect to the
            hyperparameters.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    return two_body_mc_grad_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                env2.bond_array_2, env2.ctype, env2.etypes,
                                d1, d2, sig, ls, r_cut, cutoff_func)


def two_body_mc_force_en(env1: AtomicEnvironment, env2: AtomicEnvironment,
                         d1: int, hyps: 'ndarray', cutoffs: 'ndarray',
                         cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """2-body multi-element kernel between a force component and a local
    energy.

    Args:
        env1 (AtomicEnvironment): Local environment associated with the
            force component.
        env2 (AtomicEnvironment): Local environment associated with the
            local energy.
        d1 (int): Force component of the first environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): One-element array containing the 2-body
            cutoff.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 2-body force/energy kernel.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    return two_body_mc_force_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                    env2.bond_array_2, env2.ctype, env2.etypes,
                                    d1, sig, ls, r_cut, cutoff_func) / 2


def two_body_mc_en(env1: AtomicEnvironment, env2: AtomicEnvironment,
                   hyps: 'ndarray', cutoffs: 'ndarray',
                   cutoff_func: Callable = cf.quadratic_cutoff) \
        -> float:
    """2-body multi-element kernel between two local energies.

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
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    return two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                              env2.bond_array_2, env2.ctype, env2.etypes,
                              sig, ls, r_cut, cutoff_func)


# -----------------------------------------------------------------------------
#                       many body multicomponent kernel
# -----------------------------------------------------------------------------


def many_body_mc(env1: AtomicEnvironment, env2: AtomicEnvironment,
                 d1: int, d2: int, hyps: 'ndarray', cutoffs: 'ndarray',
                 cutoff_func: Callable = cf.quadratic_cutoff) -> float:
    """many-body multi-element kernel between two force components.

    Args:
        env1 (AtomicEnvironment): First local environment.
        env2 (AtomicEnvironment): Second local environment.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        hyps (np.ndarray): Hyperparameters of the kernel function (sig, ls).
        cutoffs (np.ndarray): Two-element array containing the 2- and 3-body
            cutoffs.
        cutoff_func (Callable): Cutoff function of the kernel.

    Return:
        float: Value of the 3-body kernel.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[2]

    bond_array_1 = env1.bond_array_mb
    bond_array_2 = env2.bond_array_mb

    neigh_dists_1 = env1.neigh_dists_mb
    num_neigh_1 = env1.num_neighs_mb

    neigh_dists_2 = env2.neigh_dists_mb
    num_neigh_2 = env2.num_neighs_mb

    # Get atomic species of central atom, their neighbours, and their neighbours' neighbours
    c1, c2 = env1.ctype, env2.ctype
    etypes1, etypes2 = env1.etypes, env2.etypes
    etypes_neigh_1, etypes_neigh_2 = env1.etype_mb, env2.etype_mb

    return many_body_mc_jit(bond_array_1, bond_array_2, neigh_dists_1, neigh_dists_2, num_neigh_1,
                            num_neigh_2, c1, c2, etypes1, etypes2, etypes_neigh_1, etypes_neigh_2,
                            env1.species, env2.species, d1, d2, sig, ls, r_cut, cutoff_func)


def many_body_mc_grad(env1: AtomicEnvironment, env2: AtomicEnvironment,
                      d1: int, d2: int, hyps: 'ndarray', cutoffs: 'ndarray',
                      cutoff_func: Callable = cf.quadratic_cutoff) -> float:
    """gradient manybody-body multi-element kernel between two force components.

    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[2]

    bond_array_1 = env1.bond_array_mb
    bond_array_2 = env2.bond_array_mb

    neigh_dists_1 = env1.neigh_dists_mb
    num_neigh_1 = env1.num_neighs_mb

    neigh_dists_2 = env2.neigh_dists_mb
    num_neigh_2 = env2.num_neighs_mb

    c1, c2 = env1.ctype, env2.ctype
    etypes1, etypes2 = env1.etypes, env2.etypes
    etypes_neigh_1, etypes_neigh_2 = env1.etype_mb, env2.etype_mb

    kernel, kernel_grad = many_body_mc_grad_jit(bond_array_1, bond_array_2, neigh_dists_1,
                                                neigh_dists_2, num_neigh_1, num_neigh_2, c1,
                                                c2, etypes1, etypes2, etypes_neigh_1,
                                                etypes_neigh_2, env1.species, env2.species,
                                                d1, d2, sig, ls, r_cut, cutoff_func)

    return kernel, kernel_grad


def many_body_mc_force_en(env1, env2, d1, hyps, cutoffs,
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
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[2]

    bond_array_1 = env1.bond_array_mb
    bond_array_2 = env2.bond_array_mb

    neigh_dists_1 = env1.neigh_dists_mb
    num_neigh_1 = env1.num_neighs_mb

    c1, c2 = env1.ctype, env2.ctype
    etypes1, etypes2 = env1.etypes, env2.etypes
    etypes_neigh_1 = env1.etype_mb

    # divide by three to account for triple counting
    return many_body_mc_force_en_jit(bond_array_1, bond_array_2, neigh_dists_1, num_neigh_1,
                                     c1, c2, etypes1, etypes2, etypes_neigh_1,
                                     env1.species, env2.species, d1, sig, ls, r_cut, cutoff_func)


def many_body_mc_en(env1: AtomicEnvironment, env2: AtomicEnvironment,
                    hyps: 'ndarray', cutoffs: 'ndarray',
                    cutoff_func: Callable = cf.quadratic_cutoff) -> float:
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
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[2]

    return many_body_mc_en_jit(env1.bond_array_2, env2.bond_array_2, env1.ctype,
                               env2.ctype, env1.etypes, env2.etypes, env1.species, env2.species,
                               sig, ls, r_cut, cutoff_func)


# -----------------------------------------------------------------------------
#                 three body multicomponent kernel (numba)
# -----------------------------------------------------------------------------


# @njit
def three_body_mc_jit(bond_array_1, c1, etypes1,
                      bond_array_2, c2, etypes2,
                      cross_bond_inds_1, cross_bond_inds_2,
                      cross_bond_dists_1, cross_bond_dists_2,
                      triplets_1, triplets_2,
                      d1, d2, sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between two force components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the first local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_inds_2 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the second local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the first
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        cross_bond_dists_2 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the second
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        triplets_1 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the first local environment that are
            within a distance r_cut of atom m.
        triplets_2 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the second local environment that are
            within a distance r_cut of atom m.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the 3-body kernel.
    """
    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    # first loop over the first 3-body environment
    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
        ei1 = etypes1[m]

        # second loop over the first 3-body environment
        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            # first loop over the second 3-body environment
            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                ej1 = etypes2[p]

                # second loop over the second 3-body environment
                for q in range(triplets_2[p]):
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

                    # consider six permutations
                    if (c1 == c2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            kern += \
                                three_body_helper_1(ci1, ci2, cj1, cj2, r11,
                                                    r22, r33, fi, fj, fdi, fdj,
                                                    ls1, ls2, ls3, sig2)
                        if (ei1 == ej2) and (ei2 == ej1):
                            kern += \
                                three_body_helper_1(ci1, ci2, cj2, cj1, r12,
                                                    r21, r33, fi, fj, fdi, fdj,
                                                    ls1, ls2, ls3, sig2)
                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            kern += \
                                three_body_helper_2(ci2, ci1, cj2, cj1, r21,
                                                    r13, r32, fi, fj, fdi,
                                                    fdj, ls1, ls2, ls3, sig2)
                        if (ei1 == c2) and (ei2 == ej2):
                            kern += \
                                three_body_helper_2(ci1, ci2, cj2, cj1, r11,
                                                    r23, r32, fi, fj, fdi,
                                                    fdj, ls1, ls2, ls3, sig2)
                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            kern += \
                                three_body_helper_2(ci2, ci1, cj1, cj2, r22,
                                                    r13, r31, fi, fj, fdi,
                                                    fdj, ls1, ls2, ls3, sig2)
                        if (ei1 == c2) and (ei2 == ej1):
                            kern += \
                                three_body_helper_2(ci1, ci2, cj1, cj2, r12,
                                                    r23, r31, fi, fj, fdi,
                                                    fdj, ls1, ls2, ls3, sig2)

    return kern


# @njit
def three_body_mc_grad_jit(bond_array_1, c1, etypes1,
                           bond_array_2, c2, etypes2,
                           cross_bond_inds_1, cross_bond_inds_2,
                           cross_bond_dists_1, cross_bond_dists_2,
                           triplets_1, triplets_2,
                           d1, d2, sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the first local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_inds_2 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the second local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the first
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        cross_bond_dists_2 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the second
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        triplets_1 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the first local environment that are
            within a distance r_cut of atom m.
        triplets_2 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the second local environment that are
            within a distance r_cut of atom m.
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        (float, float):
            Value of the 3-body kernel and its gradient with respect to the
            hyperparameters.
    """
    kern = 0
    sig_derv = 0
    ls_derv = 0
    kern_grad = np.zeros(2)

    # pre-compute constants that appear in the inner loop
    sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6 = grad_constants(sig, ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
        ei1 = etypes1[m]

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri3 = cross_bond_dists_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            ei2 = etypes1[ind1]

            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
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
                                                         fdi, fdj, ls1, ls2,
                                                         ls3, ls4, ls5, ls6,
                                                         sig2, sig3)
                            kern += kern_term
                            sig_derv += sig_term
                            ls_derv += ls_term

                        if (ei1 == ej2) and (ei2 == ej1):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_1(ci1, ci2, cj2, cj1,
                                                         r12, r21, r33, fi, fj,
                                                         fdi, fdj, ls1, ls2,
                                                         ls3, ls4, ls5, ls6,
                                                         sig2, sig3)
                            kern += kern_term
                            sig_derv += sig_term
                            ls_derv += ls_term

                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_2(ci2, ci1, cj2, cj1,
                                                         r21, r13, r32, fi, fj,
                                                         fdi, fdj, ls1, ls2,
                                                         ls3, ls4, ls5, ls6,
                                                         sig2, sig3)
                            kern += kern_term
                            sig_derv += sig_term
                            ls_derv += ls_term

                        if (ei1 == c2) and (ei2 == ej2):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_2(ci1, ci2, cj2, cj1,
                                                         r11, r23, r32, fi, fj,
                                                         fdi, fdj, ls1, ls2,
                                                         ls3, ls4, ls5, ls6,
                                                         sig2, sig3)
                            kern += kern_term
                            sig_derv += sig_term
                            ls_derv += ls_term

                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_2(ci2, ci1, cj1, cj2,
                                                         r22, r13, r31, fi, fj,
                                                         fdi, fdj, ls1, ls2,
                                                         ls3, ls4, ls5, ls6,
                                                         sig2, sig3)
                            kern += kern_term
                            sig_derv += sig_term
                            ls_derv += ls_term

                        if (ei1 == c2) and (ei2 == ej1):
                            kern_term, sig_term, ls_term = \
                                three_body_grad_helper_2(ci1, ci2, cj1, cj2,
                                                         r12, r23, r31, fi, fj,
                                                         fdi, fdj, ls1, ls2,
                                                         ls3, ls4, ls5, ls6,
                                                         sig2, sig3)

                            kern += kern_term
                            sig_derv += sig_term
                            ls_derv += ls_term

    kern_grad[0] = sig_derv
    kern_grad[1] = ls_derv

    return kern, kern_grad


# @njit
def three_body_mc_force_en_jit(bond_array_1, c1, etypes1,
                               bond_array_2, c2, etypes2,
                               cross_bond_inds_1, cross_bond_inds_2,
                               cross_bond_dists_1, cross_bond_dists_2,
                               triplets_1, triplets_2,
                               d1, sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between a force component and a local
    energy accelerated with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the first local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_inds_2 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the second local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the first
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        cross_bond_dists_2 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the second
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        triplets_1 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the first local environment that are
            within a distance r_cut of atom m.
        triplets_2 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the second local environment that are
            within a distance r_cut of atom m.
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """
    kern = 0

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
        ei1 = etypes1[m]

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1 * fi2 * fi3
            fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
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
                                                         r33, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == ej2) and (ei2 == ej1):
                            kern += three_body_en_helper(ci1, ci2, r12, r21,
                                                         r33, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            kern += three_body_en_helper(ci1, ci2, r13, r21,
                                                         r32, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == c2) and (ei2 == ej2):
                            kern += three_body_en_helper(ci1, ci2, r11, r23,
                                                         r32, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            kern += three_body_en_helper(ci1, ci2, r13, r22,
                                                         r31, fi, fj, fdi, ls1,
                                                         ls2, sig2)
                        if (ei1 == c2) and (ei2 == ej1):
                            kern += three_body_en_helper(ci1, ci2, r12, r23,
                                                         r31, fi, fj, fdi, ls1,
                                                         ls2, sig2)

    return kern


# @njit
def three_body_mc_en_jit(bond_array_1, c1, etypes1,
                         bond_array_2, c2, etypes2,
                         cross_bond_inds_1, cross_bond_inds_2,
                         cross_bond_dists_1, cross_bond_dists_2,
                         triplets_1, triplets_2,
                         sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between two local energies accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the first local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_inds_2 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the second local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the first
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        cross_bond_dists_2 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the second
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        triplets_1 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the first local environment that are
            within a distance r_cut of atom m.
        triplets_2 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the second local environment that are
            within a distance r_cut of atom m.
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body local energy kernel.
    """

    kern = 0

    sig2 = sig * sig
    ls2 = 1 / (2 * ls * ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1, _ = cutoff_func(r_cut, ri1, 0)
        ei1 = etypes1[m]

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            fi2, _ = cutoff_func(r_cut, ri2, 0)
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)
            fi = fi1 * fi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
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
                            C1 = r11 * r11 + r22 * r22 + r33 * r33
                            kern += sig2 * exp(-C1 * ls2) * fi * fj
                        if (ei1 == ej2) and (ei2 == ej1):
                            C3 = r12 * r12 + r21 * r21 + r33 * r33
                            kern += sig2 * exp(-C3 * ls2) * fi * fj
                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            C5 = r13 * r13 + r21 * r21 + r32 * r32
                            kern += sig2 * exp(-C5 * ls2) * fi * fj
                        if (ei1 == c2) and (ei2 == ej2):
                            C2 = r11 * r11 + r23 * r23 + r32 * r32
                            kern += sig2 * exp(-C2 * ls2) * fi * fj
                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            C6 = r13 * r13 + r22 * r22 + r31 * r31
                            kern += sig2 * exp(-C6 * ls2) * fi * fj
                        if (ei1 == c2) and (ei2 == ej1):
                            C4 = r12 * r12 + r23 * r23 + r31 * r31
                            kern += sig2 * exp(-C4 * ls2) * fi * fj

    return kern


# -----------------------------------------------------------------------------
#                 two body multicomponent kernel (numba)
# -----------------------------------------------------------------------------


# @njit
def two_body_mc_jit(bond_array_1, c1, etypes1,
                    bond_array_2, c2, etypes2,
                    d1, d2, sig, ls, r_cut, cutoff_func):
    """2-body multi-element kernel between two force components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 2-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 2-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        d2 (int): Force component of the second environment (1=x, 2=y, 3=z).
        sig (float): 2-body signal variance hyperparameter.
        ls (float): 2-body length scale hyperparameter.
        r_cut (float): 2-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the 2-body kernel.
    """
    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                cj = bond_array_2[n, d2]
                fj, fdj = cutoff_func(r_cut, rj, cj)
                r11 = ri - rj

                A = ci * cj
                B = r11 * ci
                C = r11 * cj
                D = r11 * r11

                kern += force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                     ls1, ls2, ls3, sig2)

    return kern


# @njit
def two_body_mc_grad_jit(bond_array_1, c1, etypes1,
                         bond_array_2, c2, etypes2,
                         d1, d2, sig, ls, r_cut, cutoff_func):
    """2-body multi-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        bond_array_1 (np.ndarray): 2-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 2-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        d2 (int): Force component of the second environment (1=x, 2=y, 3=z).
        sig (float): 2-body signal variance hyperparameter.
        ls (float): 2-body length scale hyperparameter.
        r_cut (float): 2-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        (float, float):
            Value of the 2-body kernel and its gradient with respect to the
            hyperparameters.
    """

    kern = 0
    sig_derv = 0
    ls_derv = 0
    kern_grad = np.zeros(2)

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    ls4 = 1 / (ls * ls * ls)
    ls5 = ls * ls
    ls6 = ls2 * ls4

    sig2 = sig * sig
    sig3 = 2 * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                cj = bond_array_2[n, d2]
                fj, fdj = cutoff_func(r_cut, rj, cj)

                r11 = ri - rj

                A = ci * cj
                B = r11 * ci
                C = r11 * cj
                D = r11 * r11

                kern_term, sig_term, ls_term = \
                    grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3,
                                ls4, ls5, ls6, sig2, sig3)

                kern += kern_term
                sig_derv += sig_term
                ls_derv += ls_term

    kern_grad[0] = sig_derv
    kern_grad[1] = ls_derv

    return kern, kern_grad


# @njit
def two_body_mc_force_en_jit(bond_array_1, c1, etypes1,
                             bond_array_2, c2, etypes2,
                             d1, sig, ls, r_cut, cutoff_func):
    """2-body multi-element kernel between a force component and a local
    energy accelerated with Numba.

    Args:
        bond_array_1 (np.ndarray): 2-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 2-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        d1 (int): Force component of the first environment (1=x, 2=y, 3=z).
        sig (float): 2-body signal variance hyperparameter.
        ls (float): 2-body length scale hyperparameter.
        r_cut (float): 2-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 2-body force/energy kernel.
    """

    kern = 0

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)

                r11 = ri - rj
                B = r11 * ci
                D = r11 * r11
                kern += force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2)

    return kern


# @njit
def two_body_mc_en_jit(bond_array_1, c1, etypes1,
                       bond_array_2, c2, etypes2,
                       sig, ls, r_cut, cutoff_func):
    """2-body multi-element kernel between two local energies accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 2-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 2-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        sig (float): 2-body signal variance hyperparameter.
        ls (float): 2-body length scale hyperparameter.
        r_cut (float): 2-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 2-body local energy kernel.
    """
    kern = 0

    ls1 = 1 / (2 * ls * ls)
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        fi, _ = cutoff_func(r_cut, ri, 0)
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)
                r11 = ri - rj
                kern += fi * fj * sig2 * exp(-r11 * r11 * ls1)

    return kern


# -----------------------------------------------------------------------------
#                 many body multicomponent kernel (numba)
# -----------------------------------------------------------------------------

# @njit
def many_body_mc_jit_(bond_array_1, bond_array_2, neigh_dists_1, neigh_dists_2, num_neigh_1,
                      num_neigh_2, c1, c2, etypes1, etypes2, etypes_neigh_1, etypes_neigh_2,
                      species1, species2, d1, d2, sig, ls, r_cut, cutoff_func):
    """many-body multi-element kernel between two force components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        neigh_dists_1 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the first local environment.
        neigh_dists_2 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the second local environment.
        num_neigh_1 (np.ndarray): number of neighbours of each atom in the first
            local environment
        num_neigh_2 (np.ndarray): number of neighbours of each atom in the second
            local environment
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        etypes_neigh_1 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 1
        etypes_neigh_2 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the many-body kernel.
    """

    kern = 0

    useful_species = np.array(list(set(species1).union(set(species2))), dtype=np.int8)

    # loop over all possible species
    for s in useful_species:
        # Calculate many-body descriptor values for central atoms 1 and 2
        q1 = q_value_mc(bond_array_1[:, 0], r_cut, s, etypes1, cutoff_func)
        q2 = q_value_mc(bond_array_2[:, 0], r_cut, s, etypes2, cutoff_func)

        # kernel is nonzero only if central atoms are of the same species
        if c1 == c2:
            k12 = k_sq_exp_double_dev(q1, q2, sig, ls)
        else:
            k12 = 0

        qis = np.zeros(bond_array_1.shape[0])
        q1i_grads = np.zeros(bond_array_1.shape[0])
        qi1_grads = np.zeros(bond_array_1.shape[0])
        ki2s = np.zeros(bond_array_1.shape[0])

        qjs = np.zeros(bond_array_2.shape[0])
        qj2_grads = np.zeros(bond_array_2.shape[0])
        q2j_grads = np.zeros(bond_array_2.shape[0])
        k1js = np.zeros(bond_array_2.shape[0])

        # Loop over neighbours i of 1
        for i in range(bond_array_1.shape[0]):
            ri1 = bond_array_1[i, 0]
            ci1 = bond_array_1[i, d1]

            if etypes1[i] == s:
                qi1, qi1_grads[i] = coordination_number(ri1, ci1, r_cut, cutoff_func)

            if c1 == s:
                qi1, q1i_grads[i] = coordination_number(ri1, ci1, r_cut, cutoff_func)

            # kernel is nonzero only if central atoms are of the same species
            if c2 == etypes1[i]:
                ki2s[i] = k_sq_exp_double_dev(qis[i], q2, sig, ls)

        # Loop over neighbours j of 2
        for j in range(bond_array_2.shape[0]):
            rj2 = bond_array_2[j, 0]
            cj2 = bond_array_2[j, d2]

            if etypes2[j] == s:
                qj2, qj2_grads[j] = coordination_number(rj2, cj2, r_cut, cutoff_func)

            if c2 == s:
                qj2, q2j_grads[j] = coordination_number(rj2, cj2, r_cut, cutoff_func)

            # Calculate many-body descriptor value for j
            qjs[j] = q_value_mc(neigh_dists_2[j, :num_neigh_2[j]], r_cut,
                                s, etypes_neigh_2[j, :num_neigh_2[j]], cutoff_func)

            # kernel is nonzero only if central atoms are of the same species
            if c1 == etypes2[j]:
                k1js[j] = k_sq_exp_double_dev(q1, qjs[j], sig, ls)

        for i in range(bond_array_1.shape[0]):
            for j in range(bond_array_2.shape[0]):
                # kernel is nonzero only if central atoms are of the same species
                if etypes1[i] == etypes2[j]:
                    kij = k_sq_exp_double_dev(qis[i], qjs[j], sig, ls)
                else:
                    kij = 0

                kern += qi1_grads[i] * qj2_grads[j] * (k12 + ki2s[i] + k1js[j] + kij)

    return kern


def many_body_mc_jit(bond_array_1, bond_array_2, neigh_dists_1, neigh_dists_2, num_neigh_1,
                     num_neigh_2, c1, c2, etypes1, etypes2, etypes_neigh_1, etypes_neigh_2,
                     species1, species2, d1, d2, sig, ls, r_cut, cutoff_func):
    """many-body multi-element kernel between two force components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        neigh_dists_1 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the first local environment.
        neigh_dists_2 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the second local environment.
        num_neigh_1 (np.ndarray): number of neighbours of each atom in the first
            local environment
        num_neigh_2 (np.ndarray): number of neighbours of each atom in the second
            local environment
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        etypes_neigh_1 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 1
        etypes_neigh_2 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the many-body kernel.
    """

    kern = 0

    useful_species = np.array(list(set(species1).union(set(species2))), dtype=np.int8)

    # loop over all possible species
    for s in useful_species:

        # Calculate many-body descriptor values for central atoms 1 and 2
        q1 = q_value_mc(bond_array_1[:, 0], r_cut, s, etypes1, cutoff_func)
        q2 = q_value_mc(bond_array_2[:, 0], r_cut, s, etypes2, cutoff_func)

        # compute kernel between central atoms only if central atoms are of the same species
        if c1 == c2:
            k12 = k_sq_exp_double_dev(q1, q2, sig, ls)
        else:
            k12 = 0

        # initialise arrays of many body descriptors and gradients for the neighbour atoms in
        # the two configurations
        qis = np.zeros(bond_array_1.shape[0])
        q1i_grads = np.zeros(bond_array_1.shape[0])
        qi1_grads = np.zeros(bond_array_1.shape[0])
        ki2s = np.zeros(bond_array_1.shape[0])

        qjs = np.zeros(bond_array_2.shape[0])
        qj2_grads = np.zeros(bond_array_2.shape[0])
        q2j_grads = np.zeros(bond_array_2.shape[0])
        k1js = np.zeros(bond_array_2.shape[0])

        # Loop over neighbours i of 1st configuration
        for i in range(bond_array_1.shape[0]):
            ri1 = bond_array_1[i, 0]
            ci1 = bond_array_1[i, d1]

            if etypes1[i] == s:
                # derivative of pairwise component of many body descriptor q1i
                _, q1i_grads[i] = coordination_number(ri1, ci1, r_cut, cutoff_func)

            if c1 == s:
                # derivative of pairwise component of many body descriptor qi1
                _, qi1_grads[i] = coordination_number(ri1, ci1, r_cut, cutoff_func)

            # Calculate many-body descriptor value for i
            qis[i] = q_value_mc(neigh_dists_1[i, :num_neigh_1[i]], r_cut,
                                s, etypes_neigh_1[i, :num_neigh_1[i]], cutoff_func)

            # kernel is nonzero only if central atoms are of the same species
            if c2 == etypes1[i]:
                ki2s[i] = k_sq_exp_double_dev(qis[i], q2, sig, ls)

        # Loop over neighbours j of 2
        for j in range(bond_array_2.shape[0]):
            rj2 = bond_array_2[j, 0]
            cj2 = bond_array_2[j, d2]

            if etypes2[j] == s:
                _, q2j_grads[j] = coordination_number(rj2, cj2, r_cut, cutoff_func)

            if c2 == s:
                _, qj2_grads[j] = coordination_number(rj2, cj2, r_cut, cutoff_func)

            # Calculate many-body descriptor value for j
            qjs[j] = q_value_mc(neigh_dists_2[j, :num_neigh_2[j]], r_cut,
                                s, etypes_neigh_2[j, :num_neigh_2[j]], cutoff_func)

            # kernel is nonzero only if central atoms are of the same species
            if c1 == etypes2[j]:
                k1js[j] = k_sq_exp_double_dev(q1, qjs[j], sig, ls)

        for i in range(bond_array_1.shape[0]):
            for j in range(bond_array_2.shape[0]):
                # kernel is nonzero only if central atoms are of the same species
                if etypes1[i] == etypes2[j]:
                    kij = k_sq_exp_double_dev(qis[i], qjs[j], sig, ls)
                else:
                    kij = 0

                kern += q1i_grads[i] * q2j_grads[j] * k12
                kern += qi1_grads[i] * q2j_grads[j] * ki2s[i]
                kern += q1i_grads[i] * qj2_grads[j] * k1js[j]
                kern += qi1_grads[i] * qj2_grads[j] * kij
    return kern


# @njit
def many_body_mc_grad_jit(bond_array_1, bond_array_2, neigh_dists_1, neigh_dists_2, num_neigh_1,
                          num_neigh_2, c1, c2, etypes1, etypes2, etypes_neigh_1, etypes_neigh_2,
                          species1, species2, d1, d2, sig, ls, r_cut, cutoff_func):
    """gradient of many-body multi-element kernel between two force components
    w.r.t. the hyperparameters, accelerated with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        neigh_dists_1 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the first local environment.
        neigh_dists_2 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the second local environment.
        num_neigh_1 (np.ndarray): number of neighbours of each atom in the first
            local environment
        num_neigh_2 (np.ndarray): number of neighbours of each atom in the second
            local environment
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        etypes_neigh_1 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 1
        etypes_neigh_2 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        array: Value of the many-body kernel and its gradient w.r.t. sig and ls
    """

    kern = 0
    sig_derv = 0
    ls_derv = 0

    useful_species = np.array(list(set(species1).union(set(species2))), dtype=np.int8)

    for s in useful_species:

        # Calculate many-body descriptor values for 1 and 2
        q1 = q_value_mc(bond_array_1[:, 0], r_cut, s, etypes1, cutoff_func)
        q2 = q_value_mc(bond_array_2[:, 0], r_cut, s, etypes2, cutoff_func)

        if c1 == c2:
            k12 = k_sq_exp_double_dev(q1, q2, sig, ls)
            q12diffsq = (q1 - q2) ** 2  # * (q1 - q2)
            dk12 = mb_grad_helper_ls_(q12diffsq, sig, ls)
        else:
            k12 = 0
            dk12 = 0

        qis = np.zeros(bond_array_1.shape[0])
        q1i_grads = np.zeros(bond_array_1.shape[0])
        qi1_grads = np.zeros(bond_array_1.shape[0])
        ki2s = np.zeros(bond_array_1.shape[0])
        dki2s = np.zeros(bond_array_1.shape[0])

        qjs = np.zeros(bond_array_2.shape[0])
        qj2_grads = np.zeros(bond_array_2.shape[0])
        q2j_grads = np.zeros(bond_array_2.shape[0])
        k1js = np.zeros(bond_array_2.shape[0])
        dk1js = np.zeros(bond_array_2.shape[0])

        # Compute  ki2s, qi1_grads, and qis
        for i in range(bond_array_1.shape[0]):
            ri1 = bond_array_1[i, 0]
            ci1 = bond_array_1[i, d1]

            if etypes1[i] == s:
                _, q1i_grads[i] = coordination_number(ri1, ci1, r_cut, cutoff_func)

            if c1 == s:
                # derivative of pairwise component of many body descriptor qi1
                __, qi1_grads[i] = coordination_number(ri1, ci1, r_cut, cutoff_func)

            # Calculate many-body descriptor value for i
            qis[i] = q_value_mc(neigh_dists_1[i, :num_neigh_1[i]], r_cut,
                                s, etypes_neigh_1[i, :num_neigh_1[i]], cutoff_func)

            # ki2s[i] = k_sq_exp_double_dev(qis[i], q2, sig, ls)
            if c2 == etypes1[i]:
                ki2s[i] = k_sq_exp_double_dev(qis[i], q2, sig, ls)
                qi2diffsq = (qis[i] - q2) * (qis[i] - q2)
                dki2s[i] = mb_grad_helper_ls_(qi2diffsq, sig, ls)

        # Compute k1js, qj2_grads and qjs
        for j in range(bond_array_2.shape[0]):
            rj2 = bond_array_2[j, 0]
            cj2 = bond_array_2[j, d2]

            if etypes2[j] == s:
                _, q2j_grads[j] = coordination_number(rj2, cj2, r_cut, cutoff_func)

            if c2 == s:
                _, qj2_grads[j] = coordination_number(rj2, cj2, r_cut, cutoff_func)

            # Calculate many-body descriptor value for j
            qjs[j] = q_value_mc(neigh_dists_2[j, :num_neigh_2[j]], r_cut,
                                s, etypes_neigh_2[j, :num_neigh_2[j]], cutoff_func)

            # k1js[j] = k_sq_exp_double_dev(q1, qjs[j], sig, ls)

            if c1 == etypes2[j]:
                k1js[j] = k_sq_exp_double_dev(q1, qjs[j], sig, ls)
                q1jdiffsq = (q1 - qjs[j]) * (q1 - qjs[j])
                dk1js[j] = mb_grad_helper_ls_(q1jdiffsq, sig, ls)

        for i in range(bond_array_1.shape[0]):
            for j in range(bond_array_2.shape[0]):

                # kij = k_sq_exp_double_dev(qis[i], qjs[j], sig, ls)

                if etypes1[i] == etypes2[j]:
                    kij = k_sq_exp_double_dev(qis[i], qjs[j], sig, ls)
                    qijdiffsq = (qis[i] - qjs[j]) * (qis[i] - qjs[j])
                    dkij = mb_grad_helper_ls_(qijdiffsq, sig, ls)
                else:
                    kij = 0
                    dkij = 0

                kern_term = q1i_grads[i] * q2j_grads[j] * k12
                kern_term += qi1_grads[i] * q2j_grads[j] * ki2s[i]
                kern_term += q1i_grads[i] * qj2_grads[j] * k1js[j]
                kern_term += qi1_grads[i] * qj2_grads[j] * kij

                sig_term = 2. / sig * kern_term

                ls_term = q1i_grads[i] * q2j_grads[j] * dk12
                ls_term += qi1_grads[i] * q2j_grads[j] * dki2s[i]
                ls_term += q1i_grads[i] * qj2_grads[j] * dk1js[j]
                ls_term += qi1_grads[i] * qj2_grads[j] * dkij

                kern += kern_term

                sig_derv += sig_term

                ls_derv += ls_term

    grad = np.array([sig_derv, ls_derv])

    return kern, grad


# @njit
def many_body_mc_force_en_jit(bond_array_1, bond_array_2, neigh_dists_1, num_neigh_1,
                              c1, c2, etypes1, etypes2, etypes_neigh_1,
                              species1, species2, d1, sig, ls, r_cut, cutoff_func):
    """many-body many-element kernel between force and energy components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        neigh_dists_1 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the first local environment.
        num_neigh_1 (np.ndarray): number of neighbours of each atom in the first
            local environment
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        etypes_neigh_1 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 1
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the many-body kernel.
    """

    kern = 0

    useful_species = np.array(list(set(species1).union(set(species2))), dtype=np.int8)

    for s in useful_species:

        q1 = q_value_mc(bond_array_1[:, 0], r_cut, s, etypes1, cutoff_func)
        q2 = q_value_mc(bond_array_2[:, 0], r_cut, s, etypes2, cutoff_func)

        if c1 == c2:
            k12 = k_sq_exp_dev(q1, q2, sig, ls)
        else:
            k12 = 0

        qis = np.zeros(bond_array_1.shape[0])
        qi1_grads = np.zeros(bond_array_1.shape[0])
        q1i_grads = np.zeros(bond_array_1.shape[0])
        ki2s = np.zeros(bond_array_1.shape[0])

        # Loop over neighbours i of 1
        for i in range(bond_array_1.shape[0]):
            ri1 = bond_array_1[i, 0]
            ci1 = bond_array_1[i, d1]

            if etypes1[i] == s:
                _, q1i_grads[i] = coordination_number(ri1, ci1, r_cut, cutoff_func)

            if c1 == s:
                _, qi1_grads[i] = coordination_number(ri1, ci1, r_cut, cutoff_func)

            # Calculate many-body descriptor value for i
            qis[i] = q_value_mc(neigh_dists_1[i, :num_neigh_1[i]], r_cut,
                                s, etypes_neigh_1[i, :num_neigh_1[i]], cutoff_func)

            if c2 == etypes1[i]:
                ki2s[i] = k_sq_exp_dev(qis[i], q2, sig, ls)

            kern += - (q1i_grads[i] * k12 + qi1_grads[i] * ki2s[i])

    return kern


# @njit
def many_body_mc_en_jit(bond_array_1, bond_array_2, c1, c2, etypes1, etypes2,
                        species1, species2, sig, ls, r_cut, cutoff_func):
    """many-body many-element kernel between energy components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the many-body kernel.
    """
    useful_species = np.array(list(set(species1).union(set(species2))), dtype=np.int8)
    kern = 0

    if c1 == c2:
        for s in useful_species:
            q1 = q_value_mc(bond_array_1[:, 0], r_cut, s, etypes1, cutoff_func)
            q2 = q_value_mc(bond_array_2[:, 0], r_cut, s, etypes2, cutoff_func)
            q1q2diff = q1 - q2

            kern += sig * sig * exp(-q1q2diff * q1q2diff / (2 * ls * ls))

    return kern


# -----------------------------------------------------------------------------
#                           helper funcions
# -----------------------------------------------------------------------------


# @njit
def q_value_mc(distances, r_cut, ref_species, species, cutoff_func, q_func=coordination_number):
    """Compute value of many-body many components descriptor based
    on distances of atoms in the local many-body environment.

    Args:
        distances (np.ndarray): distances between atoms i and j
        r_cut (float): cutoff hyperparameter
        ref_species (int): species to consider to compute the contribution
        species (np.ndarray): atomic species of neighbours
        cutoff_func (callable): cutoff function
        q_func (callable): many-body pairwise descrptor function

    Return:
        float: the value of the many-body descriptor
    """

    q = 0

    for i in range(len(distances)):
        if species[i] == ref_species:
            q_, _ = q_func(distances[i], 0, r_cut, cutoff_func)
            q += q_

    return q


_str_to_kernel = {'two_body_mc': two_body_mc,
                  'two_body_mc_en': two_body_mc_en,
                  'two_body_mc_grad': two_body_mc_grad,
                  'two_body_mc_force_en': two_body_mc_force_en,
                  'three_body_mc': three_body_mc,
                  'three_body_mc_grad': three_body_mc_grad,
                  'three_body_mc_en': three_body_mc_en,
                  'three_body_mc_force_en': three_body_mc_force_en,
                  'two_plus_three_body_mc': two_plus_three_body_mc,
                  'two_plus_three_body_mc_grad': two_plus_three_body_mc_grad,
                  'two_plus_three_mc_en': two_plus_three_mc_en,
                  'two_plus_three_mc_force_en': two_plus_three_mc_force_en,
                  '2': two_body_mc,
                  '2_en': two_body_mc_en,
                  '2_grad': two_body_mc_grad,
                  '2_force_en': two_body_mc_force_en,
                  '3': three_body_mc,
                  '3_grad': three_body_mc_grad,
                  '3_en': three_body_mc_en,
                  '3_force_en': three_body_mc_force_en,
                  '2+3': two_plus_three_body_mc,
                  '2+3_grad': two_plus_three_body_mc_grad,
                  '2+3_en': two_plus_three_mc_en,
                  '2+3_force_en': two_plus_three_mc_force_en,
                  'many_body_mc': many_body_mc,
                  'many_body_mc_en': many_body_mc_en,
                  'many_body_mc_grad': many_body_mc_grad,
                  'many_body_mc_force_en': many_body_mc_force_en,
                  'many': many_body_mc,
                  'many_en': many_body_mc_en,
                  'many_grad': many_body_mc_grad,
                  'many_force_en': many_body_mc_force_en,
                  'two_plus_three_plus_many_body_mc': two_plus_three_plus_many_body_mc,
                  'two_plus_three_plus_many_body_mc_grad': two_plus_three_plus_many_body_mc_grad,
                  'two_plus_three_plus_many_body_mc_en': two_plus_three_plus_many_body_mc_en,
                  'two_plus_three_plus_many_body_mc_force_en': two_plus_three_plus_many_body_mc_force_en,
                  '2+3+many': two_plus_three_plus_many_body_mc,
                  '2+3+many_grad': two_plus_three_plus_many_body_mc_grad,
                  '2+3+many_en': two_plus_three_plus_many_body_mc_en,
                  '2+3+many_force_en': two_plus_three_plus_many_body_mc_force_en
                  }


def str_to_mc_kernel(string: str, include_grad: bool = False):
    """Converts a kernel label to the corresponding kernel function.

    Args:
        string (str): Name of the kernel function, e.g. "two_body_mc".
        include_grad (bool, optional): If True, returns the kernel gradient
            in addition to the kernel function. Defaults to False.

    Returns:
        Callable: Kernel function (and its gradient if include_grad = True).
    """

    if string not in _str_to_kernel.keys():
        raise ValueError("Kernel {} not found in list of available "
                         "kernels{}:".format(string, _str_to_kernel.keys()))

    if not include_grad:
        return _str_to_kernel[string]
    else:
        if 'two' in string and 'three' in string:
            return _str_to_kernel[string], two_plus_three_body_mc_grad
        elif 'two' in string and 'three' not in string:
            return _str_to_kernel[string], two_body_mc_grad
        elif 'two' not in string and 'three' in string:
            return _str_to_kernel[string], three_body_mc_grad
        else:
            raise ValueError("Gradient callable for {} not found".format(
                string))
