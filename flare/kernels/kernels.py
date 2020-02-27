"""Single element 2-, 3-, and 2+3-body kernels.
The kernel functions to choose:

* Two body:

    * two_body: force kernel
    * two_body_en: energy kernel
    * two_body_grad: gradient of kernel function
    * two_body_force_en: energy force kernel

* Three body:

    * three_body,
    * three_body_grad,
    * three_body_en,
    * three_body_force_en,
    
* Two plus three body:

    * two_plus_three_body,
    * two_plus_three_body_grad,
    * two_plus_three_en,
    * two_plus_three_force_en

**Example:**

>>> gp_model = GaussianProcess(kernel=two_body,
                               kernel_grad=two_body_grad,
                               energy_force_kernel=two_body_force_en,
                               energy_kernel=two_body_en,
                               <other arguments>)
"""


import numpy as np
from math import exp
from flare.env import AtomicEnvironment
from numba import njit
import flare.cutoffs as cf


# -----------------------------------------------------------------------------
#                        two plus three body kernels
# -----------------------------------------------------------------------------


def two_plus_three_body(env1: AtomicEnvironment, env2: AtomicEnvironment,
                        d1: int, d2: int, hyps, cutoffs,
                        cutoff_func=cf.quadratic_cutoff):
    """2+3-body single-element kernel between two force components.

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

    two_term = two_body_jit(env1.bond_array_2, env2.bond_array_2,
                            d1, d2, hyps[0], hyps[1], cutoffs[0], cutoff_func)

    three_term = \
        three_body_jit(env1.bond_array_3, env2.bond_array_3,
                       env1.cross_bond_inds, env2.cross_bond_inds,
                       env1.cross_bond_dists, env2.cross_bond_dists,
                       env1.triplet_counts, env2.triplet_counts,
                       d1, d2, hyps[2], hyps[3], cutoffs[1], cutoff_func)
    return two_term + three_term


def two_plus_three_body_grad(env1, env2, d1, d2, hyps, cutoffs,
                             cutoff_func=cf.quadratic_cutoff):
    """2+3-body single-element kernel between two force components and its
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
        (float, np.ndarray): Value of the 2+3-body kernel and its gradient
            with respect to the hyperparameters.
    """

    kern2, ls2, sig2 = \
        two_body_grad_jit(env1.bond_array_2, env2.bond_array_2,
                          d1, d2, hyps[0], hyps[1], cutoffs[0], cutoff_func)

    kern3, sig3, ls3 = \
        three_body_grad_jit(env1.bond_array_3, env2.bond_array_3,
                            env1.cross_bond_inds, env2.cross_bond_inds,
                            env1.cross_bond_dists, env2.cross_bond_dists,
                            env1.triplet_counts, env2.triplet_counts,
                            d1, d2, hyps[2], hyps[3], cutoffs[1], cutoff_func)

    return kern2 + kern3, np.array([sig2, ls2, sig3, ls3])


def two_plus_three_force_en(env1, env2, d1, hyps, cutoffs,
                            cutoff_func=cf.quadratic_cutoff):
    """2+3-body single-element kernel between a force component and a local
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

    two_term = two_body_force_en_jit(env1.bond_array_2, env2.bond_array_2,
                                     d1, hyps[0], hyps[1], cutoffs[0],
                                     cutoff_func)/2

    three_term = \
        three_body_force_en_jit(env1.bond_array_3, env2.bond_array_3,
                                env1.cross_bond_inds, env2.cross_bond_inds,
                                env1.cross_bond_dists,
                                env2.cross_bond_dists,
                                env1.triplet_counts, env2.triplet_counts,
                                d1, hyps[2], hyps[3], cutoffs[1],
                                cutoff_func)/3

    return two_term + three_term


def two_plus_three_en(env1, env2, hyps, cutoffs,
                      cutoff_func=cf.quadratic_cutoff):
    """2+3-body single-element kernel between two local energies.

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

    two_term = two_body_en_jit(env1.bond_array_2, env2.bond_array_2,
                               hyps[0], hyps[1], cutoffs[0], cutoff_func)

    three_term = \
        three_body_en_jit(env1.bond_array_3, env2.bond_array_3,
                          env1.cross_bond_inds, env2.cross_bond_inds,
                          env1.cross_bond_dists, env2.cross_bond_dists,
                          env1.triplet_counts, env2.triplet_counts,
                          hyps[2], hyps[3], cutoffs[1], cutoff_func)

    return two_term + three_term


# -----------------------------------------------------------------------------
#                              two body kernels
# -----------------------------------------------------------------------------


def two_body(env1, env2, d1, d2, hyps, cutoffs,
             cutoff_func=cf.quadratic_cutoff):
    """2-body single-element kernel between two force components.

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

    return two_body_jit(env1.bond_array_2, env2.bond_array_2,
                        d1, d2, sig, ls, r_cut, cutoff_func)


def two_body_grad(env1, env2, d1, d2, hyps, cutoffs,
                  cutoff_func=cf.quadratic_cutoff):
    """2-body single-element kernel between two force components and its
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
        (float, np.ndarray): Value of the 2-body kernel and its gradient
            with respect to the hyperparameters.
    """

    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[0]

    kernel, ls_derv, sig_derv = \
        two_body_grad_jit(env1.bond_array_2, env2.bond_array_2,
                          d1, d2, sig, ls, r_cut, cutoff_func)
    kernel_grad = np.array([sig_derv, ls_derv])
    return kernel, kernel_grad


def two_body_force_en(env1, env2, d1, hyps, cutoffs,
                      cutoff_func=cf.quadratic_cutoff):
    """2-body single-element kernel between a force component and a local
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

    # divide by two to account for double counting
    return two_body_force_en_jit(env1.bond_array_2, env2.bond_array_2,
                                 d1, sig, ls, r_cut, cutoff_func)/2


def two_body_en(env1, env2, hyps, cutoffs,
                cutoff_func=cf.quadratic_cutoff):
    """2-body single-element kernel between two local energies.

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

    return two_body_en_jit(env1.bond_array_2, env2.bond_array_2,
                           sig, ls, r_cut, cutoff_func)


# -----------------------------------------------------------------------------
#                              three body kernels
# -----------------------------------------------------------------------------


def three_body(env1, env2, d1, d2, hyps, cutoffs,
               cutoff_func=cf.quadratic_cutoff):
    """3-body single-element kernel between two force components.

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

    return three_body_jit(env1.bond_array_3, env2.bond_array_3,
                          env1.cross_bond_inds, env2.cross_bond_inds,
                          env1.cross_bond_dists, env2.cross_bond_dists,
                          env1.triplet_counts, env2.triplet_counts,
                          d1, d2, sig, ls, r_cut, cutoff_func)


def three_body_grad(env1, env2, d1, d2, hyps, cutoffs,
                    cutoff_func=cf.quadratic_cutoff):
    """3-body single-element kernel between two force components and its
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
        (float, np.ndarray): Value of the 3-body kernel and its gradient
            with respect to the hyperparameters.
    """
    sig = hyps[0]
    ls = hyps[1]
    r_cut = cutoffs[1]

    kernel, sig_derv, ls_derv = \
        three_body_grad_jit(env1.bond_array_3, env2.bond_array_3,
                            env1.cross_bond_inds, env2.cross_bond_inds,
                            env1.cross_bond_dists, env2.cross_bond_dists,
                            env1.triplet_counts, env2.triplet_counts,
                            d1, d2, sig, ls, r_cut, cutoff_func)

    kernel_grad = np.array([sig_derv, ls_derv])

    return kernel, kernel_grad


def three_body_force_en(env1, env2, d1, hyps, cutoffs,
                        cutoff_func=cf.quadratic_cutoff):
    """3-body single-element kernel between a force component and a local
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

    # divide by three to account for triple counting
    return three_body_force_en_jit(env1.bond_array_3, env2.bond_array_3,
                                   env1.cross_bond_inds, env2.cross_bond_inds,
                                   env1.cross_bond_dists,
                                   env2.cross_bond_dists,
                                   env1.triplet_counts, env2.triplet_counts,
                                   d1, sig, ls, r_cut, cutoff_func)/3


def three_body_en(env1, env2, hyps, cutoffs,
                  cutoff_func=cf.quadratic_cutoff):
    """3-body single-element kernel between two local energies.

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

    return three_body_en_jit(env1.bond_array_3, env2.bond_array_3,
                             env1.cross_bond_inds, env2.cross_bond_inds,
                             env1.cross_bond_dists, env2.cross_bond_dists,
                             env1.triplet_counts, env2.triplet_counts,
                             sig, ls, r_cut, cutoff_func)


# -----------------------------------------------------------------------------
#                           two body numba functions
# -----------------------------------------------------------------------------


@njit
def two_body_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                 r_cut, cutoff_func):
    """2-body single-element kernel between two force components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 2-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): 2-body bond array of the second local
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
    sig2 = sig*sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)

        for n in range(bond_array_2.shape[0]):
            rj = bond_array_2[n, 0]
            cj = bond_array_2[n, d2]
            fj, fdj = cutoff_func(r_cut, rj, cj)
            r11 = ri - rj

            A = ci * cj
            B = r11 * ci
            C = r11 * cj
            D = r11 * r11

            kern += force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2,
                                 ls3, sig2)

    return kern


@njit
def two_body_grad_jit(bond_array_1, bond_array_2, d1, d2, sig, ls,
                      r_cut, cutoff_func):
    """2-body single-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        bond_array_1 (np.ndarray): 2-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): 2-body bond array of the second local
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

    sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6 = grad_constants(sig, ls)

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)

        for n in range(bond_array_2.shape[0]):
            rj = bond_array_2[n, 0]
            cj = bond_array_2[n, d2]
            fj, fdj = cutoff_func(r_cut, rj, cj)

            r11 = ri - rj

            A = ci * cj
            B = r11 * ci
            C = r11 * cj
            D = r11 * r11

            kern_term, sig_term, ls_term = \
                grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4,
                            ls5, ls6, sig2, sig3)

            kern += kern_term
            sig_derv += sig_term
            ls_derv += ls_term

    return kern, ls_derv, sig_derv


@njit
def two_body_force_en_jit(bond_array_1, bond_array_2, d1, sig, ls, r_cut,
                          cutoff_func):
    """2-body single-element kernel between a force component and a local
    energy accelerated with Numba.

    Args:
        bond_array_1 (np.ndarray): 2-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): 2-body bond array of the second local
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
    sig2 = sig*sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        ci = bond_array_1[m, d1]
        fi, fdi = cutoff_func(r_cut, ri, ci)

        for n in range(bond_array_2.shape[0]):
            rj = bond_array_2[n, 0]
            fj, _ = cutoff_func(r_cut, rj, 0)

            r11 = ri - rj
            B = r11 * ci
            D = r11 * r11
            kern += force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2)

    return kern


@njit
def two_body_en_jit(bond_array_1, bond_array_2, sig, ls, r_cut, cutoff_func):
    """2-body single-element kernel between two local energies accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 2-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): 2-body bond array of the second local
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

        for n in range(bond_array_2.shape[0]):
            rj = bond_array_2[n, 0]
            fj, _ = cutoff_func(r_cut, rj, 0)
            r11 = ri - rj
            kern += fi * fj * sig2 * exp(-r11 * r11 * ls1)

    return kern


# -----------------------------------------------------------------------------
#                           three body numba functions
# -----------------------------------------------------------------------------


@njit
def three_body_jit(bond_array_1, bond_array_2,
                   cross_bond_inds_1, cross_bond_inds_2,
                   cross_bond_dists_1, cross_bond_dists_2,
                   triplets_1, triplets_2,
                   d1, d2, sig, ls, r_cut, cutoff_func):
    """3-body single-element kernel between two force components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
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
    sig2 = sig*sig
    ls1 = 1 / (2*ls*ls)
    ls2 = 1 / (ls*ls)
    ls3 = ls2*ls2

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m+n+1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)

            ri3 = cross_bond_dists_1[m, m+n+1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p+1+q]
                    rj2 = bond_array_2[ind2, 0]
                    cj2 = bond_array_2[ind2, d2]
                    fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)

                    rj3 = cross_bond_dists_2[p, p+1+q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)

                    fj = fj1*fj2*fj3
                    fdj = fdj1*fj2*fj3+fj1*fdj2*fj3

                    kern += triplet_kernel(ci1, ci2, cj1, cj2, ri1, ri2, ri3,
                                           rj1, rj2, rj3, fi, fj, fdi, fdj,
                                           ls1, ls2, ls3, sig2)
    return kern


@njit
def three_body_grad_jit(bond_array_1, bond_array_2,
                        cross_bond_inds_1, cross_bond_inds_2,
                        cross_bond_dists_1, cross_bond_dists_2,
                        triplets_1, triplets_2,
                        d1, d2, sig, ls, r_cut, cutoff_func):
    """3-body single-element kernel between two force components and its
    gradient with respect to the hyperparameters.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
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

    # pre-compute constants that appear in the inner loop
    sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6 = grad_constants(sig, ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m+n+1]
            ri3 = cross_bond_dists_1[m, m+n+1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]

            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                cj1 = bond_array_2[p, d2]
                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p+q+1]
                    rj3 = cross_bond_dists_2[p, p+q+1]
                    rj2 = bond_array_2[ind2, 0]
                    cj2 = bond_array_2[ind2, d2]

                    fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                    fj3, _ = cutoff_func(r_cut, rj3, 0)

                    fj = fj1*fj2*fj3
                    fdj = fdj1*fj2*fj3+fj1*fdj2*fj3

                    N, O, X = \
                        triplet_kernel_grad(ci1, ci2, cj1, cj2, ri1, ri2, ri3,
                                            rj1, rj2, rj3, fi, fj, fdi, fdj,
                                            ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                            sig3)

                    kern += N
                    sig_derv += O
                    ls_derv += X

    return kern, sig_derv, ls_derv


@njit
def three_body_force_en_jit(bond_array_1, bond_array_2,
                            cross_bond_inds_1,
                            cross_bond_inds_2,
                            cross_bond_dists_1,
                            cross_bond_dists_2,
                            triplets_1, triplets_2,
                            d1, sig, ls, r_cut, cutoff_func):
    """3-body single-element kernel between a force component and a local
    energy accelerated with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
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
    sig2 = sig*sig
    ls1 = 1 / (2*ls*ls)
    ls2 = 1 / (ls*ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ci1 = bond_array_1[m, d1]
        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m+n+1]
            ri2 = bond_array_1[ind1, 0]
            ci2 = bond_array_1[ind1, d1]
            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
            ri3 = cross_bond_dists_1[m, m+n+1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)
            fi = fi1*fi2*fi3
            fdi = fdi1*fi2*fi3+fi1*fdi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p+q+1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(r_cut, rj2, 0)
                    rj3 = cross_bond_dists_2[p, p+q+1]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    fj = fj1*fj2*fj3

                    kern += triplet_force_en_kernel(ci1, ci2, ri1, ri2, ri3,
                                                    rj1, rj2, rj3, fi, fj, fdi,
                                                    ls1, ls2, sig2)

    return kern


@njit
def three_body_en_jit(bond_array_1, bond_array_2,
                      cross_bond_inds_1,
                      cross_bond_inds_2,
                      cross_bond_dists_1,
                      cross_bond_dists_2,
                      triplets_1, triplets_2,
                      sig, ls, r_cut, cutoff_func):
    """3-body single-element kernel between two local energies accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
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

    sig2 = sig*sig
    ls2 = 1 / (2*ls*ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1, _ = cutoff_func(r_cut, ri1, 0)

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            fi2, _ = cutoff_func(r_cut, ri2, 0)

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)
            fi = fi1*fi2*fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(r_cut, rj2, 0)

                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    fj = fj1*fj2*fj3

                    r11 = ri1-rj1
                    r12 = ri1-rj2
                    r13 = ri1-rj3
                    r21 = ri2-rj1
                    r22 = ri2-rj2
                    r23 = ri2-rj3
                    r31 = ri3-rj1
                    r32 = ri3-rj2
                    r33 = ri3-rj3

                    C1 = r11*r11+r22*r22+r33*r33
                    C2 = r11*r11+r23*r23+r32*r32
                    C3 = r12*r12+r21*r21+r33*r33
                    C4 = r12*r12+r23*r23+r31*r31
                    C5 = r13*r13+r21*r21+r32*r32
                    C6 = r13*r13+r22*r22+r31*r31

                    k = exp(-C1*ls2)+exp(-C2*ls2)+exp(-C3*ls2)+exp(-C4*ls2) + \
                        exp(-C5*ls2)+exp(-C6*ls2)

                    kern += sig2*k*fi*fj

    return kern


# -----------------------------------------------------------------------------
#                            general helper functions
# -----------------------------------------------------------------------------


@njit
def grad_constants(sig, ls):
    sig2 = sig * sig
    sig3 = 2 * sig

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    ls4 = 1 / (ls * ls * ls)
    ls5 = ls * ls
    ls6 = ls2 * ls4

    return sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6


@njit
def force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2):
    """Helper function for computing the force/force kernel between two
    pairs or triplets of atoms of the same type.

    See Table IV of the SI of the FLARE paper for definitions of intermediate
    quantities.

    Returns:
        float: Force/force kernel between two pairs or triplets of atoms of
            the same type.
    """
    E = exp(-D * ls1)
    F = E * B * ls2
    G = -E * C * ls2
    H = A * E * ls2 - B * C * E * ls3
    I = E * fdi * fdj
    J = F * fi * fdj
    K = G * fdi * fj
    L = H * fi * fj
    M = sig2 * (I + J + K + L)

    return M


@njit
def grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6,
                sig2, sig3):
    E = exp(-D * ls1)
    F = E * B * ls2
    G = -E * C * ls2
    H = A * E * ls2 - B * C * E * ls3
    I = E * fdi * fdj
    J = F * fi * fdj
    K = G * fdi * fj
    L = H * fi * fj
    M = I + J + K + L
    N = sig2 * M
    O = sig3 * M
    P = E * D * ls4
    Q = B * (ls2 * P - 2 * E * ls4)
    R = -C * (ls2 * P - 2 * E * ls4)
    S = (A * ls5 - B * C) * (P * ls3 - 4 * E * ls6) + 2 * E * A * ls4
    T = P * fdi * fdj
    U = Q * fi * fdj
    V = R * fdi * fj
    W = S * fi * fj
    X = sig2 * (T + U + V + W)

    return N, O, X


@njit
def force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2):
    E = exp(-D * ls1)
    F = E * B * ls2
    G = -F * fi * fj
    H = -E * fdi * fj
    I = sig2 * (G + H)

    return I


# -----------------------------------------------------------------------------
#                        three body helper functions
# -----------------------------------------------------------------------------


@njit
def triplet_kernel(ci1, ci2, cj1, cj2, ri1, ri2, ri3, rj1, rj2, rj3, fi, fj,
                   fdi, fdj, ls1, ls2, ls3, sig2):
    r11 = ri1-rj1
    r12 = ri1-rj2
    r13 = ri1-rj3
    r21 = ri2-rj1
    r22 = ri2-rj2
    r23 = ri2-rj3
    r31 = ri3-rj1
    r32 = ri3-rj2
    r33 = ri3-rj3

    # sum over all six permutations
    M1 = three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M2 = three_body_helper_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M3 = three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M4 = three_body_helper_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M5 = three_body_helper_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)
    M6 = three_body_helper_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, sig2)

    return M1 + M2 + M3 + M4 + M5 + M6


@njit
def triplet_kernel_grad(ci1, ci2, cj1, cj2, ri1, ri2, ri3, rj1, rj2, rj3, fi,
                        fj, fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                        sig3):
    r11 = ri1-rj1
    r12 = ri1-rj2
    r13 = ri1-rj3
    r21 = ri2-rj1
    r22 = ri2-rj2
    r23 = ri2-rj3
    r31 = ri3-rj1
    r32 = ri3-rj2
    r33 = ri3-rj3

    N1, O1, X1 = \
        three_body_grad_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N2, O2, X2 = \
        three_body_grad_helper_2(ci2, ci1, cj2, cj1, r21, r13, r32, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N3, O3, X3 = \
        three_body_grad_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N4, O4, X4 = \
        three_body_grad_helper_1(ci1, ci2, cj2, cj1, r12, r21, r33, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N5, O5, X5 = \
        three_body_grad_helper_2(ci2, ci1, cj1, cj2, r22, r13, r31, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N6, O6, X6 = \
        three_body_grad_helper_2(ci1, ci2, cj2, cj1, r11, r23, r32, fi, fj,
                                 fdi, fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2,
                                 sig3)
    N = N1 + N2 + N3 + N4 + N5 + N6
    O = O1 + O2 + O3 + O4 + O5 + O6
    X = X1 + X2 + X3 + X4 + X5 + X6
    return N, O, X


@njit
def three_body_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33,
                        fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A = ci1*cj1+ci2*cj2
    B = r11*ci1+r22*ci2
    C = r11*cj1+r22*cj2
    D = r11*r11+r22*r22+r33*r33

    M = force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return M


@njit
def three_body_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31,
                        fi, fj, fdi, fdj,
                        ls1, ls2, ls3, sig2):
    A = ci1*cj2
    B = r12*ci1+r23*ci2
    C = r12*cj2+r31*cj1
    D = r12*r12+r23*r23+r31*r31

    M = force_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, sig2)

    return M


@njit
def three_body_grad_helper_1(ci1, ci2, cj1, cj2, r11, r22, r33, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2, sig3):
    A = ci1*cj1+ci2*cj2
    B = r11*ci1+r22*ci2
    C = r11*cj1+r22*cj2
    D = r11*r11+r22*r22+r33*r33

    N, O, X = grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4,
                          ls5, ls6, sig2, sig3)

    return N, O, X


@njit
def three_body_grad_helper_2(ci1, ci2, cj1, cj2, r12, r23, r31, fi, fj, fdi,
                             fdj, ls1, ls2, ls3, ls4, ls5, ls6, sig2, sig3):
    A = ci1*cj2
    B = r12*ci1+r23*ci2
    C = r12*cj2+r31*cj1
    D = r12*r12+r23*r23+r31*r31

    N, O, X = grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1, ls2, ls3, ls4,
                          ls5, ls6, sig2, sig3)

    return N, O, X


@njit
def three_body_en_helper(ci1, ci2, r11, r22, r33, fi, fj, fdi, ls1, ls2, sig2):
    B = r11 * ci1 + r22 * ci2
    D = r11 * r11 + r22 * r22 + r33 * r33

    return force_energy_helper(B, D, fi, fj, fdi, ls1, ls2, sig2)


@njit
def triplet_force_en_kernel(ci1, ci2, ri1, ri2, ri3, rj1, rj2, rj3,
                            fi, fj, fdi, ls1, ls2, sig2):
    r11 = ri1-rj1
    r12 = ri1-rj2
    r13 = ri1-rj3
    r21 = ri2-rj1
    r22 = ri2-rj2
    r23 = ri2-rj3
    r31 = ri3-rj1
    r32 = ri3-rj2
    r33 = ri3-rj3

    I1 = three_body_en_helper(ci1, ci2, r11, r22, r33, fi, fj,
                              fdi, ls1, ls2, sig2)
    I2 = three_body_en_helper(ci1, ci2, r13, r21, r32, fi, fj,
                              fdi, ls1, ls2, sig2)
    I3 = three_body_en_helper(ci1, ci2, r12, r23, r31, fi, fj,
                              fdi, ls1, ls2, sig2)
    I4 = three_body_en_helper(ci1, ci2, r12, r21, r33, fi, fj,
                              fdi, ls1, ls2, sig2)
    I5 = three_body_en_helper(ci1, ci2, r13, r22, r31, fi, fj,
                              fdi, ls1, ls2, sig2)
    I6 = three_body_en_helper(ci1, ci2, r11, r23, r32, fi, fj,
                              fdi, ls1, ls2, sig2)

    return I1 + I2 + I3 + I4 + I5 + I6


_str_to_kernel = {'two_body': two_body,
                  'two_body_en': two_body_en,
                  'two_body_force_en': two_body_force_en,
                  'three_body': three_body,
                  'three_body_en': three_body_en,
                  'three_body_force_en': three_body_force_en,
                  'two_plus_three_body': two_plus_three_body,
                  'two_plus_three_en': two_plus_three_en,
                  'two_plus_three_force_en': two_plus_three_force_en
                  }


def str_to_kernel(string: str, include_grad: bool = False):

    if string not in _str_to_kernel.keys():
        raise ValueError("Kernel {} not found in list of available "
                         "kernels{}:".format(string, _str_to_kernel.keys()))

    if not include_grad:
        return _str_to_kernel[string]
    else:
        if 'two' in string and 'three' in string:
            return _str_to_kernel[string], two_plus_three_body_grad
        elif 'two' in string and 'three' not in string:
            return _str_to_kernel[string], two_body_grad
        elif 'two' not in string and 'three' in string:
            return _str_to_kernel[string], three_body_grad
        else:
            raise ValueError("Gradient callable for {} not found".format(
                string))
