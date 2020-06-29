import numpy as np
from flare.kernels.kernels import force_helper, force_energy_helper, \
    grad_helper, three_body_fe_perm, three_body_ee_perm, \
    three_body_se_perm, three_body_ff_perm, three_body_sf_perm, \
    three_body_ss_perm, three_body_grad_perm, grad_constants
from numba import njit
from flare.env import AtomicEnvironment
from typing import Callable
import flare.kernels.cutoffs as cf
from math import exp


class ThreeBodyKernel:
    def __init__(self, hyperparameters: 'ndarray', cutoff: float,
                 cutoff_func: Callable = cf.quadratic_cutoff):
        self.hyperparameters = hyperparameters
        self.signal_variance = hyperparameters[0]
        self.length_scale = hyperparameters[1]
        self.cutoff = cutoff
        self.cutoff_func = cutoff_func

    def energy_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return energy_energy(*args)

    def force_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return force_energy(*args)

    def stress_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return stress_energy(*args)

    def force_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return force_force(*args)

    def stress_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return stress_force(*args)

    def stress_stress(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return stress_stress(*args)

    def force_force_gradient(self, env1: AtomicEnvironment,
                             env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return force_force_gradient(*args)

    def efs_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return efs_energy(*args)

    def efs_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        args = self.get_args(env1, env2)
        return efs_force(*args)

    def efs_self(self, env1: AtomicEnvironment):
        return efs_self(env1.bond_array_3, env1.ctype, env1.etypes,
                        env1.cross_bond_inds, env1.cross_bond_dists,
                        env1.triplet_counts, self.signal_variance,
                        self.length_scale, self.cutoff, self.cutoff_func)
    
    def get_args(self, env1, env2):
        return (env1.bond_array_3, env1.ctype, env1.etypes,
                env2.bond_array_3, env2.ctype, env2.etypes,
                env1.cross_bond_inds, env2.cross_bond_inds,
                env1.cross_bond_dists, env2.cross_bond_dists,
                env1.triplet_counts, env2.triplet_counts,
                self.signal_variance, self.length_scale,
                self.cutoff, self.cutoff_func)


@njit
def energy_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                  cross_bond_inds_1, cross_bond_inds_2,
                  cross_bond_dists_1, cross_bond_dists_2,
                  triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
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

                    kern += \
                        three_body_ee_perm(r11, r12, r13, r21, r22, r23, r31,
                                           r32, r33, c1, c2, ei1, ei2, ej1,
                                           ej2, fi, fj, ls2, sig2)

    return kern / 9


@njit
def force_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                 cross_bond_inds_1, cross_bond_inds_2,
                 cross_bond_dists_1, cross_bond_dists_2,
                 triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
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
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """
    kern = np.zeros(3)

    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ei1 = etypes1[m]

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

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

                    for d1 in range(3):
                        ci1 = bond_array_1[m, d1 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d1 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fi = fi1 * fi2 * fi3
                        fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

                        kern[d1] += \
                            three_body_fe_perm(r11, r12, r13, r21, r22, r23,
                                               r31, r32, r33, c1, c2, ci1, ci2,
                                               ei1, ei2, ej1, ej2, fi, fj, fdi,
                                               ls1, ls2, sig2)

    return kern / 3


@njit
def stress_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                  cross_bond_inds_1, cross_bond_inds_2,
                  cross_bond_dists_1, cross_bond_dists_2,
                  triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
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
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """
    kern = np.zeros(6)

    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

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

                    stress_count = 0
                    for d1 in range(3):
                        ci1 = bond_array_1[m, d1 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d1 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fdi_p1 = fdi1 * fi2 * fi3
                        fdi_p2 = fi1 * fdi2 * fi3
                        fdi = fdi_p1 + fdi_p2

                        for d2 in range(d1, 3):
                            coord1 = bond_array_1[m, d2 + 1] * ri1
                            coord2 = bond_array_1[ind1, d2 + 1] * ri2

                            kern[stress_count] += \
                                three_body_se_perm(r11, r12, r13, r21, r22,
                                                   r23, r31, r32, r33, c1, c2,
                                                   ci1, ci2, ei1, ei2, ej1,
                                                   ej2, fi, fj, fdi, ls1, ls2,
                                                   sig2, coord1, coord2,
                                                   fdi_p1, fdi_p2)

                            stress_count += 1

    return kern / 6


@njit
def force_force(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                cross_bond_inds_1, cross_bond_inds_2,
                cross_bond_dists_1, cross_bond_dists_2,
                triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
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
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the 3-body kernel.
    """
    kern = np.zeros((3, 3))

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    # first loop over the first 3-body environment
    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ei1 = etypes1[m]

        # second loop over the first 3-body environment
        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            # first loop over the second 3-body environment
            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                ej1 = etypes2[p]

                # second loop over the second 3-body environment
                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + 1 + q]
                    rj2 = bond_array_2[ind2, 0]
                    rj3 = cross_bond_dists_2[p, p + 1 + q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    ej2 = etypes2[ind2]

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    for d1 in range(3):
                        ci1 = bond_array_1[m, d1 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d1 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fi = fi1 * fi2 * fi3
                        fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

                        for d2 in range(3):
                            cj1 = bond_array_2[p, d2 + 1]
                            fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                            cj2 = bond_array_2[ind2, d2 + 1]
                            fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                            fj = fj1 * fj2 * fj3
                            fdj = fdj1 * fj2 * fj3 + fj1 * fdj2 * fj3

                            kern[d1, d2] += \
                                three_body_ff_perm(r11, r12, r13, r21, r22,
                                                   r23, r31, r32, r33, c1, c2,
                                                   ci1, ci2, cj1, cj2, ei1,
                                                   ei2, ej1, ej2, fi, fj, fdi,
                                                   fdj, ls1, ls2, ls3, sig2)

    return kern


@njit
def stress_force(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                 cross_bond_inds_1, cross_bond_inds_2,
                 cross_bond_dists_1, cross_bond_dists_2,
                 triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
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
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the 3-body kernel.
    """
    kern = np.zeros((6, 3))

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    # first loop over the first 3-body environment
    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ei1 = etypes1[m]

        # second loop over the first 3-body environment
        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            # first loop over the second 3-body environment
            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                ej1 = etypes2[p]

                # second loop over the second 3-body environment
                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + 1 + q]
                    rj2 = bond_array_2[ind2, 0]
                    rj3 = cross_bond_dists_2[p, p + 1 + q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    ej2 = etypes2[ind2]

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    stress_count = 0
                    for d1 in range(3):
                        ci1 = bond_array_1[m, d1 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d1 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fi = fi1 * fi2 * fi3
                        fdi_p1 = fdi1 * fi2 * fi3
                        fdi_p2 = fi1 * fdi2 * fi3
                        fdi = fdi_p1 + fdi_p2

                        for d2 in range(d1, 3):
                            coord1 = bond_array_1[m, d2 + 1] * ri1
                            coord2 = bond_array_1[ind1, d2 + 1] * ri2

                            for d3 in range(3):
                                cj1 = bond_array_2[p, d3 + 1]
                                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                                cj2 = bond_array_2[ind2, d3 + 1]
                                fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                                fj = fj1 * fj2 * fj3
                                fdj = fdj1 * fj2 * fj3 + fj1 * fdj2 * fj3

                                kern[stress_count, d3] += \
                                    three_body_sf_perm(r11, r12, r13, r21, r22,
                                                       r23, r31, r32, r33, c1,
                                                       c2, ci1, ci2, cj1, cj2,
                                                       ei1, ei2, ej1, ej2, fi,
                                                       fj, fdi, fdj, ls1, ls2,
                                                       ls3, sig2, coord1,
                                                       coord2, fdi_p1, fdi_p2)
                            stress_count += 1

    return kern / 2


@njit
def stress_stress(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                  cross_bond_inds_1, cross_bond_inds_2,
                  cross_bond_dists_1, cross_bond_dists_2,
                  triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
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
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the 3-body kernel.
    """
    kern = np.zeros((6, 6))

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    # first loop over the first 3-body environment
    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ei1 = etypes1[m]

        # second loop over the first 3-body environment
        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            # first loop over the second 3-body environment
            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                ej1 = etypes2[p]

                # second loop over the second 3-body environment
                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + 1 + q]
                    rj2 = bond_array_2[ind2, 0]
                    rj3 = cross_bond_dists_2[p, p + 1 + q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    ej2 = etypes2[ind2]

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    stress_count_1 = 0
                    for d1 in range(3):
                        ci1 = bond_array_1[m, d1 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d1 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fi = fi1 * fi2 * fi3
                        fdi_p1 = fdi1 * fi2 * fi3
                        fdi_p2 = fi1 * fdi2 * fi3
                        fdi = fdi_p1 + fdi_p2

                        for d2 in range(d1, 3):
                            coord1 = bond_array_1[m, d2 + 1] * ri1
                            coord2 = bond_array_1[ind1, d2 + 1] * ri2

                            stress_count_2 = 0
                            for d3 in range(3):
                                cj1 = bond_array_2[p, d3 + 1]
                                fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                                cj2 = bond_array_2[ind2, d3 + 1]
                                fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                                fj = fj1 * fj2 * fj3
                                fdj_p1 = fdj1 * fj2 * fj3
                                fdj_p2 = fj1 * fdj2 * fj3
                                fdj = fdj_p1 + fdj_p2

                                for d4 in range(d3, 3):
                                    coord3 = bond_array_2[p, d4 + 1] * rj1
                                    coord4 = bond_array_2[ind2, d4 + 1] * rj2

                                    kern[stress_count_1, stress_count_2] += \
                                        three_body_ss_perm(r11, r12, r13, r21,
                                                           r22, r23, r31, r32,
                                                           r33, c1, c2, ci1,
                                                           ci2, cj1, cj2, ei1,
                                                           ei2, ej1, ej2, fi,
                                                           fj, fdi, fdj, ls1,
                                                           ls2, ls3, sig2,
                                                           coord1, coord2,
                                                           coord3, coord4,
                                                           fdi_p1, fdi_p2,
                                                           fdj_p1, fdj_p2)
                                    stress_count_2 += 1
                            stress_count_1 += 1

    return kern / 4


@njit
def force_force_gradient(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                         cross_bond_inds_1, cross_bond_inds_2,
                         cross_bond_dists_1, cross_bond_dists_2,
                         triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
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
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        (float, float):
            Value of the 3-body kernel and its gradient with respect to the
            hyperparameters.
    """
    kernel_matrix = np.zeros((3, 3))
    kernel_grad = np.zeros((2, 3, 3))

    # pre-compute constants that appear in the inner loop
    sig2, sig3, ls1, ls2, ls3, ls4, ls5, ls6 = grad_constants(sig, ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ei1 = etypes1[m]

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri3 = cross_bond_dists_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ei2 = etypes1[ind1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    ej2 = etypes2[ind2]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    for d1 in range(3):
                        ci1 = bond_array_1[m, d1 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d1 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3
                        fi = fi1 * fi2 * fi3

                        for d2 in range(3):
                            cj1 = bond_array_2[p, d2 + 1]
                            fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                            cj2 = bond_array_2[ind2, d2 + 1]
                            fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                            fdj = fdj1 * fj2 * fj3 + fj1 * fdj2 * fj3
                            fj = fj1 * fj2 * fj3

                            kern_term, sig_term, ls_term = \
                                three_body_grad_perm(r11, r12, r13, r21, r22,
                                                     r23, r31, r32, r33, c1,
                                                     c2, ci1, ci2, cj1, cj2,
                                                     ei1, ei2, ej1, ej2, fi,
                                                     fj, fdi, fdj, ls1, ls2,
                                                     ls3, ls4, ls5, ls6, sig2,
                                                     sig3)

                            kernel_matrix[d1, d2] += kern_term
                            kernel_grad[0, d1, d2] += sig_term
                            kernel_grad[1, d1, d2] += ls_term

    return kernel_matrix, kernel_grad


@njit
def efs_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
               cross_bond_inds_1, cross_bond_inds_2,
               cross_bond_dists_1, cross_bond_dists_2,
               triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):

    energy_kernel = 0
    force_kernels = np.zeros(3)
    stress_kernels = np.zeros(6)

    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

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

                    energy_kernel += \
                        three_body_ee_perm(r11, r12, r13, r21, r22, r23, r31,
                                           r32, r33, c1, c2, ei1, ei2, ej1,
                                           ej2, fi, fj, ls1, sig2) / 9

                    stress_count = 0
                    for d1 in range(3):
                        ci1 = bond_array_1[m, d1 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d1 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fi = fi1 * fi2 * fi3
                        fdi_p1 = fdi1 * fi2 * fi3
                        fdi_p2 = fi1 * fdi2 * fi3
                        fdi = fdi_p1 + fdi_p2

                        force_kernels[d1] += \
                            three_body_fe_perm(r11, r12, r13, r21, r22, r23,
                                               r31, r32, r33, c1, c2, ci1, ci2,
                                               ei1, ei2, ej1, ej2, fi, fj, fdi,
                                               ls1, ls2, sig2) / 3

                        for d2 in range(d1, 3):
                            coord1 = bond_array_1[m, d2 + 1] * ri1
                            coord2 = bond_array_1[ind1, d2 + 1] * ri2

                            stress_kernels[stress_count] += \
                                three_body_se_perm(r11, r12, r13, r21, r22,
                                                   r23, r31, r32, r33, c1, c2,
                                                   ci1, ci2, ei1, ei2, ej1,
                                                   ej2, fi, fj, fdi, ls1, ls2,
                                                   sig2, coord1, coord2,
                                                   fdi_p1, fdi_p2) / 6

                            stress_count += 1

    return energy_kernel, force_kernels, stress_kernels


@njit
def efs_force(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
              cross_bond_inds_1, cross_bond_inds_2,
              cross_bond_dists_1, cross_bond_dists_2,
              triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):

    energy_kernels = np.zeros(3)
    force_kernels = np.zeros((3, 3))
    stress_kernels = np.zeros((6, 3))

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

    # first loop over the first 3-body environment
    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1, _ = cutoff_func(r_cut, ri1, 0)
        ei1 = etypes1[m]

        # second loop over the first 3-body environment
        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            fi2, _ = cutoff_func(r_cut, ri2, 0)
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)
            fi = fi1 * fi2 * fi3

            # first loop over the second 3-body environment
            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)
                ej1 = etypes2[p]

                # second loop over the second 3-body environment
                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + 1 + q]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(r_cut, rj2, 0)
                    rj3 = cross_bond_dists_2[p, p + 1 + q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    ej2 = etypes2[ind2]

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    for d3 in range(3):
                        cj1 = bond_array_2[p, d3 + 1]
                        fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                        cj2 = bond_array_2[ind2, d3 + 1]
                        fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                        fj = fj1 * fj2 * fj3
                        fdj = fdj1 * fj2 * fj3 + fj1 * fdj2 * fj3

                        energy_kernels[d3] += \
                            three_body_fe_perm(r11, r21, r31, r12, r22, r32,
                                               r13, r23, r33, c2, c1, -cj1,
                                               -cj2, ej1, ej2, ei1, ei2, fj,
                                               fi, fdj, ls1, ls2, sig2) / 3

                        stress_count = 0
                        for d1 in range(3):
                            ci1 = bond_array_1[m, d1 + 1]
                            fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                            ci2 = bond_array_1[ind1, d1 + 1]
                            fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                            fi = fi1 * fi2 * fi3
                            fdi_p1 = fdi1 * fi2 * fi3
                            fdi_p2 = fi1 * fdi2 * fi3
                            fdi = fdi_p1 + fdi_p2

                            force_kernels[d1, d3] += \
                                three_body_ff_perm(r11, r12, r13, r21, r22,
                                                   r23, r31, r32, r33, c1, c2,
                                                   ci1, ci2, cj1, cj2, ei1,
                                                   ei2, ej1, ej2, fi, fj, fdi,
                                                   fdj, ls1, ls2, ls3, sig2)

                            for d2 in range(d1, 3):
                                coord1 = bond_array_1[m, d2 + 1] * ri1
                                coord2 = bond_array_1[ind1, d2 + 1] * ri2

                                stress_kernels[stress_count, d3] += \
                                    three_body_sf_perm(r11, r12, r13, r21, r22,
                                                       r23, r31, r32, r33, c1,
                                                       c2, ci1, ci2, cj1, cj2,
                                                       ei1, ei2, ej1, ej2, fi,
                                                       fj, fdi, fdj, ls1, ls2,
                                                       ls3, sig2, coord1,
                                                       coord2, fdi_p1,
                                                       fdi_p2) / 2
                                stress_count += 1

    return energy_kernels, force_kernels, stress_kernels


@njit
def efs_self(bond_array_1, c1, etypes1, cross_bond_inds_1, cross_bond_dists_1,
             triplets_1, sig, ls, r_cut, cutoff_func):

    energy_kernel = 0
    force_kernels = np.zeros(3)
    stress_kernels = np.zeros(6)

    # pre-compute constants that appear in the inner loop
    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2

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

            for p in range(bond_array_1.shape[0]):
                rj1 = bond_array_1[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)
                ej1 = etypes1[p]

                for q in range(triplets_1[p]):
                    ind2 = cross_bond_inds_1[p, p + 1 + q]
                    rj2 = bond_array_1[ind2, 0]
                    fj2, _ = cutoff_func(r_cut, rj2, 0)
                    rj3 = cross_bond_dists_1[p, p + 1 + q]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    fj = fj1 * fj2 * fj3
                    ej2 = etypes1[ind2]

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    energy_kernel += \
                        three_body_ee_perm(r11, r12, r13, r21, r22, r23, r31,
                                           r32, r33, c1, c1, ei1, ei2, ej1,
                                           ej2, fi, fj, ls1, sig2) / 9

                    stress_count = 0
                    for d3 in range(3):
                        cj1 = bond_array_1[p, d3 + 1]
                        fj1, fdj1 = cutoff_func(r_cut, rj1, cj1)
                        cj2 = bond_array_1[ind2, d3 + 1]
                        fj2, fdj2 = cutoff_func(r_cut, rj2, cj2)
                        fdj_p1 = fdj1 * fj2 * fj3
                        fdj_p2 = fj1 * fdj2 * fj3
                        fdj = fdj_p1 + fdj_p2

                        ci1 = bond_array_1[m, d3 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d3 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fi = fi1 * fi2 * fi3
                        fdi_p1 = fdi1 * fi2 * fi3
                        fdi_p2 = fi1 * fdi2 * fi3
                        fdi = fdi_p1 + fdi_p2

                        force_kernels[d3] += \
                            three_body_ff_perm(r11, r12, r13, r21, r22,
                                               r23, r31, r32, r33, c1, c1,
                                               ci1, ci2, cj1, cj2, ei1,
                                               ei2, ej1, ej2, fi, fj, fdi,
                                               fdj, ls1, ls2, ls3, sig2)

                        for d2 in range(d3, 3):
                            coord1 = bond_array_1[m, d2 + 1] * ri1
                            coord2 = bond_array_1[ind1, d2 + 1] * ri2
                            coord3 = bond_array_1[p, d2 + 1] * rj1
                            coord4 = bond_array_1[ind2, d2 + 1] * rj2

                            stress_kernels[stress_count] += \
                                three_body_ss_perm(r11, r12, r13, r21, r22,
                                                   r23, r31, r32, r33, c1, c1,
                                                   ci1, ci2, cj1, cj2, ei1,
                                                   ei2, ej1, ej2, fi, fj, fdi,
                                                   fdj, ls1, ls2, ls3, sig2,
                                                   coord1, coord2, coord3,
                                                   coord4, fdi_p1, fdi_p2,
                                                   fdj_p1, fdj_p2) / 4
                            stress_count += 1

    return energy_kernel, force_kernels, stress_kernels
