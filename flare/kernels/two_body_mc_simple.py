import numpy as np
from flare.kernels.kernels import force_helper, force_energy_helper, \
    grad_helper
from numba import njit
from flare.env import AtomicEnvironment
from typing import Callable
import flare.kernels.cutoffs as cf
from math import exp


class TwoBodyKernel:
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
        return efs_self(env1.bond_array_2, env1.ctype, env1.etypes,
                        self.signal_variance, self.length_scale,
                        self.cutoff, self.cutoff_func)

    def get_args(self, env1, env2):
        return (env1.bond_array_2, env1.ctype, env1.etypes,
                env2.bond_array_2, env2.ctype, env2.etypes,
                self.signal_variance, self.length_scale,
                self.cutoff, self.cutoff_func)

@njit
def energy_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
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

    # Divide by 4 to eliminate double counting (each pair will be counted
    # twice when summing over all environments in the structures).
    return kern / 4


@njit
def force_force(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                sig, ls, r_cut, cutoff_func):
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
    kernel_matrix = np.zeros((3, 3))

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                r11 = ri - rj
                D = r11 * r11

                # Note: Some redundancy here; can move this higher up.
                for d1 in range(3):
                    ci = bond_array_1[m, d1 + 1]
                    fi, fdi = cutoff_func(r_cut, ri, ci)

                    for d2 in range(3):
                        cj = bond_array_2[n, d2 + 1]
                        fj, fdj = cutoff_func(r_cut, rj, cj)

                        A = ci * cj
                        B = r11 * ci
                        C = r11 * cj

                        kernel_matrix[d1, d2] += \
                            force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                         ls1, ls2, ls3, sig2)

    return kernel_matrix


@njit
def force_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                 sig, ls, r_cut, cutoff_func):
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

    kern = np.zeros(3)

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # Check if species agree.
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)
                r11 = ri - rj
                D = r11 * r11

                for d1 in range(3):
                    ci = bond_array_1[m, d1 + 1]
                    fi, fdi = cutoff_func(r_cut, ri, ci)
                    B = r11 * ci
                    kern[d1] += \
                        force_energy_helper(B, D, fi, fj, fdi, ls1, ls2,
                                            sig2)

    return kern / 2


@njit
def stress_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                  sig, ls, r_cut, cutoff_func):
    """2-body multi-element kernel between a partial stress component and a
    local energy accelerated with Numba.

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
            Value of the 2-body partial-stress/energy kernel.
    """

    kern = np.zeros(6)

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # Check if the species agree.
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)
                r11 = ri - rj
                D = r11 * r11

                # Compute the force kernel.
                stress_count = 0
                for d1 in range(3):
                    ci = bond_array_1[m, d1 + 1]
                    B = r11 * ci
                    fi, fdi = cutoff_func(r_cut, ri, ci)
                    force_kern = \
                        force_energy_helper(B, D, fi, fj, fdi, ls1, ls2,
                                            sig2)

                    # Compute the stress kernel from the force kernel.
                    for d2 in range(d1, 3):
                        coordinate = bond_array_1[m, d2 + 1] * ri
                        kern[stress_count] -= force_kern * coordinate
                        stress_count += 1

    return kern / 4


@njit
def stress_force(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                 sig, ls, r_cut, cutoff_func):
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
        sig (float): 2-body signal variance hyperparameter.
        ls (float): 2-body length scale hyperparameter.
        r_cut (float): 2-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the 2-body kernel.
    """
    kernel_matrix = np.zeros((6, 3))

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                r11 = ri - rj

                stress_count = 0
                for d1 in range(3):
                    ci = bond_array_1[m, d1 + 1]
                    fi, fdi = cutoff_func(r_cut, ri, ci)
                    for d2 in range(d1, 3):
                        coordinate = bond_array_1[m, d2 + 1] * ri
                        for d3 in range(3):
                            cj = bond_array_2[n, d3 + 1]
                            fj, fdj = cutoff_func(r_cut, rj, cj)

                            A = ci * cj
                            B = r11 * ci
                            C = r11 * cj
                            D = r11 * r11

                            force_kern = \
                                force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                             ls1, ls2, ls3, sig2)
                            kernel_matrix[stress_count, d3] -= \
                                force_kern * coordinate

                        stress_count += 1

    return kernel_matrix / 2


@njit
def stress_stress(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                  sig, ls, r_cut, cutoff_func):
    """2-body multi-element kernel between two partial stress components
        accelerated with Numba.

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

    Return:
        float: Value of the 2-body kernel.
    """
    kernel_matrix = np.zeros((6, 6))

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                r11 = ri - rj
                D = r11 * r11

                s1 = 0
                for d1 in range(3):
                    ci = bond_array_1[m, d1 + 1]
                    B = r11 * ci
                    fi, fdi = cutoff_func(r_cut, ri, ci)
                    for d2 in range(d1, 3):
                        coordinate_1 = bond_array_1[m, d2 + 1] * ri
                        s2 = 0
                        for d3 in range(3):
                            cj = bond_array_2[n, d3 + 1]
                            A = ci * cj
                            C = r11 * cj
                            fj, fdj = cutoff_func(r_cut, rj, cj)
                            for d4 in range(d3, 3):
                                coordinate_2 = bond_array_2[n, d4 + 1] * rj
                                force_kern = \
                                    force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                                 ls1, ls2, ls3, sig2)
                                kernel_matrix[s1, s2] += \
                                    force_kern * coordinate_1 * \
                                    coordinate_2

                                s2 += 1
                        s1 += 1

    return kernel_matrix / 4


@njit
def force_force_gradient(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                         sig, ls, r_cut, cutoff_func):
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
        sig (float): 2-body signal variance hyperparameter.
        ls (float): 2-body length scale hyperparameter.
        r_cut (float): 2-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        (float, float):
            Value of the 2-body kernel and its gradient with respect to the
            hyperparameters.
    """

    kernel_matrix = np.zeros((3, 3))
    kernel_grad = np.zeros((2, 3, 3))

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
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                r11 = ri - rj
                D = r11 * r11

                for d1 in range(3):
                    ci = bond_array_1[m, d1 + 1]
                    B = r11 * ci
                    fi, fdi = cutoff_func(r_cut, ri, ci)

                    for d2 in range(3):
                        cj = bond_array_2[n, d2 + 1]
                        fj, fdj = cutoff_func(r_cut, rj, cj)

                        A = ci * cj
                        C = r11 * cj

                        kern_term, sig_term, ls_term = \
                            grad_helper(A, B, C, D, fi, fj, fdi, fdj, ls1,
                                        ls2, ls3, ls4, ls5, ls6, sig2, sig3)

                        kernel_matrix[d1, d2] += kern_term
                        kernel_grad[0, d1, d2] += sig_term
                        kernel_grad[1, d1, d2] += ls_term

    return kernel_matrix, kernel_grad


@njit
def efs_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
               sig, ls, r_cut, cutoff_func):

    energy_kernel = 0
    # TODO: add dtype to other zeros
    force_kernels = np.zeros(3)
    stress_kernels = np.zeros(6)

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        fi, _ = cutoff_func(r_cut, ri, 0)
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # Check if the species agree.
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)
                r11 = ri - rj
                D = r11 * r11

                energy_kernel += fi * fj * sig2 * exp(-D * ls1) / 4

                # Compute the force kernel.
                stress_count = 0
                for d1 in range(3):
                    ci = bond_array_1[m, d1 + 1]
                    B = r11 * ci
                    _, fdi = cutoff_func(r_cut, ri, ci)
                    force_kern = \
                        force_energy_helper(B, D, fi, fj, fdi, ls1, ls2,
                                            sig2)
                    force_kernels[d1] += force_kern / 2

                    # Compute the stress kernel from the force kernel.
                    for d2 in range(d1, 3):
                        coordinate = bond_array_1[m, d2 + 1] * ri
                        stress_kernels[stress_count] -= \
                            force_kern * coordinate / 4
                        stress_count += 1

    return energy_kernel, force_kernels, stress_kernels


@njit
def efs_force(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
              sig, ls, r_cut, cutoff_func):

    energy_kernels = np.zeros(3)
    force_kernels = np.zeros((3, 3))
    stress_kernels = np.zeros((6, 3))

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        fi, _ = cutoff_func(r_cut, ri, 0)
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)
                r11 = ri - rj
                D = r11 * r11

                for d1 in range(3):
                    cj = bond_array_2[n, d1 + 1]
                    _, fdj = cutoff_func(r_cut, rj, cj)

                    C = r11 * cj

                    energy_kernels[d1] += \
                        force_energy_helper(-C, D, fj, fi, fdj, ls1, ls2,
                                            sig2) / 2

                    stress_count = 0
                    for d3 in range(3):
                        ci = bond_array_1[m, d3 + 1]
                        _, fdi = cutoff_func(r_cut, ri, ci)
                        A = ci * cj
                        B = r11 * ci

                        force_kern = \
                            force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                         ls1, ls2, ls3, sig2)
                        force_kernels[d3, d1] += force_kern

                        for d2 in range(d3, 3):
                            coordinate = bond_array_1[m, d2 + 1] * ri
                            stress_kernels[stress_count, d1] -= \
                                force_kern * coordinate / 2

                            stress_count += 1

    return energy_kernels, force_kernels, stress_kernels


@njit
def efs_self(bond_array_1, c1, etypes1, sig, ls, r_cut, cutoff_func):

    energy_kernel = 0
    force_kernels = np.zeros(3)
    stress_kernels = np.zeros(6)

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    ls3 = ls2 * ls2
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        fi, _ = cutoff_func(r_cut, ri, 0)
        e1 = etypes1[m]

        for n in range(bond_array_1.shape[0]):
            e2 = etypes1[n]

            # check if bonds agree
            if (e1 == e2) or (c1 == e2 and c1 == e1):
                rj = bond_array_1[n, 0]
                fj, _ = cutoff_func(r_cut, rj, 0)
                r11 = ri - rj
                D = r11 * r11

                energy_kernel += fi * fj * sig2 * exp(-D * ls1) / 4

                stress_count = 0
                for d1 in range(3):
                    cj = bond_array_1[n, d1 + 1]
                    _, fdj = cutoff_func(r_cut, rj, cj)
                    C = r11 * cj

                    ci = bond_array_1[m, d1 + 1]
                    _, fdi = cutoff_func(r_cut, ri, ci)
                    A = ci * cj
                    B = r11 * ci

                    force_kern = \
                        force_helper(A, B, C, D, fi, fj, fdi, fdj,
                                     ls1, ls2, ls3, sig2)
                    force_kernels[d1] += force_kern

                    for d2 in range(d1, 3):
                        coord1 = bond_array_1[m, d2 + 1] * ri
                        coord2 = bond_array_1[n, d2 + 1] * rj
                        stress_kernels[stress_count] += \
                            force_kern * coord1 * coord2 / 4

                        stress_count += 1

    return energy_kernel, force_kernels, stress_kernels
