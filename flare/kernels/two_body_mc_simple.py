import numpy as np
from flare.kernels.kernels import force_helper
from numba import njit
from flare.env import AtomicEnvironment
from typing import Callable
import flare.cutoffs as cf
from math import exp


class TwoBodyKernel:
    def __init__(self, signal_variance: float, length_scale: float,
                 cutoff: float, cutoff_func: Callable = cf.quadratic_cutoff):
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        self.cutoff = cutoff
        self.cutoff_func = cutoff_func

    def energy_energy_kernel(self, env1: AtomicEnvironment,
                             env2: AtomicEnvironment):

        return two_body_mc_en_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                                  env2.bond_array_2, env2.ctype, env2.etypes,
                                  self.signal_variance, self.length_scale,
                                  self.cutoff, self.cutoff_func)

    def force_energy_kernel(self, env1: AtomicEnvironment,
                            env2: AtomicEnvironment,
                            d1: int, hyps: 'ndarray', cutoffs: 'ndarray',
                            cutoff_func: Callable = cf.quadratic_cutoff):
        pass

    def stress_energy_kernel(self):
        pass

    def force_force_kernel(self, env1: AtomicEnvironment,
                           env2: AtomicEnvironment):

        return two_body_mc_jit(env1.bond_array_2, env1.ctype, env1.etypes,
                               env2.bond_array_2, env2.ctype, env2.etypes,
                               self.signal_variance, self.length_scale,
                               self.cutoff, self.cutoff_func)

    def stress_force_kernel(self):
        pass

    def stress_stress_kernel(self):
        pass

    def force_force_gradient(self):
        pass


@njit
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

                # Divide by 4 to eliminate double counting (each pair will be
                # counted twice when summing over all environments in the
                # structures)
                kern += fi * fj * sig2 * exp(-r11 * r11 * ls1) / 4

    return kern


@njit
def two_body_mc_jit(bond_array_1, c1, etypes1,
                    bond_array_2, c2, etypes2,
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
