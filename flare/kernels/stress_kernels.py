import numpy as np
from numba import njit
from math import exp


# -----------------------------------------------------------------------------
#                                2-body kernels
# -----------------------------------------------------------------------------

@njit
def two_body_energy(bond_array_1, c1, etypes1,
                    bond_array_2, c2, etypes2,
                    sig, ls, r_cut, cutoff_func):
    """Compute the 2-body covariance between the local energy of one
        environment and the local energy, forces, and partial stresses (times
        the structure volume) of a second environment."""

    energy_kernel = 0.
    force_kernels = np.zeros(3)
    stress_kernels = np.zeros(6)

    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)
    sig2 = sig * sig

    for m in range(bond_array_1.shape[0]):
        ri = bond_array_1[m, 0]
        fi, _ = cutoff_func(r_cut, ri, 1)
        e1 = etypes1[m]

        for n in range(bond_array_2.shape[0]):
            e2 = etypes2[n]

            # check if bonds agree
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rj = bond_array_2[n, 0]
                xrel = bond_array_2[n, 1]
                yrel = bond_array_2[n, 2]
                zrel = bond_array_2[n, 3]
                xval = xrel * rj
                yval = yrel * rj
                zval = zrel * rj
                fj, fdj = cutoff_func(r_cut, rj, 1)

                r11 = ri - rj

                # local energy kernel
                cons1 = r11 * r11
                cons2 = exp(-cons1 * ls1)
                energy_kernel += fi * fj * sig2 * cons2 / 4

                # helper constants
                cons3 = cons2 * ls2 * fi * fj * r11
                cons4 = cons2 * fi * fdj

                # fx + exx, exy, exz stress components
                fx = xrel * cons3 - cons4 * xrel
                force_kernels[0] += sig2 * fx / 2
                stress_kernels[0] -= sig2 * fx * xval / 4
                stress_kernels[1] -= sig2 * fx * yval / 4
                stress_kernels[2] -= sig2 * fx * zval / 4

                # fy + eyy, eyz stress components
                fy = yrel * cons3 - cons4 * yrel
                force_kernels[1] += sig2 * fy / 2
                stress_kernels[3] -= sig2 * fy * yval / 4
                stress_kernels[4] -= sig2 * fy * zval / 4

                # fz + ezz stress component
                fz = zrel * cons3 - cons4 * zrel
                force_kernels[2] += sig2 * fz / 2
                stress_kernels[5] -= sig2 * fz * zval / 4

    return energy_kernel, force_kernels, stress_kernels
