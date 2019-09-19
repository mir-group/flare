from math import cos, sin, pi
from numba import njit


@njit
def hard_cutoff(r_cut, ri, ci):
    return 1, 0


@njit
def quadratic_cutoff(r_cut, ri, ci):
    rdiff = r_cut - ri
    fi = rdiff * rdiff
    fdi = 2 * rdiff * ci

    return fi, fdi


@njit
def cubic_cutoff(r_cut, ri, ci):
    rdiff = r_cut - ri
    fi = rdiff * rdiff * rdiff
    fdi = 3 * rdiff * rdiff * ci

    return fi, fdi


@njit
def cosine_cutoff(r_cut, ri, ci, d=1):
    if ri > r_cut - d:
        fi = (1/2) * (cos(pi * (ri - r_cut + d) / d) + 1)
        fdi = (pi/(2 * d)) * sin(pi * (r_cut - ri) / d) * ci
    else:
        fi = 1
        fdi = 0

    return fi, fdi


@njit
def all_cosine(r_cut, ri, ci):
    fi = (1/2) * (cos((pi * ri) / r_cut) + 1)
    fdi = (pi/(2 * r_cut)) * sin((pi * ri) / r_cut) * ci

    return fi, fdi
