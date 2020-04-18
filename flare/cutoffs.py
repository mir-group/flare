"""
The cutoffs module gives a few different options for smoothly sending the GP
kernel to zero near the boundary of the cutoff sphere.
"""
from math import cos, sin, pi
from numba import njit


@njit
def hard_cutoff(r_cut: float, ri: float, ci: float):
    """A hard cutoff that assigns a value of 1 to all interatomic distances.

    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Cartesian coordinate divided by the distance.

    Returns:
        (float, float): Cutoff value and its derivative.
    """
    return 1, 0

@njit
def quadratic_cutoff_bound(r_cut: float, ri: float, ci: float):
    """A quadratic cutoff that goes to zero smoothly at the cutoff boundary.

    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Cartesian coordinate divided by the distance.

    Returns:
        (float, float): Cutoff value and its derivative.
    """

    if (r_cut > ri):
        rdiff = r_cut - ri
        fi = rdiff * rdiff
        fdi = 2 * rdiff * ci
    else:
        fi = 0
        fdi = 0

    return fi, fdi


@njit
def quadratic_cutoff(r_cut: float, ri: float, ci: float):
    """A quadratic cutoff that goes to zero smoothly at the cutoff boundary.

    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Cartesian coordinate divided by the distance.

    Returns:
        (float, float): Cutoff value and its derivative.
    """

    rdiff = r_cut - ri
    fi = rdiff * rdiff
    fdi = 2 * rdiff * ci

    return fi, fdi


@njit
def cubic_cutoff(r_cut: float, ri: float, ci: float):
    """A cubic cutoff that goes to zero smoothly at the cutoff boundary.

    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Cartesian coordinate divided by the distance.

    Returns:
        (float, float): Cutoff value and its derivative.
    """

    rdiff = r_cut - ri
    fi = rdiff * rdiff * rdiff
    fdi = 3 * rdiff * rdiff * ci

    return fi, fdi


@njit
def cosine_cutoff(r_cut: float, ri: float, ci: float, d: float = 1):
    """A cosine cutoff that returns 1 up to r_cut - d, and assigns a cosine
    envelope to values of r between r_cut - d and r_cut. Based on Eq. 24 of
    Albert P. Bartók and Gábor Csányi. "Gaussian approximation potentials: A
    brief tutorial introduction." International Journal of Quantum Chemistry
    115.16 (2015): 1051-1057.

    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Cartesian coordinate divided by the distance.

    Returns:
        (float, float): Cutoff value and its derivative.
    """

    if ri > r_cut - d:
        fi = (1/2) * (cos(pi * (ri - r_cut + d) / d) + 1)
        fdi = (pi/(2 * d)) * sin(pi * (r_cut - ri) / d) * ci
    else:
        fi = 1
        fdi = 0

    return fi, fdi
