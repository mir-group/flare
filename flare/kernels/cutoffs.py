"""
The cutoffs module gives a few different options for smoothly sending the GP
kernel to zero near the boundary of the cutoff sphere.
"""
from math import cos, sin, pi, sqrt
from numba import njit
from typing import Tuple

quarterpi = 1 / (4 * pi)


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
def quadratic_cutoff_bound(r_cut: float, ri: float, ci: float) -> Tuple[float, float]:
    """A quadratic cutoff that goes to zero smoothly at the cutoff boundary.

    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Cartesian coordinate divided by the distance.

    Returns:
        (float, float): Cutoff value and its derivative.
    """

    if r_cut > ri:
        rdiff = r_cut - ri
        fi = rdiff * rdiff
        fdi = 2 * rdiff * ci
    else:
        fi = 0
        fdi = 0

    return fi, fdi


@njit
def quadratic_cutoff(r_cut: float, ri: float, ci: float) -> Tuple[float, float]:
    """A quadratic cutoff that goes to zero smoothly at the cutoff boundary.

    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Cartesian coordinate divided by the distance. Scales derivative.

    Returns:
        (float, float): Cutoff value and its derivative.
    """

    rdiff = r_cut - ri
    fi = rdiff * rdiff
    fdi = 2 * rdiff * ci

    return fi, fdi


@njit
def cubic_cutoff(r_cut: float, ri: float, ci: float) -> Tuple[float, float]:
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
def cosine_cutoff(
    r_cut: float, ri: float, ci: float, d: float = 1
) -> Tuple[float, float]:
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
        fi = (1 / 2) * (cos(pi * (ri - r_cut + d) / d) + 1)
        fdi = (pi / (2 * d)) * sin(pi * (r_cut - ri) / d) * ci
    else:
        fi = 1
        fdi = 0

    return fi, fdi


@njit
def sqrt_cutoff(r_cut: float, ri: float, ci: float) -> Tuple[float, float]:
    """
    Uses cutoff as sqrt(r_cut-ri).
    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Dummy variable for argument compatibility.
    Returns:
        (float, float): Cutoff value and its derivative.
    """

    if ri < r_cut:

        f1 = sqrt(r_cut - ri)
        return f1, -0.5 / f1

    else:
        fi = 0
        fdi = 0

    return fi, fdi


@njit
def spherical_cutoff(r_cut: float, ri: float, ci: float) -> Tuple[float, float]:
    """
    A cutoff which divides by the surface area of a sphere
    out to the point ri, and 0 if r_cut exceeds ri.
    Args:
        r_cut (float): Cutoff value (in angstrom).
        ri (float): Interatomic distance.
        ci (float): Dummy variable for argument compatibility.
    Returns:
        (float, float): Cutoff value and its derivative.
    """

    if ri > r_cut:
        return 0, 0

    else:
        # 1/(4pi) precomputed as 'quarterpi'.
        # Return 1/(4 pi r^2) for cutoff
        # and -2 /(4 pi r^3) for derivative
        res = quarterpi / (ri * ri)
        return res, -2 * res / ri
