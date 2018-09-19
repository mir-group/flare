from scipy.special import sph_harm
from scipy.special import spherical_in
from scipy.special import erf
import numpy as np
from math import exp
import math


def sph_coeff(atom_pos, r, l, m, alpha):
    ri = np.linalg.norm(atom_pos)
    theta = math.acos(atom_pos[2] / ri)
    phi = math.atan2(atom_pos[1], atom_pos[0])
    c_lm = 4 * np.pi * exp(-alpha * (r**2 + ri**2)) * \
        spherical_in(l, 2*alpha*r*ri) * \
        np.conj(sph_harm(m, l, phi, theta))

    return c_lm


def estimate_rho(pos, atom_pos_list, alpha, max_l):
    r = np.linalg.norm(pos)
    theta = math.acos(pos[2] / r)
    phi = math.atan2(pos[1], pos[0])

    rho_est = 0
    for p in range(len(atom_pos_list)):
        atom_pos = atom_pos_list[p]
        for l in range(max_l + 1):
            for m in range(-l, l+1):
                rho_est += sph_coeff(atom_pos, r, l, m, alpha) * \
                    sph_harm(m, l, phi, theta)

    return np.real(rho_est)


def rho_exact(pos, atom_pos_list, alpha):
    rho_ex = 0
    for n in range(len(atom_pos_list)):
        atom_pos = atom_pos_list[n]
        rho_ex += exp(-alpha * np.linalg.norm(pos - atom_pos)**2)
    return rho_ex


def basis_overlap(n, nprime, nmax, rcut, sig):
    # integral computed in Mathematica
    # full simplified version recorded here
    prefactor = (1 / (8 * nmax**2)) * sig * \
        exp(-((n**2 + nprime**2)*rcut**2) / (2 * nmax**2 * sig**2))

    term1 = 2 * nmax * rcut * sig * \
        (n + nprime - exp(((n - nmax + nprime) * rcut**2) / (nmax * sig**2)) *
         (n + 2 * nmax + nprime))

    term2 = exp(((n + nprime)**2*rcut**2) / (4 * nmax**2 * sig**2)) * \
        np.sqrt(np.pi) * ((n + nprime)**2 * rcut**2 + 2 * nmax**2 * sig**2)

    term3 = erf(((n + nprime) * rcut) / (2 * nmax * sig)) - \
        erf(((n - 2 * nmax + nprime) * rcut) / (2 * nmax * sig))

    overlap = prefactor * (term1 + term2 * term3)

    return overlap

if __name__ == '__main__':
    # test estimate_rho against rho_exact
    pos = np.array([2, 1.25, 2])
    atom_pos_list = [np.array([1, 2, 3]),
                     np.array([2, 2, 2])]
    alpha = 0.4
    max_l = 14

    rho_est = estimate_rho(pos, atom_pos_list, alpha, max_l)
    print(rho_est)

    rho_ex = rho_exact(pos, atom_pos_list, alpha)
    print(rho_ex)

    # test basis_overlap
    test_overlap = basis_overlap(4, 6, 14, 5, 0.5)
    print(test_overlap)
