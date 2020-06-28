#ifndef RADIAL_H
#define RADIAL_H

#include <vector>

// Radial basis sets.
void equispaced_gaussians(double * basis_vals, double * basis_derivs,
                          double r, int N, std::vector<double> radial_hyps);

void chebyshev(double * basis_vals, double * basis_derivs,
               double r, int N, std::vector<double> radial_hyps);

void positive_chebyshev(double * basis_vals, double * basis_derivs,
                        double r, int N, std::vector<double> radial_hyps);

// The weighted Chebyshev radial basis set is based on Eqs. 21-24 of Drautz, Ralf. "Atomic cluster expansion for accurate and transferable interatomic potentials." Physical Review B 99.1 (2019): 014104. Atoms closer to the central atom are given exponentially more weight.
void weighted_chebyshev(double * basis_vals, double * basis_derivs, double r,
    int N, std::vector<double> radial_hyps);

void weighted_positive_chebyshev(
    double * basis_vals, double * basis_derivs, double r, int N,
    std::vector<double> radial_hyps);

void calculate_radial(
    double * comb_vals, double * comb_x, double * comb_y, double * comb_z,
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>),
    void (*cutoff_function)(double *, double, double, std::vector<double>),
    double x, double y, double z, double r, double rcut, int N,
    std::vector<double> radial_hyps, std::vector<double> cutoff_hyps);

#endif