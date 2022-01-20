#ifndef RADIAL_H
#define RADIAL_H

#include <functional>
#include <vector>
#include <string>

// Radial basis sets.
void fourier(std::vector<double> &basis_vals,
             std::vector<double> &basis_derivs, double r, int N,
             std::vector<double> radial_hyps);

void fourier_quarter(std::vector<double> &basis_vals,
                     std::vector<double> &basis_derivs, double r, int N,
                     std::vector<double> radial_hyps);

void fourier_half(std::vector<double> &basis_vals,
                  std::vector<double> &basis_derivs, double r, int N,
                  std::vector<double> radial_hyps);

void bessel(std::vector<double> &basis_vals, std::vector<double> &basis_derivs,
            double r, int N, std::vector<double> radial_hyps);

void equispaced_gaussians(std::vector<double> &basis_vals,
                          std::vector<double> &basis_derivs, double r, int N,
                          std::vector<double> radial_hyps);

void chebyshev(std::vector<double> &basis_vals,
               std::vector<double> &basis_derivs, double r, int N,
               std::vector<double> radial_hyps);

void positive_chebyshev(std::vector<double> &basis_vals,
                        std::vector<double> &basis_derivs, double r, int N,
                        std::vector<double> radial_hyps);

// The weighted Chebyshev radial basis set is based on Eqs. 21-24 of Drautz,
// Ralf. "Atomic cluster expansion for accurate and transferable interatomic
// potentials." Physical Review B 99.1 (2019): 014104. Atoms closer to the
// central atom are given exponentially more weight.
void weighted_chebyshev(std::vector<double> &basis_vals,
                        std::vector<double> &basis_derivs, double r, int N,
                        std::vector<double> radial_hyps);

void weighted_positive_chebyshev(std::vector<double> &basis_vals,
                                 std::vector<double> &basis_derivs, double r,
                                 int N, std::vector<double> radial_hyps);

void set_radial_basis(const std::string &basis_name,
                      std::function <void(std::vector<double> &,
                                          std::vector<double> &,
                                          double, int,
                                          std::vector<double>)>
                                          &radial_pointer);

void calculate_radial(
    std::vector<double> &comb_vals, std::vector<double> &comb_x,
    std::vector<double> &comb_y, std::vector<double> &comb_z,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    double x, double y, double z, double r, double rcut, int N,
    std::vector<double> radial_hyps, std::vector<double> cutoff_hyps);

#endif
