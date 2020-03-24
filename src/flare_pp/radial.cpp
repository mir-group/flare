#include "radial.h"
#include <cmath>
#include <iostream>
#define Pi 3.14159265358979323846

void chebyshev(double * basis_vals, double * basis_derivs,
               double r, int N, std::vector<double> radial_hyps){

    double r1 = radial_hyps[0];
    double r2 = radial_hyps[1];

    // If r is ouside the support of the radial basis set, return.
    if ((r < r1) || (r > r2)){
        return;
    }

    double c = 1 / (r2 - r1);
    double x = (r - r1) * c;

    for (int n = 0; n < N; n ++){
        if (n == 0){
            basis_vals[n] = 1;
            basis_derivs[n] = 0;
        }
        else if (n == 1){
            basis_vals[n] = x;
            basis_derivs[n] = c;
        }
        // TODO: Check if using Chebyshev polynomials of the second kind improves the derivative.
        else{
            basis_vals[n] = 2 * x * basis_vals[n - 1] - basis_vals[n - 2];
            basis_derivs[n] = 2 * basis_vals[n - 1] * c +
                2 * x * basis_derivs[n - 1] - basis_derivs[n - 2];
        }
    }
}

void positive_chebyshev(double * basis_vals, double * basis_derivs,
                        double r, int N, std::vector<double> radial_hyps){

    double r1 = radial_hyps[0];
    double r2 = radial_hyps[1];

    // If r is ouside the support of the radial basis set, return.
    if ((r < r1) || (r > r2)){
        return;
    }

    std::vector<double> cheby_vals = std::vector<double>(N, 0);
    std::vector<double> cheby_derivs = std::vector<double>(N, 0);

    double c = 1 / (r2 - r1);
    double x = (r - r1) * c;
    double half = 1./2.;

    for (int n = 0; n < N; n ++){
        if (n == 0){
            cheby_vals[n] = 1;
            cheby_derivs[n] = 0;

            basis_vals[n] = 1;
            basis_derivs[n] = 0;
        }
        else if (n == 1){
            cheby_vals[n] = x;
            cheby_derivs[n] = c;

            basis_vals[n] = half * (1 - x);
            basis_derivs[n] = -half * c;
        }
        else{
            cheby_vals[n] = 2 * x * basis_vals[n - 1] - basis_vals[n - 2];
            cheby_derivs[n] = 2 * basis_vals[n - 1] * c +
                2 * x * basis_derivs[n - 1] - basis_derivs[n - 2];

            basis_vals[n] = half * (1 - cheby_vals[n]);
            basis_derivs[n] = -half * cheby_derivs[n];
        }
    }
}

void weighted_chebyshev(double * basis_vals, double * basis_derivs, double r,
    int N, std::vector<double> radial_hyps){

    double r1 = radial_hyps[0];
    double r2 = radial_hyps[1];
    double lambda = radial_hyps[2];

    // If r is ouside the support of the radial basis set, return.
    if ((r < r1) || (r > r2)){
        return;
    }

    double c = 1 / (r2 - r1);
    double x = (r - r1) * c;
    double exp_const = exp(-lambda * (x - 1));
    double lambda_const = exp(lambda) - 1;
    double x_weighted =
        1 - 2 * (exp_const - 1) / lambda_const;
    double dx_dr = 2 * c * lambda * exp_const / lambda_const;
    double half = 1./2.;

    std::vector<double> cheby_vals = std::vector<double>(N, 0);
    std::vector<double> cheby_derivs = std::vector<double>(N, 0);

    for (int n = 0; n < N; n ++){
        if (n == 0){
            cheby_vals[n] = 1;
            cheby_derivs[n] = 0;

            basis_vals[n] = 1;
            basis_derivs[n] = 0;
        }
        else if (n == 1){
            cheby_vals[n] = x_weighted;
            cheby_derivs[n] = 1;

            basis_vals[n] = cheby_vals[n];
            basis_derivs[n] = cheby_derivs[n] * dx_dr;
        }
        else{
            cheby_vals[n] = 2 * x_weighted * cheby_vals[n - 1] -
                cheby_vals[n - 2];
            cheby_derivs[n] = 2 * cheby_vals[n - 1] +
                2 * x_weighted * cheby_derivs[n - 1] - cheby_derivs[n - 2];

            basis_vals[n] = cheby_vals[n];
            basis_derivs[n] = cheby_derivs[n] * dx_dr;
        }
    }
}

void weighted_positive_chebyshev(
    double * basis_vals, double * basis_derivs, double r, int N,
    std::vector<double> radial_hyps){

    double r1 = radial_hyps[0];
    double r2 = radial_hyps[1];
    double lambda = radial_hyps[2];

    // If r is ouside the support of the radial basis set, return.
    if ((r < r1) || (r > r2)){
        return;
    }

    double c = 1 / (r2 - r1);
    double x = (r - r1) * c;
    double exp_const = exp(-lambda * (x - 1));
    double lambda_const = exp(lambda) - 1;
    double x_weighted =
        1 - 2 * (exp_const - 1) / lambda_const;
    double dx_dr = 2 * c * lambda * exp_const / lambda_const;
    double half = 1./2.;

    std::vector<double> cheby_vals = std::vector<double>(N, 0);
    std::vector<double> cheby_derivs = std::vector<double>(N, 0);

    for (int n = 0; n < N; n ++){
        if (n == 0){
            cheby_vals[n] = 1;
            cheby_derivs[n] = 0;

            basis_vals[n] = 1;
            basis_derivs[n] = 0;
        }
        else if (n == 1){
            cheby_vals[n] = x_weighted;
            cheby_derivs[n] = 1;

            basis_vals[n] = half * (1 - x_weighted);
            basis_derivs[n] = -half * dx_dr;
        }
        else{
            cheby_vals[n] = 2 * x_weighted * cheby_vals[n - 1] -
                cheby_vals[n - 2];
            cheby_derivs[n] = 2 * cheby_vals[n - 1] +
                2 * x_weighted * cheby_derivs[n - 1] - cheby_derivs[n - 2];

            basis_vals[n] = half * (1 - cheby_vals[n]);
            basis_derivs[n] = -half * cheby_derivs[n] * dx_dr;
        }
    }
}

void equispaced_gaussians(double * basis_vals, double * basis_derivs,
                          double r, int N, std::vector<double> radial_hyps){

    // Define Gaussian hyperparameters (width and locations of first and final gaussians)
    double sigma = radial_hyps[0];
    double first_gauss = radial_hyps[1];
    double final_gauss = radial_hyps[2];
    double gauss_sep = final_gauss - first_gauss;

    // Calculate equispaced Gaussians and their gradients.
    double norm_factor = 1 / (sigma * sqrt(2 * Pi));
    double sig2 = 1 / (sigma * sigma);
    double half_sig2 = sig2 / 2;

    double mean_val;
    double exp_arg;
    double mean_diff;
    double gn_val;
    double gn_derv;

    for (int n = 0; n < N; n++){
        mean_val = first_gauss + (n * gauss_sep / (N - 1));
        mean_diff = r - mean_val;
        exp_arg = -half_sig2 * mean_diff * mean_diff;
        gn_val = norm_factor * exp(exp_arg);
        basis_vals[n] = gn_val;
        basis_derivs[n] = -sig2 * gn_val * mean_diff;
    }
}

void calculate_radial(
    double * comb_vals, double * comb_x, double * comb_y, double * comb_z,
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>),
    void (*cutoff_function)(double *, double, double, std::vector<double>),
    double x, double y, double z, double r, double rcut, int N,
    std::vector<double> radial_hyps, std::vector<double> cutoff_hyps){

    // Calculate cutoff values.
    double rcut_vals[2];
    (*cutoff_function)(rcut_vals, r, rcut, cutoff_hyps);

    // Calculate radial basis values.
    double * basis_vals = new double[N]();
    double * basis_derivs = new double[N]();
    (*basis_function)(basis_vals, basis_derivs, r, N, radial_hyps);

    // Store the product.
    double xrel = x / r;
    double yrel = y / r;
    double zrel = z / r;

    for (int n = 0; n < N; n++){
        comb_vals[n] = basis_vals[n] * rcut_vals[0];
        comb_x[n] = basis_derivs[n] * xrel * rcut_vals[0] + basis_vals[n] * 
                    xrel * rcut_vals[1];
        comb_y[n] = basis_derivs[n] * yrel * rcut_vals[0] + basis_vals[n] * 
                    yrel * rcut_vals[1];
        comb_z[n] = basis_derivs[n] * zrel * rcut_vals[0] + basis_vals[n] * 
                    zrel * rcut_vals[1];
    }

    delete[] basis_vals;
    delete[] basis_derivs;

}
