#include <cmath>
#include "cutoffs.h"
#define Pi 3.14159265358979323846

void equispaced_gaussians(double * basis_vals, double * basis_derivs,
                          double r, int N, double * radial_hyps){

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
