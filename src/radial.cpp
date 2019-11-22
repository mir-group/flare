#include <cmath>
#include "cutoffs.h"
#define Pi 3.14159265358979323846

// void cos_cutoff(double * rcut_vals, double r, double rcut){
//     // Calculate the cosine cutoff function and its gradient. Assumes that the input is an array of four zeros. If r > rcut, the array is returned unchanged.

//     if (r > rcut){
//         return;
//     }

//     double cutoff_val = (1./2.) * (cos(Pi * r / rcut) + 1);
//     double cutoff_derv = -Pi * sin(Pi * r / rcut) / (2 * rcut);

//     rcut_vals[0] = cutoff_val;
//     rcut_vals[1] = cutoff_derv;
// }

void get_gns(double * g, double * gx, double * gy, double * gz,
             double x, double y, double z, double r, double sigma,
             double rcut, int N){

    // Calculate cutoff function and its gradient.
    double rcut_vals[2];
    cos_cutoff(rcut_vals, r, rcut);

    // Calculate equispaced Gaussians and their gradients.
    double norm_factor = 1 / (sigma * sqrt(2 * Pi));
    double sig2 = 1 / (sigma * sigma);
    double half_sig2 = sig2 / 2;

    double mean_val;
    double exp_arg;
    double mean_diff;
    double gn_val;
    double gn_derv;

    double xrel = x/r;
    double yrel = y/r;
    double zrel = z/r;

    for (int n = 0; n < N; n++){
        mean_val = n * rcut / (N - 1);
        mean_diff = r - mean_val;
        exp_arg = -half_sig2 * mean_diff * mean_diff;
        gn_val = norm_factor * exp(exp_arg);
        gn_derv = -sig2 * gn_val * mean_diff;

        g[n] = gn_val * rcut_vals[0];
        gx[n] = gn_derv * xrel * rcut_vals[0] + gn_val * xrel * rcut_vals[1];
        gy[n] = gn_derv * yrel * rcut_vals[0] + gn_val * yrel * rcut_vals[1];
        gz[n] = gn_derv * zrel * rcut_vals[0] + gn_val * zrel * rcut_vals[1];
    }
}
