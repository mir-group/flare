#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>

using namespace std;

// Create test inputs for radial functions.
class RadialTest : public ::testing::Test{
    protected:
    double x = 2.19;
    double y = 1.23;
    double z = -0.24;
    double r = sqrt(x * x + y * y + z * z);
    double rcut = 7;

    double delta = 1e-8;
    double x_delt = x + delta;
    double y_delt = y + delta;
    double z_delt = z + delta;
    double r_delt = r + delta;
    double r_x = sqrt(x_delt * x_delt + y * y + z * z);
    double r_y = sqrt(x * x + y_delt * y_delt + z * z);
    double r_z = sqrt(x * x + y * y + z_delt * z_delt);
};

TEST_F(RadialTest, LongR){
    // Test that the cutoff value and its gradient are zero when r > rcut.
    double rcut = 0.001;
    double cutoff_vals[2] = {};
    double * cutoff_hyps;
    cos_cutoff(cutoff_vals, r, rcut, cutoff_hyps);

    EXPECT_EQ(cutoff_vals[0], 0);
    EXPECT_EQ(cutoff_vals[1], 0);
}

TEST_F(RadialTest, CutoffGrad){
    // Test that the derivative of the cosine cutoff function is correctly computed when r < rcut.
    double cutoff_vals[2] = {};
    double cutoff_vals_rdelt[2] = {};
    double * cutoff_hyps;

    cos_cutoff(cutoff_vals, r, rcut, cutoff_hyps);
    cos_cutoff(cutoff_vals_rdelt, r_delt, rcut, cutoff_hyps);

    double r_fin_diff = (cutoff_vals_rdelt[0] - cutoff_vals[0]) / delta;
    double r_diff = abs(r_fin_diff - cutoff_vals[1]);

    double tolerance = 1e-6;
    EXPECT_LE(r_diff, tolerance);
}

TEST_F(RadialTest, HardCutoff){
    // Test that the hard cutoff returns 1 and 0 when rcut > r.
    double cutoff_vals[2] = {};
    double * cutoff_hyps;
    hard_cutoff(cutoff_vals, r, rcut, cutoff_hyps);
    EXPECT_EQ(cutoff_vals[0], 1);
    EXPECT_EQ(cutoff_vals[1], 0);

    // Test that it returns 0 and 0 when r > rcut.
    double rcut = 0.01;
    hard_cutoff(cutoff_vals, r, rcut, cutoff_hyps);
    EXPECT_EQ(cutoff_vals[0], 0);
    EXPECT_EQ(cutoff_vals[1], 0);

}

TEST_F(RadialTest, GnDerv){
    // Test that the Gaussian gradients are correctly computed.
    double sigma = 1;
    double first_gauss = 1;
    double final_gauss = 6;
    int N = 10;
    double hyps[3] = {sigma, first_gauss, final_gauss};

    double * g = new double[N];
    double * gderv = new double[N];
    double * g_rdelt = new double[N];
    double * gderv_rdelt = new double[N];

    equispaced_gaussians(g, gderv, r, N, hyps);
    equispaced_gaussians(g_rdelt, gderv_rdelt, r_delt, N, hyps);

    double r_finite_diff, r_diff;
    double tolerance = 1e-6;

    for (int n = 0; n < N; n++){
        // Check r derivative
        r_finite_diff = (g_rdelt[n] - g[n]) / delta;
        r_diff = abs(r_finite_diff - gderv[n]);
        EXPECT_LE(r_diff, tolerance);
    }

    delete[] g; delete[] gderv; delete[] g_rdelt; delete[] gderv_rdelt;
}

TEST_F(RadialTest, CombDerv){
    // Test that the combined gradients are correctly computed.
    double sigma = 1;
    double first_gauss = 1;
    double final_gauss = 6;
    int N = 10;
    double radial_hyps[3] = {sigma, first_gauss, final_gauss};
    double * cutoff_hyps;

    // Initialize arrays.
    double * g = new double[N];
    double * gx = new double[N];
    double * gy = new double[N];
    double * gz = new double[N];

    double * g_xdelt = new double[N];
    double * g_ydelt = new double[N];
    double * g_zdelt = new double[N];

    double * gx_delt = new double[N];
    double * gy_delt = new double[N];
    double * gz_delt = new double[N];

    // Set the basis and cutoff function.
    void (*basis_function)(double *, double *, double, int, double *) =
        equispaced_gaussians;
    void (*cutoff_function)(double *, double, double, double *) = cos_cutoff;

    calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, 
                     x, y, z, r, rcut, N, radial_hyps, cutoff_hyps);
    calculate_radial(g_xdelt, gx_delt, gy_delt, gz_delt,
                     basis_function, cutoff_function, 
                     x_delt, y, z, r_x, rcut, N,
                     radial_hyps, cutoff_hyps);
    calculate_radial(g_ydelt, gx_delt, gy_delt, gz_delt,
                     basis_function, cutoff_function, 
                     x, y_delt, z, r_y, rcut, N,
                     radial_hyps, cutoff_hyps);
    calculate_radial(g_zdelt, gx_delt, gy_delt, gz_delt,
                     basis_function, cutoff_function,
                     x, y, z_delt, r_z, rcut, N,
                     radial_hyps, cutoff_hyps);

    double x_finite_diff, y_finite_diff, z_finite_diff, x_diff, y_diff, z_diff;
    double tolerance = 1e-6;
    for (int n = 0; n < N; n++){
        // Check x derivative
        x_finite_diff = (g_xdelt[n] - g[n]) / delta;
        x_diff = abs(x_finite_diff - gx[n]);
        EXPECT_LE(x_diff, tolerance);

        // Check y derivative
        y_finite_diff = (g_ydelt[n] - g[n]) / delta;
        y_diff = abs(y_finite_diff - gy[n]);
        EXPECT_LE(y_diff, tolerance);

        // Check z derivative
        z_finite_diff = (g_zdelt[n] - g[n]) / delta;
        z_diff = abs(z_finite_diff - gz[n]);
        EXPECT_LE(z_diff, tolerance);
    }

    delete[] g; delete[] gx; delete[] gy; delete[] gz;
    delete[] g_xdelt; delete[] g_ydelt; delete[] g_zdelt;
    delete[] gx_delt; delete[] gy_delt; delete[] gz_delt;
}
