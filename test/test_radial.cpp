#include "gtest/gtest.h"
#include "radial.h"
#include <iostream>
#include <cmath>

using namespace std;

// Create test inputs for radial functions.
class RadialTest : public ::testing::Test{
    protected:
    double x = 2.1;
    double y = 1.23;
    double z = -0.24;
    double r = sqrt(x * x + y * y + z * z);
    double rcut = 7;

    double delta = 1e-8;
    double x_delt = x + delta;
    double y_delt = y + delta;
    double z_delt = z + delta;
    double r_x = sqrt(x_delt * x_delt + y * y + z * z);
    double r_y = sqrt(x * x + y_delt * y_delt + z * z);
    double r_z = sqrt(x * x + y * y + z_delt * z_delt);
};

TEST_F(RadialTest, LongR){
    // Test that the cutoff value and its gradient are zero when r > rcut.
    double rcut = 0.001;
    double cutoff_vals[4] = {};
    cos_cutoff(cutoff_vals, x, y, z, r, rcut);

    EXPECT_EQ(cutoff_vals[0], 0);
    EXPECT_EQ(cutoff_vals[1], 0);
    EXPECT_EQ(cutoff_vals[2], 0);
    EXPECT_EQ(cutoff_vals[3], 0);

}

TEST_F(RadialTest, CutoffGrad){
    // Test that the gradient of the cosine cutoff function is correctly computed when r < rcut.
    double cutoff_vals[4] = {};
    double cutoff_vals_xdelt[4] = {};
    double cutoff_vals_ydelt[4] = {};
    double cutoff_vals_zdelt[4] = {};

    cos_cutoff(cutoff_vals, x, y, z, r, rcut);
    cos_cutoff(cutoff_vals_xdelt, x_delt, y, z, r_x, rcut);
    cos_cutoff(cutoff_vals_ydelt, x, y_delt, z, r_y, rcut);
    cos_cutoff(cutoff_vals_zdelt, x, y, z_delt, r_z, rcut);

    double x_fin_diff = (cutoff_vals_xdelt[0] - cutoff_vals[0]) / delta;
    double y_fin_diff = (cutoff_vals_ydelt[0] - cutoff_vals[0]) / delta;
    double z_fin_diff = (cutoff_vals_zdelt[0] - cutoff_vals[0]) / delta;

    double x_diff = abs(x_fin_diff - cutoff_vals[1]);
    double y_diff = abs(y_fin_diff - cutoff_vals[2]);
    double z_diff = abs(z_fin_diff - cutoff_vals[3]);

    double tolerance = 1e-6;
    EXPECT_LE(x_diff, tolerance);
    EXPECT_LE(y_diff, tolerance);
    EXPECT_LE(z_diff, tolerance);
}

TEST_F(RadialTest, GnDerv){
    // Test that the Gaussian gradients are correctly computed.
    double sigma = 1;
    int N = 10;

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

    get_gns(g, gx, gy, gz, x, y, z, r, sigma, rcut, N);
    get_gns(g_xdelt, gx_delt, gy_delt, gz_delt, x_delt, y, z, r_x, sigma,
            rcut, N);
    get_gns(g_ydelt, gx_delt, gy_delt, gz_delt, x, y_delt, z, r_y, sigma,
            rcut, N);
    get_gns(g_zdelt, gx_delt, gy_delt, gz_delt, x, y, z_delt, r_z, sigma,
            rcut, N);

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