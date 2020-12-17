#include "cutoffs.h"
#include "radial.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

using namespace std;

// Create test inputs for radial functions.
class RadialTest : public ::testing::Test {
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

TEST_F(RadialTest, LongR) {
  // Test that the cutoff value and its gradient are zero when r > rcut.
  double rcut = 0.001;
  std::vector<double> cutoff_vals(2, 0);
  std::vector<double> cutoff_hyps;
  cos_cutoff(cutoff_vals, r, rcut, cutoff_hyps);

  EXPECT_EQ(cutoff_vals[0], 0);
  EXPECT_EQ(cutoff_vals[1], 0);
}

TEST_F(RadialTest, CutoffGrad) {
  // Test that the derivative of the cosine cutoff function is correctly
  // computed when r < rcut.
  std::vector<double> cutoff_vals(2, 0), cutoff_vals_rdelt(2, 0);
  std::vector<double> cutoff_hyps;

  cos_cutoff(cutoff_vals, r, rcut, cutoff_hyps);
  cos_cutoff(cutoff_vals_rdelt, r_delt, rcut, cutoff_hyps);

  double r_fin_diff = (cutoff_vals_rdelt[0] - cutoff_vals[0]) / delta;
  double r_diff = abs(r_fin_diff - cutoff_vals[1]);

  double tolerance = 1e-6;
  EXPECT_LE(r_diff, tolerance);
}

TEST_F(RadialTest, QuadGrad) {
  std::vector<double> cutoff_vals(2, 0), cutoff_vals_rdelt(2, 0);
  std::vector<double> cutoff_hyps;

  quadratic_cutoff(cutoff_vals, r, rcut, cutoff_hyps);
  quadratic_cutoff(cutoff_vals_rdelt, r_delt, rcut, cutoff_hyps);

  double r_fin_diff = (cutoff_vals_rdelt[0] - cutoff_vals[0]) / delta;
  double r_diff = abs(r_fin_diff - cutoff_vals[1]);

  double tolerance = 1e-6;
  EXPECT_LE(r_diff, tolerance);
}

TEST_F(RadialTest, HardCutoff) {
  // Test that the hard cutoff returns 1 and 0 when rcut > r.
  std::vector<double> cutoff_vals(2, 0);
  std::vector<double> cutoff_hyps;
  hard_cutoff(cutoff_vals, r, rcut, cutoff_hyps);
  EXPECT_EQ(cutoff_vals[0], 1);
  EXPECT_EQ(cutoff_vals[1], 0);

  // Test that it returns 0 and 0 when r > rcut.
  double rcut = 0.01;
  hard_cutoff(cutoff_vals, r, rcut, cutoff_hyps);
  EXPECT_EQ(cutoff_vals[0], 0);
  EXPECT_EQ(cutoff_vals[1], 0);
}

TEST(ChebyTest, ChebySupport) {
  double r1 = 2;
  double r2 = 4;
  double r = 1;
  int N = 10;
  std::vector<double> hyps = {r1, r2};

  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gderv = std::vector<double>(N, 0);

  chebyshev(g, gderv, r, N, hyps);
  for (int n = 0; n < N; n++) {
    EXPECT_EQ(g[n], 0);
  }

  r = 10;
  chebyshev(g, gderv, r, N, hyps);
  for (int n = 0; n < N; n++) {
    EXPECT_EQ(g[n], 0);
  }
}

TEST(FourierDerv, FourierDerv) {
  double r1 = 0;
  double r2 = 4;
  double r = 2.5;
  int N = 10;
  double delta = 1e-8;
  double rdelt = r + delta;
  double rdelt_2 = r - delta;
  double r_finite_diff, r_diff;
  double tolerance = 1e-4;
  std::vector<double> hyps = {r1, r2};

  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gderv = std::vector<double>(N, 0);
  std::vector<double> g_rdelt = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt = std::vector<double>(N, 0);
  std::vector<double> g_rdelt_2 = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt_2 = std::vector<double>(N, 0);

  fourier_quarter(g, gderv, r, N, hyps);
  fourier_quarter(g_rdelt, gderv_rdelt, rdelt, N, hyps);
  fourier_quarter(g_rdelt_2, gderv_rdelt_2, rdelt_2, N, hyps);

  for (int n = 0; n < N; n++) {
    r_finite_diff = (g_rdelt[n] - g_rdelt_2[n]) / (2 * delta);
    r_diff = abs(r_finite_diff - gderv[n]);
    EXPECT_LE(r_diff, tolerance);
  }
}

TEST(BesselDerv, BesselDerv) {
  double r1 = 0;
  double r2 = 4;
  double r = 2.5;
  int N = 10;
  double delta = 1e-8;
  double rdelt = r + delta;
  double rdelt_2 = r - delta;
  double r_finite_diff, r_diff;
  double tolerance = 1e-4;
  std::vector<double> hyps = {r1, r2};

  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gderv = std::vector<double>(N, 0);
  std::vector<double> g_rdelt = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt = std::vector<double>(N, 0);
  std::vector<double> g_rdelt_2 = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt_2 = std::vector<double>(N, 0);

  bessel(g, gderv, r, N, hyps);
  bessel(g_rdelt, gderv_rdelt, rdelt, N, hyps);
  bessel(g_rdelt_2, gderv_rdelt_2, rdelt_2, N, hyps);

  for (int n = 0; n < N; n++) {
    r_finite_diff = (g_rdelt[n] - g_rdelt_2[n]) / (2 * delta);
    r_diff = abs(r_finite_diff - gderv[n]);
    EXPECT_LE(r_diff, tolerance);
  }
}

TEST(ChebyTest, ChebyDerv) {
  double r1 = 2;
  double r2 = 4;
  double r = 2.5;
  int N = 10;
  double delta = 1e-8;
  double rdelt = r + delta;
  double rdelt_2 = r - delta;
  double r_finite_diff, r_diff;
  double tolerance = 1e-4;
  std::vector<double> hyps = {r1, r2};

  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gderv = std::vector<double>(N, 0);
  std::vector<double> g_rdelt = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt = std::vector<double>(N, 0);
  std::vector<double> g_rdelt_2 = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt_2 = std::vector<double>(N, 0);

  chebyshev(g, gderv, r, N, hyps);
  chebyshev(g_rdelt, gderv_rdelt, rdelt, N, hyps);
  chebyshev(g_rdelt_2, gderv_rdelt_2, rdelt_2, N, hyps);

  for (int n = 0; n < N; n++) {
    r_finite_diff = (g_rdelt[n] - g_rdelt_2[n]) / (2 * delta);

    r_diff = abs(r_finite_diff - gderv[n]);
    EXPECT_LE(r_diff, tolerance);
  }
}

TEST(ChebyTest, PositiveChebyDerv) {
  double r1 = 2;
  double r2 = 4;
  double r = 2.5;
  int N = 10;
  double delta = 1e-8;
  double rdelt = r + delta;
  double rdelt_2 = r - delta;
  double r_finite_diff, r_diff;
  double tolerance = 1e-4;
  std::vector<double> hyps = {r1, r2};

  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gderv = std::vector<double>(N, 0);
  std::vector<double> g_rdelt = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt = std::vector<double>(N, 0);
  std::vector<double> g_rdelt_2 = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt_2 = std::vector<double>(N, 0);

  positive_chebyshev(g, gderv, r, N, hyps);
  positive_chebyshev(g_rdelt, gderv_rdelt, rdelt, N, hyps);
  positive_chebyshev(g_rdelt_2, gderv_rdelt_2, rdelt_2, N, hyps);

  for (int n = 0; n < N; n++) {
    r_finite_diff = (g_rdelt[n] - g_rdelt_2[n]) / (2 * delta);
    r_diff = abs(r_finite_diff - gderv[n]);
    EXPECT_LE(r_diff, tolerance);
  }
}

TEST(ChebyTest, WeightedChebyDerv) {
  double r1 = 2;
  double r2 = 4;
  double r = 2.5;
  int N = 10;
  double delta = 1e-8;
  double rdelt = r + delta;
  double rdelt_2 = r - delta;
  double lambda = 5;
  double r_finite_diff, r_diff;
  double tolerance = 1e-4;
  std::vector<double> hyps = {r1, r2, lambda};

  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gderv = std::vector<double>(N, 0);
  std::vector<double> g_rdelt = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt = std::vector<double>(N, 0);
  std::vector<double> g_rdelt_2 = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt_2 = std::vector<double>(N, 0);

  weighted_chebyshev(g, gderv, r, N, hyps);
  weighted_chebyshev(g_rdelt, gderv_rdelt, rdelt, N, hyps);
  weighted_chebyshev(g_rdelt_2, gderv_rdelt_2, rdelt_2, N, hyps);

  for (int n = 0; n < N; n++) {
    r_finite_diff = (g_rdelt[n] - g_rdelt_2[n]) / (2 * delta);
    r_diff = abs(r_finite_diff - gderv[n]);
    EXPECT_LE(r_diff, tolerance);
  }
}

TEST_F(RadialTest, GnDerv) {
  // Test that the Gaussian gradients are correctly computed.
  double sigma = 1;
  double first_gauss = 1;
  double final_gauss = 6;
  int N = 10;
  std::vector<double> hyps = {sigma, first_gauss, final_gauss};

  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gderv = std::vector<double>(N, 0);
  std::vector<double> g_rdelt = std::vector<double>(N, 0);
  std::vector<double> gderv_rdelt = std::vector<double>(N, 0);

  equispaced_gaussians(g, gderv, r, N, hyps);
  equispaced_gaussians(g_rdelt, gderv_rdelt, r_delt, N, hyps);

  double r_finite_diff, r_diff;
  double tolerance = 1e-6;

  for (int n = 0; n < N; n++) {
    // Check r derivative
    r_finite_diff = (g_rdelt[n] - g[n]) / delta;
    r_diff = abs(r_finite_diff - gderv[n]);
    EXPECT_LE(r_diff, tolerance);
  }
}

TEST_F(RadialTest, CombDerv) {
  // Test that the combined gradients are correctly computed.
  double sigma = 1;
  double first_gauss = 1;
  double final_gauss = 6;
  int N = 10;
  double p = 6;
  std::vector<double> radial_hyps = {sigma, first_gauss, final_gauss};
  std::vector<double> cutoff_hyps = {p};

  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gx = std::vector<double>(N, 0);
  std::vector<double> gy = std::vector<double>(N, 0);
  std::vector<double> gz = std::vector<double>(N, 0);
  std::vector<double> g_xdelt = std::vector<double>(N, 0);
  std::vector<double> g_ydelt = std::vector<double>(N, 0);
  std::vector<double> g_zdelt = std::vector<double>(N, 0);
  std::vector<double> gx_delt = std::vector<double>(N, 0);
  std::vector<double> gy_delt = std::vector<double>(N, 0);
  std::vector<double> gz_delt = std::vector<double>(N, 0);

  // Set the basis and cutoff function.
  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      basis_function = equispaced_gaussians;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function = polynomial_cutoff;

  calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, x, y, z, r,
                   rcut, N, radial_hyps, cutoff_hyps);
  calculate_radial(g_xdelt, gx_delt, gy_delt, gz_delt, basis_function,
                   cutoff_function, x_delt, y, z, r_x, rcut, N, radial_hyps,
                   cutoff_hyps);
  calculate_radial(g_ydelt, gx_delt, gy_delt, gz_delt, basis_function,
                   cutoff_function, x, y_delt, z, r_y, rcut, N, radial_hyps,
                   cutoff_hyps);
  calculate_radial(g_zdelt, gx_delt, gy_delt, gz_delt, basis_function,
                   cutoff_function, x, y, z_delt, r_z, rcut, N, radial_hyps,
                   cutoff_hyps);

  double x_finite_diff, y_finite_diff, z_finite_diff, x_diff, y_diff, z_diff;
  double tolerance = 1e-6;
  for (int n = 0; n < N; n++) {
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
}
