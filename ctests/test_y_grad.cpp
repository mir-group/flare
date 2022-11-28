#include "y_grad.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cmath>
#include <iostream>
using namespace std;

// Create a test fixture containing spherical harmonics up to l = 10 for an
// arbitrary vector (x, y, z).
class YGradTest : public ::testing::Test {
public:
  int l, sz;

  double x, y, z, delta, x_delt, y_delt, z_delt;

  vector<double> Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10;

  YGradTest() {

    //  Define an arbitrary vector (x, y, z).
    x = 2.16;
    y = 7.12;
    z = -3.14;

    // Define a slightly perturbed vector, which will be used to calculate the
    // gradient with finite difference.
    delta = 1e-8;
    x_delt = x + delta;
    y_delt = y + delta;
    z_delt = z + delta;

    // Set the size of the array containing the spherical harmonics.

    // Set the size of the array containing the spherical harmonics.
    l = 10;
    sz = (l + 1) * (l + 1);

    // Allocate arrays for Y and its derivatives.
    Y1 = vector<double>(sz, 0);
    Y2 = vector<double>(sz, 0);
    Y3 = vector<double>(sz, 0);
    Y4 = vector<double>(sz, 0);

    Y5 = vector<double>(sz, 0);
    Y6 = vector<double>(sz, 0);
    Y7 = vector<double>(sz, 0);
    Y8 = vector<double>(sz, 0);

    Y9 = vector<double>(sz, 0);
    Y10 = vector<double>(sz, 0);
  }
};

TEST(ComplexY, ComplexY) {
  Eigen::VectorXcd Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8;
  int l = 6;

  // Choose arbitrary rotation angles.
  double xrot = 1.28;
  double yrot = -3.21;
  double zrot = 0.42;

  // Define rotation matrices.
  Eigen::MatrixXd Rx{3, 3}, Ry{3, 3}, Rz{3, 3}, R{3, 3};
  Rx << 1, 0, 0, 0, cos(xrot), -sin(xrot), 0, sin(xrot), cos(xrot);
  Ry << cos(yrot), 0, sin(yrot), 0, 1, 0, -sin(yrot), 0, cos(yrot);
  Rz << cos(zrot), -sin(zrot), 0, sin(zrot), cos(zrot), 0, 0, 0, 1;
  R = Rx * Ry * Rz;

  // Define vectors.
  Eigen::VectorXd vec(3), rotated_vec(3);
  vec << 1.2, 3.4, -2.8;
  rotated_vec = R * vec;

  get_complex_Y(Y1, Y2, Y3, Y4, vec(0), vec(1), vec(2), l);
  get_complex_Y(Y5, Y6, Y7, Y8, rotated_vec(0), rotated_vec(1), rotated_vec(2),
                l);

  // Check addition theorem.
  std::complex<double> test, test2;
  double test_diff;
  double tolerance = 1e-8;
  int count = 0;
  for (int l_val = 0; l_val < l + 1; l_val++) {
    int m_no = 2 * l_val + 1;
    test = 0;
    test2 = 0;

    for (int m_val = count; m_val < count + m_no; m_val++) {
      test += Y1[m_val] * conj(Y1[m_val]);
      test2 += Y5[m_val] * conj(Y5[m_val]);
    }

    EXPECT_NEAR(imag(test), 0, tolerance);
    EXPECT_NEAR(imag(test2), 0, tolerance);
    EXPECT_NEAR(real(test), real(test2), tolerance);

    count += m_no;
  }

  // Check derivatives.
  double delta = 1e-8;
  tolerance = 1e-7;
  // x derivative
  get_complex_Y(Y5, Y6, Y7, Y8, vec(0) + delta, vec(1), vec(2), l);
  for (int test_val = 0; test_val < (l + 1) * (l + 1); test_val++) {
    std::complex<double> x_derv = Y2(test_val);
    std::complex<double> x_val_1 = Y1(test_val);
    std::complex<double> x_val_2 = Y5(test_val);
    std::complex<double> finite_diff = (x_val_2 - x_val_1) / delta;

    EXPECT_NEAR(real(x_derv), real(finite_diff), tolerance);
    EXPECT_NEAR(imag(x_derv), imag(finite_diff), tolerance);
  }

  // y derivative
  get_complex_Y(Y5, Y6, Y7, Y8, vec(0), vec(1) + delta, vec(2), l);
  for (int test_val = 0; test_val < (l + 1) * (l + 1); test_val++) {
    std::complex<double> x_derv = Y3(test_val);
    std::complex<double> x_val_1 = Y1(test_val);
    std::complex<double> x_val_2 = Y5(test_val);
    std::complex<double> finite_diff = (x_val_2 - x_val_1) / delta;

    EXPECT_NEAR(real(x_derv), real(finite_diff), tolerance);
    EXPECT_NEAR(imag(x_derv), imag(finite_diff), tolerance);
  }

  // z derivative
  get_complex_Y(Y5, Y6, Y7, Y8, vec(0), vec(1), vec(2) + delta, l);
  for (int test_val = 0; test_val < (l + 1) * (l + 1); test_val++) {
    std::complex<double> x_derv = Y4(test_val);
    std::complex<double> x_val_1 = Y1(test_val);
    std::complex<double> x_val_2 = Y5(test_val);
    std::complex<double> finite_diff = (x_val_2 - x_val_1) / delta;

    EXPECT_NEAR(real(x_derv), real(finite_diff), tolerance);
    EXPECT_NEAR(imag(x_derv), imag(finite_diff), tolerance);
  }
}

TEST_F(YGradTest, Grad) {
  // Check that the spherical harmonic gradients are correctly computed.

  // Get spherical harmonics and their gradients.
  get_Y(Y1, Y2, Y3, Y4, x, y, z, l);

  // Check x derivative.
  get_Y(Y5, Y6, Y7, Y8, x_delt, y, z, l);
  for (int test_val = 0; test_val < sz; test_val++) {
    double x_derv = Y2[test_val];
    double x_val_1 = Y1[test_val];
    double x_val_2 = Y5[test_val];
    double finite_diff = (x_val_2 - x_val_1) / delta;
    double diff_val = abs(finite_diff - x_derv);
    double tol = 1e-6;

    EXPECT_LE(diff_val, tol);
  }

  // Check y derivative.
  get_Y(Y5, Y6, Y7, Y8, x, y_delt, z, l);
  for (int test_val = 0; test_val < sz; test_val++) {
    double y_derv = Y3[test_val];
    double y_val_1 = Y1[test_val];
    double y_val_2 = Y5[test_val];
    double finite_diff = (y_val_2 - y_val_1) / delta;
    double diff_val = abs(finite_diff - y_derv);
    double tol = 1e-6;

    EXPECT_LE(diff_val, tol);
  }

  // Check z derivative.
  get_Y(Y5, Y6, Y7, Y8, x, y, z_delt, l);
  for (int test_val = 0; test_val < sz; test_val++) {
    double z_derv = Y4[test_val];
    double z_val_1 = Y1[test_val];
    double z_val_2 = Y5[test_val];
    double finite_diff = (z_val_2 - z_val_1) / delta;
    double diff_val = abs(finite_diff - z_derv);
    double tol = 1e-6;

    EXPECT_LE(diff_val, tol);
  }
}

TEST_F(YGradTest, AdditionTest) {
  // Check that the spherical harmonics satisfy the addition theorem, i.e. that
  // when the m's are summed over, the result is invariant to 3D rotations.

  // Define rotation matrices.
  double rot_x[9];
  double rot_y[9];
  double rot_z[9];

  double theta_x = 1.15346;
  double theta_y = -6.125;

  // Initialize vectors.
  double vec[3] = {x, y, z};
  double vec2[3] = {10.12, -5.23, 5.6};
  double vec_rot[3] = {};
  double vec2_rot[3] = {};
  double vec_rot2[3] = {};
  double vec2_rot2[3] = {};

  // Define rotation matrices.
  rot_x[0] = 1;
  rot_x[1] = 0;
  rot_x[2] = 0;
  rot_x[3] = 0;
  rot_x[4] = cos(theta_x);
  rot_x[5] = -sin(theta_x);
  rot_x[6] = 0;
  rot_x[7] = sin(theta_x);
  rot_x[8] = cos(theta_x);

  rot_y[0] = cos(theta_y);
  rot_y[1] = 0;
  rot_y[2] = sin(theta_y);
  rot_y[3] = 0;
  rot_y[4] = 1;
  rot_y[5] = 0;
  rot_y[6] = -sin(theta_y);
  rot_y[7] = 0;
  rot_y[8] = cos(theta_y);

  // Rotate the vector.
  // x rotation:
  int count = 0;
  for (int dim1 = 0; dim1 < 3; dim1++) {
    for (int dim2 = 0; dim2 < 3; dim2++) {
      vec_rot[dim1] += rot_x[count] * vec[dim2];
      vec2_rot[dim1] += rot_x[count] * vec2[dim2];
      count++;
    }
  }

  // y rotation:
  count = 0;
  for (int dim1 = 0; dim1 < 3; dim1++) {
    for (int dim2 = 0; dim2 < 3; dim2++) {
      vec_rot2[dim1] += rot_y[count] * vec_rot[dim2];
      vec2_rot2[dim1] += rot_y[count] * vec2_rot[dim2];
      count++;
    }
  }

  // Calculate spherical harmonics.
  get_Y(Y1, Y2, Y3, Y4, vec[0], vec[1], vec[2], l);
  get_Y(Y5, Y6, Y7, Y8, vec2[0], vec2[1], vec2[2], l);

  get_Y(Y9, Y2, Y3, Y4, vec_rot2[0], vec_rot2[1], vec_rot2[2], l);
  get_Y(Y10, Y6, Y7, Y8, vec2_rot2[0], vec2_rot2[1], vec2_rot2[2], l);

  // Check that the addition theorem is satisfied up to l = 10.
  double tolerance = 1e-6;
  count = 0;
  double test, test2, test_diff;

  for (int l_val = 0; l_val < l; l_val++) {
    int m_no = 2 * l_val + 1;
    test = 0;
    test2 = 0;

    for (int m_val = count; m_val < count + m_no; m_val++) {
      test += Y1[m_val] * Y5[m_val];
      test2 += Y9[m_val] * Y10[m_val];
    }

    test_diff = abs(test2 - test);
    EXPECT_LE(test_diff, tolerance);

    count += m_no;
  }
}
