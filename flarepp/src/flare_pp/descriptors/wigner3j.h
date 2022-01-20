#ifndef WIGNER3J
#define WIGNER3J
#include <Eigen/Dense>

// Wigner 3j coefficients generated for l = 0, 1, 2, 3 using
// sympy.physics.wigner.wigner_3j

Eigen::VectorXd compute_coeffs(int lmax);

#endif
