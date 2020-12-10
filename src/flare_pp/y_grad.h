#ifndef Y_GRAD_H
#define Y_GRAD_H

#include <Eigen/Dense>
#include <vector>

// Spherical harmonics.
void get_Y(std::vector<double> &Y, std::vector<double> &Yx,
           std::vector<double> &Yy, std::vector<double> &Yz, const double x,
           const double y, const double z, const int l);

void get_complex_Y(Eigen::VectorXcd &Y, Eigen::VectorXcd &Yx,
                   Eigen::VectorXcd &Yy, Eigen::VectorXcd &Yz, const double x,
                   const double y, const double z, const int l);

#endif