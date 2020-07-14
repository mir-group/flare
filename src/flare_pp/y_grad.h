#ifndef Y_GRAD_H
#define Y_GRAD_H

#include <vector>

// Spherical harmonics.
void get_Y(std::vector<double> &Y, std::vector<double> &Yx,
           std::vector<double> &Yy, std::vector<double> &Yz, const double x,
           const double y, const double z, const int l);

#endif