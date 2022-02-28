#include "cutoffs.h"
#include <cmath>
#include <iostream>
#define Pi 3.14159265358979323846

// This polynomial cutoff was introduced in Klicpera et al. arXiv:2003.03123.
void polynomial_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                       std::vector<double> cutoff_hyps) {

  if (r > rcut) {
    rcut_vals[0] = 0;
    rcut_vals[1] = 0;
    return;
  }

  int p = cutoff_hyps[0];
  double d = r / rcut;
  double c1 = (p + 1) * (p + 2) / 2;
  double c2 = p * (p + 2);
  double c3 = p * (p + 1) / 2;
  rcut_vals[0] = 1 - c1 * pow(d, p) + c2 * pow(d, p + 1) - c3 * pow(d, p + 2);

  double c4 = c1 * p / rcut;
  double c5 = c2 * (p + 1) / rcut;
  double c6 = c3 * (p + 2) / rcut;
  rcut_vals[1] = -c4 * pow(d, p - 1) + c5 * pow(d, p) - c6 * pow(d, p + 1);
}

void power_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                  std::vector<double> cutoff_hyps) {

  if (r > rcut) {
    rcut_vals[0] = 0;
    rcut_vals[1] = 0;
    return;
  }

  double pow_val = cutoff_hyps[0];
  double rdiff = rcut - r;
  rcut_vals[0] = pow(rdiff, pow_val);
  rcut_vals[1] = -pow_val * pow(rdiff, pow_val - 1);
}

void quadratic_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                      std::vector<double> cutoff_hyps) {

  if (r > rcut) {
    rcut_vals[0] = 0;
    rcut_vals[1] = 0;
    return;
  }

  double rdiff = r - rcut;
  rcut_vals[0] = rdiff * rdiff;
  rcut_vals[1] = 2 * rdiff;
}

void cos_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                std::vector<double> cutoff_hyps) {

  // Calculate the cosine cutoff function and its gradient.
  if (r > rcut) {
    rcut_vals[0] = 0;
    rcut_vals[1] = 0;
    return;
  }

  double cutoff_val = (1. / 2.) * (cos(Pi * r / rcut) + 1);
  double cutoff_derv = -Pi * sin(Pi * r / rcut) / (2 * rcut);

  rcut_vals[0] = cutoff_val;
  rcut_vals[1] = cutoff_derv;
}

void hard_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                 std::vector<double> cutoff_hyps) {
  if (r > rcut) {
    rcut_vals[0] = 0;
    rcut_vals[1] = 0;
    return;
  }

  rcut_vals[0] = 1;
  rcut_vals[1] = 0;
}

void set_cutoff(const std::string &cutoff_function,
                std::function<void(std::vector<double> &, double, double,
                                   std::vector<double>)> &cutoff_pointer){

  // Set the cutoff function.
  if (cutoff_function == "quadratic") {
    cutoff_pointer = quadratic_cutoff;
  } else if (cutoff_function == "hard") {
    cutoff_pointer = hard_cutoff;
  } else if (cutoff_function == "cosine") {
    cutoff_pointer = cos_cutoff;
  } else if (cutoff_function == "polynomial") {
    cutoff_pointer = polynomial_cutoff;
  } else if (cutoff_function == "power") {
    cutoff_pointer = power_cutoff;
  }
}
