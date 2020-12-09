#ifndef CUTOFFS_H
#define CUTOFFS_H

#include <vector>

// Radial cutoff functions.
void polynomial_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                       std::vector<double> cutoff_hyps);

void quadratic_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                      std::vector<double> cutoff_hyps);

void cos_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                std::vector<double> cutoff_hyps);

void hard_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                 std::vector<double> cutoff_hyps);

#endif