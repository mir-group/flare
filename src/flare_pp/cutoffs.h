#ifndef CUTOFFS_H
#define CUTOFFS_H

#include <vector>

// Radial cutoff functions.
void quadratic_cutoff(double *rcut_vals, double r, double rcut,
                      std::vector<double> cutoff_hyps);

void cos_cutoff(double *rcut_vals, double r, double rcut,
                std::vector<double> cutoff_hyps);

void hard_cutoff(double *rcut_vals, double r, double rcut,
                 std::vector<double> cutoff_hyps);

#endif