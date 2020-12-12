#ifndef CUTOFFS_H
#define CUTOFFS_H

#include <vector>
#include <string>
#include <functional>

// Radial cutoff functions.
void polynomial_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                       std::vector<double> cutoff_hyps);

void power_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                  std::vector<double> cutoff_hyps);

void quadratic_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                      std::vector<double> cutoff_hyps);

void cos_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                std::vector<double> cutoff_hyps);

void hard_cutoff(std::vector<double> &rcut_vals, double r, double rcut,
                 std::vector<double> cutoff_hyps);
    
void set_cutoff(const std::string &cutoff_function,
                std::function<void(std::vector<double> &, double, double,
                                   std::vector<double>)> &cutoff_pointer);

#endif