#ifndef LAMMPS_DESCRIPTOR_H
#define LAMMPS_DESCRIPTOR_H

#include <functional>
#include <Eigen/Dense>
#include <vector>

void single_bond(double **x, int atom_index, int *type, int inum,
    int *ilist, int *numneigh, int **firstneigh,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
        int, std::vector<double>)> basis_function,
    std::function<void(std::vector<double> &, double, double,
        std::vector<double>)> cutoff_function,
    double cutoff, int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps,
    Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &environment_force_dervs,
    Eigen::MatrixXd &central_force_dervs);

#endif