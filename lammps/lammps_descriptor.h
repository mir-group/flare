#ifndef LAMMPS_DESCRIPTOR_H
#define LAMMPS_DESCRIPTOR_H

#include <functional>
#include <Eigen/Dense>
#include <vector>

void single_bond(double **x, int *type, int jnum, int i, double xtmp,
    double ytmp, double ztmp, int *jlist,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
        int, std::vector<double>)> basis_function,
    std::function<void(std::vector<double> &, double, double,
        std::vector<double>)> cutoff_function,
    double cutoff, int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps,
    Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &single_bond_env_dervs);

void B2_descriptor(Eigen::VectorXd &B2_vals, Eigen::MatrixXd &B2_env_dervs,
    double &norm_squared, Eigen::VectorXd &B2_env_dot,
    const Eigen::VectorXd &single_bond_vals,
    const Eigen::MatrixXd &single_bond_env_dervs, int n_species, int N,
    int lmax);

#endif