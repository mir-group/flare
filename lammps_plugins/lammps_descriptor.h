#ifndef LAMMPS_DESCRIPTOR_H
#define LAMMPS_DESCRIPTOR_H

#include <Eigen/Dense>
#include <functional>
#include <vector>


void single_bond(
    double **x, int *type, int jnum, int n_inner, int i, double xtmp,
    double ytmp, double ztmp, int *jlist,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    double cutoff, int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &single_bond_env_dervs);

void single_bond_multiple_cutoffs(
    double **x, int *type, int jnum, int n_inner, int i, double xtmp,
    double ytmp, double ztmp, int *jlist,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &single_bond_env_dervs,
    const Eigen::MatrixXd &cutoff_matrix);

void B1_descriptor(Eigen::VectorXd &B1_vals, Eigen::MatrixXd &B1_env_dervs,
                   double &norm_squared, Eigen::VectorXd &B1_env_dot,
                   const Eigen::VectorXd &single_bond_vals,
                   const Eigen::MatrixXd &single_bond_env_dervs, int n_species,
                   int N, int lmax); 

void B2_descriptor(Eigen::VectorXd &B2_vals,
                   double &norm_squared,
                   const Eigen::VectorXd &single_bond_vals,
                   int n_species,
                   int N, int lmax);

void compute_energy_and_u(Eigen::VectorXd &B2_vals, 
                   double &norm_squared,
                   const Eigen::VectorXd &single_bond_vals,
                   int power, int n_species,
                   int N, int lmax, const Eigen::MatrixXd &beta_matrix, 
                   Eigen::VectorXd &u, double *evdwl);

void compute_Bk(Eigen::VectorXd &Bk_vals, Eigen::MatrixXd &Bk_force_dervs,
                double &norm_squared, Eigen::VectorXd &Bk_force_dots,
                const Eigen::VectorXcd &single_bond_vals,
                const Eigen::MatrixXcd &single_bond_force_dervs,
                std::vector<std::vector<int>> nu, int nos, int K, int N,
                int lmax, const Eigen::VectorXd &coeffs,
                const Eigen::MatrixXd &beta_matrix, Eigen::VectorXcd &u, 
                double *evdwl);

void complex_single_bond(
    Eigen::MatrixXcd &single_bond_vals, Eigen::MatrixXcd &force_dervs,
    Eigen::MatrixXd &neighbor_coordinates, Eigen::VectorXi &neighbor_count,
    Eigen::VectorXi &cumulative_neighbor_count,
    Eigen::VectorXi &neighbor_indices,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        radial_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int nos, int N, int lmax, const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, const Structure &structure);

#endif
