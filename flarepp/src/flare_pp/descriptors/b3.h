#ifndef B3_H
#define B3_H

#include "descriptor.h"
#include <string>
#include <vector>

class Structure;

class B3 : public Descriptor {
public:
  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      radial_pointer;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_pointer;
  std::string radial_basis, cutoff_function;
  std::vector<double> radial_hyps, cutoff_hyps;
  std::vector<int> descriptor_settings;
  Eigen::VectorXd wigner3j_coeffs;

  std::string descriptor_name = "B3";

  B3();

  B3(const std::string &radial_basis, const std::string &cutoff_function,
     const std::vector<double> &radial_hyps,
     const std::vector<double> &cutoff_hyps,
     const std::vector<int> &descriptor_settings);

  DescriptorValues compute_struc(Structure &structure);

  nlohmann::json return_json();
};

void compute_B3(Eigen::MatrixXd &B3_vals, Eigen::MatrixXd &B3_force_dervs,
                Eigen::VectorXd &B3_norms, Eigen::VectorXd &B3_force_dots,
                const Eigen::MatrixXcd &single_bond_vals,
                const Eigen::MatrixXcd &single_bond_force_dervs,
                const Eigen::VectorXi &unique_neighbor_count,
                const Eigen::VectorXi &cumulative_neighbor_count,
                const Eigen::VectorXi &descriptor_indices, int nos, int N,
                int lmax, const Eigen::VectorXd &wigner3j_coeffs);

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
