#ifndef B2_SIMPLE_H
#define B2_SIMPLE_H

#include "descriptor.h"
#include <string>
#include <vector>

class Structure;

class B2_Simple : public Descriptor {
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

  B2_Simple();

  B2_Simple(const std::string &radial_basis,
     const std::string &cutoff_function,
     const std::vector<double> &radial_hyps,
     const std::vector<double> &cutoff_hyps,
     const std::vector<int> &descriptor_settings);

  DescriptorValues compute_struc(Structure &structure);

  void write_to_file(std::ofstream &coeff_file, int coeff_size);

  nlohmann::json return_json();
};

void compute_b2_simple(Eigen::MatrixXd &B2_vals,
                Eigen::MatrixXd &B2_force_dervs,
                Eigen::VectorXd &B2_norms, Eigen::VectorXd &B2_force_dots,
                const Eigen::MatrixXd &single_bond_vals,
                const Eigen::MatrixXd &single_bond_force_dervs,
                const Eigen::VectorXi &unique_neighbor_count,
                const Eigen::VectorXi &cumulative_neighbor_count,
                const Eigen::VectorXi &descriptor_indices, int nos, int N,
                int lmax);

#endif
