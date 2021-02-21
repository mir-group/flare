#ifndef B2_NORM_H
#define B2_NORM_H

#include "descriptor.h"
#include "b2.h"
#include <string>
#include <vector>

class Structure;

class B2_Norm : public B2 {
public:
  B2_Norm();

  B2_Norm(const std::string &radial_basis, const std::string &cutoff_function,
     const std::vector<double> &radial_hyps,
     const std::vector<double> &cutoff_hyps,
     const std::vector<int> &descriptor_settings);

  DescriptorValues compute_struc(Structure &structure);

  nlohmann::json return_json();
};

void compute_b2_norm(Eigen::MatrixXd &B2_vals, Eigen::MatrixXd &B2_force_dervs,
                     Eigen::VectorXd &B2_norms, Eigen::VectorXd &B2_force_dots,
                     const Eigen::MatrixXd &single_bond_vals,
                     const Eigen::MatrixXd &single_bond_force_dervs,
                     const Eigen::VectorXi &unique_neighbor_count,
                     const Eigen::VectorXi &cumulative_neighbor_count,
                     const Eigen::VectorXi &descriptor_indices, int nos, int N,
                     int lmax);

#endif
