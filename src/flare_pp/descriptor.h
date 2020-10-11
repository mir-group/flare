#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <Eigen/Dense>
#include <vector>

class LocalEnvironment;
class CompactStructure;

// Descriptor calculator.
class DescriptorCalculator {
public:
  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      radial_pointer;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_pointer;
  Eigen::VectorXd single_bond_vals, descriptor_vals;
  Eigen::MatrixXd single_bond_force_dervs, single_bond_stress_dervs,
      descriptor_force_dervs, descriptor_stress_dervs;
  std::string radial_basis, cutoff_function;
  std::vector<double> radial_hyps, cutoff_hyps;
  std::vector<int> descriptor_settings;
  int descriptor_index;

  DescriptorCalculator();

  DescriptorCalculator(const std::string &radial_basis,
                       const std::string &cutoff_function,
                       const std::vector<double> &radial_hyps,
                       const std::vector<double> &cutoff_hyps,
                       const std::vector<int> &descriptor_settings,
                       int descriptor_index);

  virtual void compute(const LocalEnvironment &env) = 0;
  virtual void compute_struc(CompactStructure &structure) = 0;

  void destroy_matrices();

  virtual ~DescriptorCalculator() = default;
};

void B2_descriptor(Eigen::VectorXd &B2_vals, Eigen::MatrixXd &B2_force_dervs,
                   Eigen::MatrixXd &B2_stress_dervs,
                   const Eigen::VectorXd &single_bond_vals,
                   const Eigen::MatrixXd &single_bond_force_dervs,
                   const Eigen::MatrixXd &single_bond_stress_dervs,
                   const LocalEnvironment &env, int nos, int N, int lmax);

void B2_descriptor_struc(
    Eigen::MatrixXd &B2_vals, Eigen::MatrixXd &B2_force_dervs,
    Eigen::VectorXd &B2_norms, Eigen::VectorXd &B2_force_dots,
    const Eigen::MatrixXd &single_bond_vals,
    const Eigen::MatrixXd &single_bond_force_dervs,
    const Eigen::VectorXi &unique_neighbor_count,
    const Eigen::VectorXi &cumulative_neighbor_count,
    const Eigen::VectorXi &descriptor_indices, int nos, int N, int lmax);

class B1_Calculator : public DescriptorCalculator {
public:
  B1_Calculator();

  B1_Calculator(const std::string &radial_basis,
                const std::string &cutoff_function,
                const std::vector<double> &radial_hyps,
                const std::vector<double> &cutoff_hyps,
                const std::vector<int> &descriptor_settings,
                int descriptor_index);

  void compute(const LocalEnvironment &env);
  void compute_struc(CompactStructure &structure);
};

class B2_Calculator : public DescriptorCalculator {
public:
  B2_Calculator();

  B2_Calculator(const std::string &radial_basis,
                const std::string &cutoff_function,
                const std::vector<double> &radial_hyps,
                const std::vector<double> &cutoff_hyps,
                const std::vector<int> &descriptor_settings,
                int descriptor_index);

  void compute(const LocalEnvironment &env);
  void compute_struc(CompactStructure &structure);
};

#endif