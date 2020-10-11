#ifndef POWER_SPECTRUM_H
#define POWER_SPECTRUM_H

#include <vector>
#include <string>
#include "compact_descriptor.h"

class CompactStructure;

class PowerSpectrum : public CompactDescriptor {
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

  PowerSpectrum();

  PowerSpectrum(const std::string &radial_basis,
                const std::string &cutoff_function,
                const std::vector<double> &radial_hyps,
                const std::vector<double> &cutoff_hyps,
                const std::vector<int> &descriptor_settings);

  DescriptorValues compute_struc(CompactStructure &structure);
};

void compute_power_spectrum(
    Eigen::MatrixXd &B2_vals, Eigen::MatrixXd &B2_force_dervs,
    Eigen::VectorXd &B2_norms, Eigen::VectorXd &B2_force_dots,
    const Eigen::MatrixXd &single_bond_vals,
    const Eigen::MatrixXd &single_bond_force_dervs,
    const Eigen::VectorXi &unique_neighbor_count,
    const Eigen::VectorXi &cumulative_neighbor_count,
    const Eigen::VectorXi &descriptor_indices, int nos, int N, int lmax);

void compute_single_bond(
    Eigen::MatrixXd &single_bond_vals, Eigen::MatrixXd &force_dervs,
    Eigen::MatrixXd &neighbor_coordinates,
    Eigen::VectorXi &neighbor_count, Eigen::VectorXi &cumulative_neighbor_count,
    Eigen::VectorXi &neighbor_indices,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        radial_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int nos, int N, int lmax, const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, const CompactStructure &structure);

#endif
