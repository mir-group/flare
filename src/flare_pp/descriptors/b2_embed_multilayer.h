#ifndef B2_EMBED_MULTILAYER_H
#define B2_EMBED_MULTILAYER_H

#include "descriptor.h"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

class Structure;

class B2_Embed_Multilayer : public Descriptor {
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

  std::string descriptor_name = "B2_Embed_Multilayer";

  /** Matrix of cutoff values, with element (i, j) corresponding to the cutoff
   * assigned to the species pair (i, j), where i is the central species
   * and j is the environment species.
   */
  Eigen::MatrixXd cutoffs;

  Eigen::MatrixXd embed_coeffs;

  B2_Embed_Multilayer();

  B2_Embed_Multilayer(const std::string &radial_basis, const std::string &cutoff_function,
     const std::vector<double> &radial_hyps,
     const std::vector<double> &cutoff_hyps,
     const std::vector<int> &descriptor_settings, 
     const Eigen::MatrixXd &embed_coefficients);

  /**
   * Construct the B2 descriptor with distinct cutoffs for each pair of species.
   */
  B2_Embed_Multilayer(const std::string &radial_basis, const std::string &cutoff_function,
     const std::vector<double> &radial_hyps,
     const std::vector<double> &cutoff_hyps,
     const std::vector<int> &descriptor_settings,
     const Eigen::MatrixXd &cutoffs, 
     const Eigen::MatrixXd &embed_coefficients);

  DescriptorValues compute_struc(Structure &structure);
  DescriptorValues compute_single_bonds(Structure &structure);

  void write_to_file(std::ofstream &coeff_file, int coeff_size);

  nlohmann::json return_json();
};

void compute_b2_embed_multilayer(Eigen::MatrixXd &B2_vals, Eigen::MatrixXd &B2_force_dervs,
                Eigen::VectorXd &B2_norms, Eigen::VectorXd &B2_force_dots,
                const Eigen::MatrixXd &single_bond_vals,
                const Eigen::MatrixXd &single_bond_force_dervs,
                const Eigen::VectorXi &unique_neighbor_count,
                const Eigen::VectorXi &cumulative_neighbor_count,
                const Eigen::VectorXi &descriptor_indices, int nos, int N,
                int lmax);

#endif