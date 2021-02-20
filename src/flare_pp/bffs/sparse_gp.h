#ifndef SPARSE_GP_H
#define SPARSE_GP_H

#include "descriptor.h"
#include "kernel.h"
#include "structure.h"
#include <Eigen/Dense>
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

class SparseGP {
public:
  Eigen::VectorXd hyperparameters;

  // Kernel attributes.
  std::vector<Kernel *> kernels;
  std::vector<Eigen::MatrixXd> Kuu_kernels, Kuf_kernels;
  Eigen::MatrixXd Kuu, Kuf;
  int n_kernels = 0;
  double Kuu_jitter;

  // Solution attributes.
  Eigen::MatrixXd Sigma, Kuu_inverse, R_inv, L_inv;
  Eigen::VectorXd alpha, R_inv_diag, L_diag;

  // Training and sparse points.
  std::vector<ClusterDescriptor> sparse_descriptors;
  std::vector<Structure> training_structures;
  std::vector<std::vector<std::vector<int>>> sparse_indices;

  // Label attributes.
  Eigen::VectorXd noise_vector, y, label_count;
  int n_energy_labels = 0, n_force_labels = 0, n_stress_labels = 0,
      n_sparse = 0, n_labels = 0, n_strucs = 0;
  double energy_noise, force_noise, stress_noise;

  // Likelihood attributes.
  double log_marginal_likelihood, data_fit, complexity_penalty, trace_term,
      constant_term;
  Eigen::VectorXd likelihood_gradient;

  // Constructors.
  SparseGP();
  SparseGP(std::vector<Kernel *> kernels, double energy_noise,
           double force_noise, double stress_noise);

  void initialize_sparse_descriptors(const Structure &structure);
  void add_all_environments(const Structure &structure);

  void add_specific_environments(const Structure &structure,
                                 const std::vector<int> atoms);
  void add_random_environments(const Structure &structure,
                               const std::vector<int> &n_added);
  void add_uncertain_environments(const Structure &structure,
                                  const std::vector<int> &n_added);
  std::vector<Eigen::VectorXd>
  compute_cluster_uncertainties(const Structure &structure);
  std::vector<std::vector<int>>
  sort_clusters_by_uncertainty(const Structure &structure);

  void add_training_structure(const Structure &structure);
  void update_Kuu(const std::vector<ClusterDescriptor> &cluster_descriptors);
  void update_Kuf(const std::vector<ClusterDescriptor> &cluster_descriptors);
  void stack_Kuu();
  void stack_Kuf();

  void update_matrices_QR();

  void predict_SOR(Structure &structure);
  void predict_DTC(Structure &structure);
  void predict_local_uncertainties(Structure &structure);

  void compute_likelihood_stable();
  void compute_likelihood();

  double compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters);
  void set_hyperparameters(Eigen::VectorXd hyps);

  void write_mapping_coefficients(std::string file_name,
                                  std::string contributor,
                                  int kernel_index);

  Eigen::MatrixXd varmap_coeffs; // for debugging. TODO: remove this line 
  void write_varmap_coefficients(std::string file_name,
                                  std::string contributor,
                                  int kernel_index);

  // TODO: Make kernels jsonable.
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SparseGP, hyperparameters, kernels,    
    Kuu_kernels, Kuf_kernels, Kuu, Kuf, n_kernels, Kuu_jitter, Sigma,
    Kuu_inverse, R_inv, L_inv, alpha, R_inv_diag, L_diag, sparse_descriptors,
    training_structures, sparse_indices, noise_vector, y, label_count,
    n_energy_labels, n_force_labels, n_stress_labels, n_sparse, n_labels,
    n_strucs, energy_noise, force_noise, stress_noise, log_marginal_likelihood,
    data_fit, complexity_penalty, trace_term, constant_term,
    likelihood_gradient)

  static void to_json(std::string file_name, const SparseGP & sgp);
  static SparseGP from_json(std::string file_name);
};

#endif
