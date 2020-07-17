#ifndef SPARSE_GP_H
#define SPARSE_GP_H

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <vector>

#include "kernels.h"
#include "local_environment.h"
#include "structure.h"

class SparseGP {
public:
  Eigen::MatrixXd Kuu, Kuf_env, Kuf_struc, beta;
  Eigen::VectorXd y_struc, y_env, alpha, hyperparameters, noise_env,
      noise_struc;
  std::vector<Kernel *> kernels;

  double model_size, sparse_size, training_size, Kuu_size, Kuf_env_size,
      noise_env_size;

  std::vector<LocalEnvironment> sparse_environments, training_environments;
  std::vector<StructureDescriptor> training_structures;
  std::vector<int> label_count;

  double energy_norm, forces_norm, stresses_norm, energy_offset, forces_offset,
      stresses_offset;

  double sigma_e, sigma_f, sigma_s, Kuu_jitter;

  SparseGP();

  SparseGP(std::vector<Kernel *> kernels, double sigma_e, double sigma_f,
           double sigma_s);

  void add_sparse_environment(const LocalEnvironment &env);
  void add_sparse_environments(const std::vector<LocalEnvironment> &envs);
  void add_training_environment(const LocalEnvironment &training_environment);
  void add_training_environments(const std::vector<LocalEnvironment> &envs);
  void add_training_structure(const StructureDescriptor &training_structure);

  void three_body_grid(double min_dist, double max_dist, double cutoff,
                       int n_species, int n_dist, int n_angle);

  void update_alpha();      // find alpha with naive matrix inversion
  void update_alpha_LLT();  // find alpha with Cholesky decomposition
  void update_alpha_LDLT(); // Cholesky minus square root
  void update_alpha_CG();   // find alpha with conjugate gradient methods

  void compute_beta(int kernel_index, int descriptor_index);

  void write_beta(std::string file_name, std::string contributor,
                  int descriptor_index);

  void memory_profile();

  void compute_likelihood();

  Eigen::VectorXd predict(const StructureDescriptor &test_structure);

  Eigen::VectorXd predict_force(const LocalEnvironment &test_environment);

  double predict_local_energy(const LocalEnvironment &test_environment);

  void clear_environment_lists();

  // void predict_DTC(StructureDescriptor test_structure,
  //     Eigen::VectorXd & mean_vector, Eigen::VectorXd & std_vector);

  // void predict_SOR(StructureDescriptor test_structure,
  //     Eigen::VectorXd & mean_vector, Eigen::VectorXd & std_vector);
};

#endif