#include "Sparse_GP/sparse_gp.h"
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

static const double Pi = 3.14159265358979323846;

SparseGP ::SparseGP() {}

SparseGP ::SparseGP(std::vector<Kernel *> kernels, double sigma_e,
                    double sigma_f, double sigma_s) {

  this->kernels = kernels;
  Kuu_jitter = 1e-8; // default value

  // Count hyperparameters.
  int n_hyps = 0;
  for (int i = 0; i < kernels.size(); i++) {
    n_hyps += kernels[i]->kernel_hyperparameters.size();
  }

  // Set the kernel hyperparameters.
  hyperparameters = Eigen::VectorXd::Zero(n_hyps + 3);
  std::vector<double> hyps_curr;
  int hyp_counter = 0;
  for (int i = 0; i < kernels.size(); i++) {
    hyps_curr = kernels[i]->kernel_hyperparameters;

    for (int j = 0; j < hyps_curr.size(); j++) {
      hyperparameters(hyp_counter) = hyps_curr[j];
      hyp_counter++;
    }
  }

  // Set the noise hyperparameters.
  hyperparameters(n_hyps) = sigma_e;
  hyperparameters(n_hyps + 1) = sigma_f;
  hyperparameters(n_hyps + 2) = sigma_s;

  this->sigma_e = sigma_e;
  this->sigma_f = sigma_f;
  this->sigma_s = sigma_s;
}

void SparseGP ::memory_profile() {
  // Record sizes in bytes
  model_size = sizeof(*this);
  sparse_size = sizeof(sparse_environments);
  training_size = sizeof(training_environments);
  Kuu_size = sizeof(Kuu);
  Kuf_env_size = sizeof(Kuf_env);
  noise_env_size = sizeof(noise_env);
}

void SparseGP ::add_sparse_environment(const LocalEnvironment &env) {
  // Compute kernels between new environment and previous sparse
  // environments.
  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();

  Eigen::VectorXd uu_vector = Eigen::VectorXd::Zero(n_sparse + 1);
  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < n_sparse; i++) {
    for (int j = 0; j < n_kernels; j++) {
      uu_vector(i) += kernels[j]->env_env(sparse_environments[i], env);
    }
  }

  // Compute self kernel.
  double self_kernel = 0;
  for (int j = 0; j < n_kernels; j++) {
    self_kernel += kernels[j]->env_env(env, env);
  }

  // Update Kuu matrix.
  Kuu.conservativeResize(Kuu.rows() + 1, Kuu.cols() + 1);
  Kuu.col(Kuu.cols() - 1) = uu_vector;
  Kuu.row(Kuu.rows() - 1) = uu_vector;
  Kuu(n_sparse, n_sparse) = self_kernel;

  // Compute kernels between new environment and training structures.
  int n_labels = Kuf_struc.cols();
  int n_strucs = training_structures.size();
  Eigen::VectorXd uf_vector = Eigen::VectorXd::Zero(n_labels);

  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < n_strucs; i++) {
    int initial_index, index;

    // Get initial index.
    if (i == 0) {
      initial_index = 0;
    } else {
      initial_index = label_count[i - 1];
    }
    index = initial_index;

    int n_atoms = training_structures[i].nat;
    Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
    for (int j = 0; j < n_kernels; j++) {
      kernel_vector += kernels[j]->env_struc(env, training_structures[i]);
    }

    if (training_structures[i].energy.size() != 0) {
      uf_vector(index) = kernel_vector(0);
      index += 1;
    }

    if (training_structures[i].forces.size() != 0) {
      uf_vector.segment(index, n_atoms * 3) =
          kernel_vector.segment(1, n_atoms * 3);
      index += n_atoms * 3;
    }

    if (training_structures[i].stresses.size() != 0) {
      uf_vector.segment(index, 6) = kernel_vector.tail(6);
    }
  }

  // Update Kuf_struc matrix.
  Kuf_struc.conservativeResize(Kuf_struc.rows() + 1, Kuf_struc.cols());
  Kuf_struc.row(Kuf_struc.rows() - 1) = uf_vector;

  // Compute kernels between new environment and training environments.
  n_labels = Kuf_env.cols();
  int n_envs = training_environments.size();
  uf_vector = Eigen::VectorXd::Zero(n_labels);

  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < n_envs; i++) {
    for (int j = 0; j < n_kernels; j++) {
      uf_vector.segment(3 * i, 3) +=
          kernels[j]->env_env_force(env, training_environments[i]);
    }
  }

  // Update Kuf_env matrix.
  Kuf_env.conservativeResize(Kuf_env.rows() + 1, Kuf_env.cols());
  Kuf_env.row(Kuf_env.rows() - 1) = uf_vector;

  // Store sparse environment.
  sparse_environments.push_back(env);
}

void SparseGP ::add_sparse_environments(
    const std::vector<LocalEnvironment> &envs) {

  // Compute kernels between new environment and previous sparse
  // environments.
  int n_envs = envs.size();
  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();

  Eigen::MatrixXd prev_block = Eigen::MatrixXd::Zero(n_sparse, n_envs);
#pragma omp parallel for schedule(static)
  for (int k = 0; k < n_envs; k++) {
    for (int i = 0; i < n_sparse; i++) {
      for (int j = 0; j < n_kernels; j++) {
        prev_block(i, k) +=
            kernels[j]->env_env(sparse_environments[i], envs[k]);
      }
    }
  }

  // Compute self block. (Note that the work can be cut in half by exploiting
  // the symmetry of the matrix, but this makes parallelization slightly
  // trickier.)
  Eigen::MatrixXd self_block = Eigen::MatrixXd::Zero(n_envs, n_envs);
#pragma omp parallel for schedule(static)
  for (int k = 0; k < n_envs; k++) {
    for (int l = 0; l < n_envs; l++) {
      for (int j = 0; j < n_kernels; j++) {
        self_block(k, l) += kernels[j]->env_env(envs[k], envs[l]);
      }
    }
  }

  // Update Kuu matrix.
  Kuu.conservativeResize(n_sparse + n_envs, n_sparse + n_envs);
  Kuu.block(0, n_sparse, n_sparse, n_envs) = prev_block;
  Kuu.block(n_sparse, 0, n_envs, n_sparse) = prev_block.transpose();
  Kuu.block(n_sparse, n_sparse, n_envs, n_envs) = self_block;

  // TODO: compute kernels between new environments and training structures.

  // Update Kuf_struc matrix.
  Kuf_struc.conservativeResize(Kuf_struc.rows() + n_envs, Kuf_struc.cols());

  // Compute kernels between new sparse environments and training environments.
  int n_labels = Kuf_env.cols();
  int n_training_envs = training_environments.size();

  Eigen::MatrixXd uf_block = Eigen::MatrixXd::Zero(n_envs, n_labels);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_envs; i++) {
    for (int j = 0; j < n_training_envs; j++) {
      for (int k = 0; k < n_kernels; k++) {
        uf_block.row(i).segment(3 * j, 3) +=
            kernels[k]->env_env_force(envs[i], training_environments[j]);
      }
    }
  }

  // Update Kuf_env matrix.
  Kuf_env.conservativeResize(n_sparse + n_envs, n_labels);
  Kuf_env.block(n_sparse, 0, n_envs, n_labels) = uf_block;

  // Store sparse environments.
  for (int i = 0; i < n_envs; i++) {
    sparse_environments.push_back(envs[i]);
  }
}

void SparseGP ::add_training_structure(
    const StructureDescriptor &training_structure) {

  int n_labels = training_structure.energy.size() +
                 training_structure.forces.size() +
                 training_structure.stresses.size();
  int n_atoms = training_structure.nat;
  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();

  // Update label counts.
  int prev_count;
  int curr_size = label_count.size();
  if (label_count.size() == 0) {
    label_count.push_back(n_labels);
  } else {
    prev_count = label_count[curr_size - 1];
    label_count.push_back(n_labels + prev_count);
  }

  // Calculate kernels between sparse environments and training structure.
  Eigen::MatrixXd kernel_block = Eigen::MatrixXd::Zero(n_sparse, n_labels);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_sparse; i++) {
    Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
    for (int j = 0; j < n_kernels; j++) {
      kernel_vector +=
          kernels[j]->env_struc(sparse_environments[i], training_structure);
    }

    // Update kernel block.
    int count = 0;
    if (training_structure.energy.size() != 0) {
      kernel_block(i, 0) = kernel_vector(0);
      count += 1;
    }

    if (training_structure.forces.size() != 0) {
      kernel_block.row(i).segment(count, n_atoms * 3) =
          kernel_vector.segment(1, n_atoms * 3);
    }

    if (training_structure.stresses.size() != 0) {
      kernel_block.row(i).tail(6) = kernel_vector.tail(6);
    }
  }

  // Add kernel block to Kuf_struc.
  int prev_cols = Kuf_struc.cols();
  Kuf_struc.conservativeResize(n_sparse, prev_cols + n_labels);
  Kuf_struc.block(0, prev_cols, n_sparse, n_labels) = kernel_block;

  // Store training structure.
  training_structures.push_back(training_structure);

  // Update y vector and noise matrix.
  Eigen::VectorXd labels = Eigen::VectorXd::Zero(n_labels);
  Eigen::VectorXd noise_vector = Eigen::VectorXd::Zero(n_labels);

  int count = 0;
  if (training_structure.energy.size() != 0) {
    labels.head(1) = training_structure.energy;
    noise_vector(0) = 1 / (sigma_e * sigma_e);
    count++;
  }

  if (training_structure.forces.size() != 0) {
    labels.segment(count, n_atoms * 3) = training_structure.forces;
    noise_vector.segment(count, n_atoms * 3) =
        Eigen::VectorXd::Constant(n_atoms * 3, 1 / (sigma_f * sigma_f));
  }

  if (training_structure.stresses.size() != 0) {
    labels.tail(6) = training_structure.stresses;
    noise_vector.tail(6) =
        Eigen::VectorXd::Constant(6, 1 / (sigma_s * sigma_s));
  }

  y_struc.conservativeResize(y_struc.size() + n_labels);
  y_struc.tail(n_labels) = labels;

  noise_struc.conservativeResize(prev_cols + n_labels);
  noise_struc.tail(n_labels) = noise_vector;
}

void SparseGP ::add_training_environment(
    const LocalEnvironment &training_environment) {

  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();
  int n_labels = 3; // Assumes that all three force components are given.

  // Calculate kernels between sparse environments and training environment.
  Eigen::MatrixXd kernel_block = Eigen::MatrixXd::Zero(n_sparse, n_labels);

  // Note: this doesn't appear to give an efficient speed-up.
  // (Probably not enough action in the for loop.)
  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < n_sparse; i++) {
    for (int j = 0; j < n_kernels; j++) {
      kernel_block.row(i) += kernels[j]->env_env_force(sparse_environments[i],
                                                       training_environment);
    }
  }

  // Add kernel block to Kuf_env.
  int prev_cols = Kuf_env.cols();
  Kuf_env.conservativeResize(n_sparse, prev_cols + n_labels);
  Kuf_env.block(0, prev_cols, n_sparse, n_labels) = kernel_block;

  // Store training structure.
  training_environments.push_back(training_environment);

  // Update y vector and noise matrix.
  Eigen::VectorXd labels = training_environment.force;
  Eigen::VectorXd noise_vector =
      Eigen::VectorXd::Constant(3, 1 / (sigma_f * sigma_f));

  y_env.conservativeResize(y_env.size() + n_labels);
  y_env.tail(n_labels) = labels;

  noise_env.conservativeResize(prev_cols + n_labels);
  noise_env.tail(n_labels) = noise_vector;
}

void SparseGP ::add_training_environments(
    const std::vector<LocalEnvironment> &envs) {

  int n_sparse = sparse_environments.size();
  int n_envs = envs.size();
  int n_kernels = kernels.size();
  int n_labels = 3; // Assumes that all three force components are given.

  // Calculate kernels between sparse environments and training environments.
  Eigen::MatrixXd kernel_block =
      Eigen::MatrixXd::Zero(n_sparse, n_labels * n_envs);

#pragma omp parallel for schedule(static)
  for (int k = 0; k < n_envs; k++) {
    for (int i = 0; i < n_sparse; i++) {
      for (int j = 0; j < n_kernels; j++) {
        kernel_block.row(i).segment(k * n_labels, 3) +=
            kernels[j]->env_env_force(sparse_environments[i], envs[k]);
      }
    }
  }

  // Add kernel block to Kuf_env.
  int prev_cols = Kuf_env.cols();
  Kuf_env.conservativeResize(n_sparse, prev_cols + n_labels * n_envs);
  Kuf_env.block(0, prev_cols, n_sparse, n_labels * n_envs) = kernel_block;

  // Store training environments and forces.
  Eigen::VectorXd labels = Eigen::VectorXd::Zero(n_labels * n_envs);
  for (int i = 0; i < n_envs; i++) {
    training_environments.push_back(envs[i]);
    labels.segment(i * n_labels, n_labels) = envs[i].force;
  }

  // Update y vector.
  y_env.conservativeResize(y_env.size() + n_labels * n_envs);
  y_env.tail(n_labels * n_envs) = labels;

  // Update noise matrix.
  Eigen::VectorXd noise_vector =
      Eigen::VectorXd::Constant(n_labels * n_envs, 1 / (sigma_f * sigma_f));

  noise_env.conservativeResize(prev_cols + n_labels * n_envs);
  noise_env.tail(n_labels * n_envs) = noise_vector;
}

// TODO: decouple triplets of different types.
void SparseGP ::three_body_grid(double min_dist, double max_dist, double cutoff,
                                int n_species, int n_dist, int n_angle) {

  double dist1, dist2, dist3, angle;
  double dist_step = (max_dist - min_dist) / (n_dist - 1);
  double angle_step = Pi / (n_angle - 1);

  // Create cell. Both max_dist and cutoff should be no greater than half the
  // cell size to avoid periodic images.
  double max_val;
  if (max_dist > cutoff) {
    max_val = max_dist;
  } else {
    max_val = cutoff;
  }

  double struc_factor = 10;
  Eigen::MatrixXd cell{3, 3};
  cell << max_val * struc_factor, 0, 0, 0, max_val * struc_factor, 0, 0, 0,
      max_val * struc_factor;
  std::vector<int> species;
  Eigen::MatrixXd positions = Eigen::MatrixXd::Zero(3, 3);
  std::vector<double> n_body_cutoffs{cutoff, cutoff};
  StructureDescriptor struc;
  LocalEnvironment env;
  int counter = 0;

  // Loop over species.
  for (int s1 = 0; s1 < n_species; s1++) {
    for (int s2 = s1; s2 < n_species; s2++) {
      for (int s3 = s2; s3 < n_species; s3++) {
        species = {s1, s2, s3};

        // Loop over distances.
        for (int d1 = 0; d1 < n_dist; d1++) {
          dist1 = min_dist + d1 * dist_step;

          for (int d2 = 0; d2 < n_dist; d2++) {
            dist2 = min_dist + d2 * dist_step;

            if ((s2 == s3) && (dist2 < dist1)) {
              continue;
            } else {
              // Loop over angles ranging from 0 to Pi.
              for (int a1 = 0; a1 < n_angle; a1++) {
                angle = a1 * angle_step;
                dist3 = sqrt(dist1 * dist1 + dist2 * dist2 -
                             2 * dist1 * dist2 * cos(angle));

                // Skip over triplets for which dist3 exceeds the cutoff.
                if (dist3 > cutoff) {
                  continue;
                }
                // If all three species are the same, neglect the case where
                // dist3 < dist2.
                else if ((s1 == s2) && (s2 == s3) && (dist3 < dist2)) {
                  continue;
                } else {
                  // Create structure of 3 atoms.
                  positions(1, 0) = dist1;
                  positions(2, 0) = dist2 * cos(angle);
                  positions(2, 1) = dist2 * sin(angle);
                  struc = StructureDescriptor(cell, species, positions, cutoff,
                                              n_body_cutoffs);

                  // Create environment.
                  env = struc.local_environments[0];

                  // Add to the training set
                  this->add_sparse_environment(env);

                  counter++;
                }
              }
            }
          }
        }
      }
    }
  }
}

void SparseGP::update_alpha() {
  // Combine Kuf_struc and Kuf_env.
  int n_sparse = Kuf_struc.rows();
  int n_struc_labels = Kuf_struc.cols();
  int n_env_labels = Kuf_env.cols();
  Eigen::MatrixXd Kuf =
      Eigen::MatrixXd::Zero(n_sparse, n_struc_labels + n_env_labels);
  Kuf.block(0, 0, n_sparse, n_struc_labels) = Kuf_struc;
  Kuf.block(0, n_struc_labels, n_sparse, n_env_labels) = Kuf_env;

  // Combine noise_struc and noise_env.
  Eigen::VectorXd noise = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  noise.segment(0, n_struc_labels) = noise_struc;
  noise.segment(n_struc_labels, n_env_labels) = noise_env;

  // Combine training labels.
  Eigen::VectorXd y = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  y.segment(0, n_struc_labels) = y_struc;
  y.segment(n_struc_labels, n_env_labels) = y_env;

  // Set alpha.
  Eigen::MatrixXd sigma_inv =
      Kuu + Kuf * noise.asDiagonal() * Kuf.transpose() +
      Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());

  alpha = sigma_inv.inverse() * Kuf * noise.asDiagonal() * y;
}

void SparseGP::update_alpha_LLT() {
  // Combine Kuf_struc and Kuf_env.
  int n_sparse = Kuf_struc.rows();
  int n_struc_labels = Kuf_struc.cols();
  int n_env_labels = Kuf_env.cols();
  Eigen::MatrixXd Kuf =
      Eigen::MatrixXd::Zero(n_sparse, n_struc_labels + n_env_labels);
  Kuf.block(0, 0, n_sparse, n_struc_labels) = Kuf_struc;
  Kuf.block(0, n_struc_labels, n_sparse, n_env_labels) = Kuf_env;

  // Combine noise_struc and noise_env.
  Eigen::VectorXd noise = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  noise.segment(0, n_struc_labels) = noise_struc;
  noise.segment(n_struc_labels, n_env_labels) = noise_env;

  // Combine training labels.
  Eigen::VectorXd y = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  y.segment(0, n_struc_labels) = y_struc;
  y.segment(n_struc_labels, n_env_labels) = y_env;

  // Solve for alpha with Cholesky decomposition.
  Eigen::MatrixXd sigma_inv =
      Kuu + Kuf * noise.asDiagonal() * Kuf.transpose() +
      Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
  Eigen::VectorXd b = Kuf * noise.asDiagonal() * y;

  Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(sigma_inv);
  alpha = llt.solve(b);
}

void SparseGP::update_alpha_LDLT() {
  // Solve for alpha with inplace Cholesky decomposition.
  // Experiment with constant noise to check the effect on memory.
  double noise_val = 1 / (sigma_f * sigma_f);

  Eigen::MatrixXd sigma_inv = Kuu;
  sigma_inv.noalias() += noise_val * Kuf_env * Kuf_env.transpose();
  sigma_inv.noalias() += noise_val * Kuf_struc * Kuf_struc.transpose();
  sigma_inv.noalias() +=
      Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());

  Eigen::VectorXd b =
      noise_val * Kuf_env * y_env + noise_val * Kuf_struc * y_struc;

  Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt(sigma_inv);
  alpha = ldlt.solve(b);
}

void SparseGP::update_alpha_CG() {
  // Combine Kuf_struc and Kuf_env.
  int n_sparse = Kuf_struc.rows();
  int n_struc_labels = Kuf_struc.cols();
  int n_env_labels = Kuf_env.cols();
  Eigen::MatrixXd Kuf =
      Eigen::MatrixXd::Zero(n_sparse, n_struc_labels + n_env_labels);
  Kuf.block(0, 0, n_sparse, n_struc_labels) = Kuf_struc;
  Kuf.block(0, n_struc_labels, n_sparse, n_env_labels) = Kuf_env;

  // Combine noise_struc and noise_env.
  Eigen::VectorXd noise = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  noise.segment(0, n_struc_labels) = noise_struc;
  noise.segment(n_struc_labels, n_env_labels) = noise_env;

  // Combine training labels.
  Eigen::VectorXd y = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  y.segment(0, n_struc_labels) = y_struc;
  y.segment(n_struc_labels, n_env_labels) = y_env;

  // Solve for alpha with an iterative conjugate gradient algorithm.
  Eigen::MatrixXd sigma_inv =
      Kuu + Kuf * noise.asDiagonal() * Kuf.transpose() +
      Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
  Eigen::VectorXd b = Kuf * noise.asDiagonal() * y;

  Eigen::ConjugateGradient<Eigen::MatrixXd> cg;
  cg.compute(sigma_inv);
  alpha = cg.solve(b);
}

void SparseGP::compute_beta(int kernel_index, int descriptor_index) {
  // Assumptions:
  //  1. power = 2, so that the potential is linear in the square of the power
  //  spectrum.
  // 2. There is at least one sparse environment stored in the sparse GP.

  // TODO: Compute and store betas for all descriptors.

  // Initialize beta vector.
  int p_size = sparse_environments[0].descriptor_vals[descriptor_index].size();
  int beta_size = p_size * (p_size + 1) / 2;

  // Should be a better way to access the number of species.
  // Consider assigning descriptor calculators to the sparse GP object rather
  // than to local environments (also reduces memory).
  int n_species = sparse_environments[0]
                      .descriptor_calculators[descriptor_index]
                      ->descriptor_settings[0];

  beta = Eigen::MatrixXd::Zero(n_species, beta_size);

  double sig = kernels[kernel_index]->kernel_hyperparameters[0];
  double sig2 = sig * sig;
  int n_sparse = sparse_environments.size();

  for (int i = 0; i < n_sparse; i++) {
    int species = sparse_environments[i].central_species;
    Eigen::VectorXd p_current =
        sparse_environments[i].descriptor_vals[descriptor_index];
    double p_norm = sparse_environments[i].descriptor_norm[descriptor_index];
    double alpha_val = alpha(i);
    int beta_count = 0;

    for (int j = 0; j < p_size; j++) {
      double p_ij = p_current(j) / p_norm;

      for (int k = j; k < p_size; k++) {
        double p_ik = p_current(k) / p_norm;
        double beta_val = sig2 * p_ij * p_ik * alpha_val;

        // Update beta vector.
        if (j != k) {
          beta(species, beta_count) += 2 * beta_val;
        } else {
          beta(species, beta_count) += beta_val;
        }

        beta_count++;
      }
    }
  }
}

void SparseGP::write_beta(std::string file_name, std::string contributor,
                          int descriptor_index) {

  std::ofstream beta_file;

  beta_file.open(file_name);

  // Record the date.
  time_t now = std::time(0);
  std::string t(ctime(&now));
  beta_file << "DATE: ";
  beta_file << t.substr(0, t.length() - 1) << " ";

  // Record the contributor.
  beta_file << "CONTRIBUTOR: ";
  beta_file << contributor << "\n";

  // Report radial basis set.
  std::string radial_basis = sparse_environments[0]
                                 .descriptor_calculators[descriptor_index]
                                 ->radial_basis;
  beta_file << radial_basis << "\n";

  // Record number of species, nmax, lmax, and the cutoff.
  std::vector<int> descriptor_settings =
      sparse_environments[0]
          .descriptor_calculators[descriptor_index]
          ->descriptor_settings;
  int n_species = descriptor_settings[0];
  int n_max = descriptor_settings[1];
  int l_max = descriptor_settings[2];

  double cutoff = sparse_environments[0].many_body_cutoffs[descriptor_index];

  beta_file << n_species << " " << n_max << " " << l_max << " ";

  // Report the size of each beta vector.
  beta_file << beta.row(0).size() << "\n";

  // Report the cutoff function.
  std::string cutoff_func = sparse_environments[0]
                                .descriptor_calculators[descriptor_index]
                                ->cutoff_function;
  beta_file << cutoff_func << "\n";

  // Report cutoff to 2 decimal places.
  beta_file << std::fixed << std::setprecision(2);
  beta_file << cutoff << "\n";

  // Write beta vectors to file.
  beta_file << std::scientific << std::setprecision(16);

  int count = 0;
  for (int i = 0; i < n_species; i++) {
    Eigen::VectorXd beta_vals = beta.row(i);

    // Start a new line for each beta.
    if (count != 0) {
      beta_file << "\n";
    }

    for (int j = 0; j < beta_vals.size(); j++) {
      double beta_val = beta_vals[j];

      // Pad with 2 spaces if positive, 1 if negative.
      if (beta_val > 0) {
        beta_file << "  ";
      } else {
        beta_file << " ";
      }

      beta_file << beta_vals[j];
      count++;

      // New line if 5 numbers have been added.
      if (count == 5) {
        count = 0;
        beta_file << "\n";
      }
    }
  }

  beta_file.close();
}

Eigen::VectorXd SparseGP::predict(const StructureDescriptor &test_structure) {
  int n_atoms = test_structure.nat;
  int n_out = 1 + 3 * n_atoms + 6;
  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();
  Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(n_out, n_sparse);

// Compute the kernel between the test structure and each sparse
// environment, parallelizing over environments.
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_sparse; i++) {
    for (int j = 0; j < n_kernels; j++) {
      kern_mat.col(i) +=
          kernels[j]->env_struc(sparse_environments[i], test_structure);
    }
  }

  return kern_mat * alpha;
}

Eigen::VectorXd
SparseGP::predict_force(const LocalEnvironment &test_environment) {

  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();
  Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(3, n_sparse);

  for (int i = 0; i < n_sparse; i++) {
    for (int j = 0; j < n_kernels; j++) {
      kern_mat.col(i) +=
          kernels[j]->env_env_force(sparse_environments[i], test_environment);
    }
  }

  return kern_mat * alpha;
}

double
SparseGP::predict_local_energy(const LocalEnvironment &test_environment) {

  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();
  Eigen::VectorXd kern_vec = Eigen::VectorXd::Zero(n_sparse);

  for (int i = 0; i < n_sparse; i++) {
    for (int j = 0; j < n_kernels; j++) {
      kern_vec(i) +=
          kernels[j]->env_env(sparse_environments[i], test_environment);
    }
  }

  return kern_vec.dot(alpha);
}

void SparseGP::clear_environment_lists() {
  sparse_environments.clear();
  training_environments.clear();
  training_structures.clear();
}
