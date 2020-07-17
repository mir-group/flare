#include "sparse_gp_dtc.h"

SparseGP_DTC ::SparseGP_DTC() {}

SparseGP_DTC ::SparseGP_DTC(std::vector<Kernel *> kernels, double sigma_e,
                            double sigma_f, double sigma_s)
    : SparseGP(kernels, sigma_e, sigma_f, sigma_s) {
    
    // Initialize kernel lists.
    Eigen::MatrixXd empty_matrix;
    for (int i = 0; i < kernels.size(); i++){
        Kuf_env_kernels.push_back(empty_matrix);
        Kuf_struc_kernels.push_back(empty_matrix);
        Kuu_kernels.push_back(empty_matrix);
    }
    }

void SparseGP_DTC ::add_sparse_environments(
    const std::vector<LocalEnvironment> &envs) {

  int n_envs = envs.size();
  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();
  int n_labels = Kuf_struc.cols();
  int n_strucs = training_structures.size();

  // Compute kernels between new environment and previous sparse
  // environments.
  std::vector<Eigen::MatrixXd> prev_blocks;
  for (int i = 0; i < n_kernels; i++){
      prev_blocks.push_back(Eigen::MatrixXd::Zero(n_sparse, n_envs));
  }

#pragma omp parallel for schedule(static)
  for (int k = 0; k < n_envs; k++) {
    for (int i = 0; i < n_sparse; i++) {
      for (int j = 0; j < n_kernels; j++) {
        prev_blocks[j](i, k) +=
            kernels[j]->env_env(sparse_environments[i], envs[k]);
      }
    }
  }

  // Compute self block. (Note that the work can be cut in half by exploiting
  // the symmetry of the matrix, but this makes parallelization slightly
  // trickier.)
  std::vector<Eigen::MatrixXd> self_blocks;
  for (int i = 0; i < n_kernels; i++){
      self_blocks.push_back(Eigen::MatrixXd::Zero(n_envs, n_envs));
  }

#pragma omp parallel for schedule(static)
  for (int k = 0; k < n_envs; k++) {
    for (int l = 0; l < n_envs; l++) {
      for (int j = 0; j < n_kernels; j++) {
        self_blocks[j](k, l) += kernels[j]->env_env(envs[k], envs[l]);
      }
    }
  }

  // Update Kuu matrices.
  Eigen::MatrixXd self_block = Eigen::MatrixXd::Zero(n_envs, n_envs);
  Eigen::MatrixXd prev_block = Eigen::MatrixXd::Zero(n_sparse, n_envs);
  for (int i = 0; i < n_kernels; i++){
    Kuu_kernels[i].conservativeResize(n_sparse + n_envs, n_sparse + n_envs);
    Kuu_kernels[i].block(0, n_sparse, n_sparse, n_envs) = prev_blocks[i];
    Kuu_kernels[i].block(n_sparse, 0, n_envs, n_sparse) =
        prev_blocks[i].transpose();
    Kuu_kernels[i].block(n_sparse, n_sparse, n_envs, n_envs) = self_blocks[i];

    self_block += self_blocks[i];
    prev_block += prev_blocks[i];
  }

  Kuu.conservativeResize(n_sparse + n_envs, n_sparse + n_envs);
  Kuu.block(0, n_sparse, n_sparse, n_envs) = prev_block;
  Kuu.block(n_sparse, 0, n_envs, n_sparse) = prev_block.transpose();
  Kuu.block(n_sparse, n_sparse, n_envs, n_envs) = self_block;

  // Compute kernels between new sparse environment and training structures.
  std::vector<Eigen::MatrixXd> uf_blocks;
  for (int i = 0; i < n_kernels; i++){
      uf_blocks.push_back(Eigen::MatrixXd::Zero(n_envs, n_labels));
  }

// TODO: Check parallel version -- may need to eliminate kernel vector initialization inside the second for loop.
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n_envs; i++){
    for (int j = 0; j < n_strucs; j++) {
        int initial_index, index;

        // Get initial index.
        if (j == 0) {
            initial_index = 0;
        } else {
            initial_index = label_count[j - 1];
        }
        index = initial_index;

        // Initialize kernel vector.
        int n_atoms = training_structures[j].noa;
        Eigen::VectorXd kernel_vector = 
            Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);

        for (int k = 0; k < n_kernels; k++){
            kernel_vector =
                kernels[k]->env_struc(envs[i], training_structures[j]);

            if (training_structures[j].energy.size() != 0) {
                uf_blocks[k](i, index) = kernel_vector(0);
                index += 1;
            }

            if (training_structures[j].forces.size() != 0) {
                uf_blocks[k].row(i).segment(index, n_atoms * 3) =
                    kernel_vector.segment(1, n_atoms * 3);
                index += n_atoms * 3;
            }

            if (training_structures[j].stresses.size() != 0) {
                uf_blocks[k].row(i).segment(index, 6) = kernel_vector.tail(6);
            }
        }
    }
  }
  // Update Kuf_struc matrices.
  Eigen::MatrixXd uf_block = Eigen::MatrixXd::Zero(n_envs, n_labels);
  for (int i = 0; i < n_kernels; i++){
    Kuf_struc_kernels[i].conservativeResize(n_sparse + n_envs, n_labels);
    Kuf_struc_kernels[i].block(n_sparse, 0, n_envs, n_labels) = uf_blocks[i];

    uf_block += uf_blocks[i];
  }

  Kuf_struc.conservativeResize(n_sparse + n_envs, n_labels);
  Kuf_struc.block(n_sparse, 0, n_envs, n_labels) = uf_block;

  // Store sparse environments.
  for (int i = 0; i < n_envs; i++) {
    sparse_environments.push_back(envs[i]);
  }
}

// TODO: Update kernel lists.
void SparseGP_DTC ::add_training_structure(
    const StructureDescriptor &training_structure) {

  int n_labels = training_structure.energy.size() +
                 training_structure.forces.size() +
                 training_structure.stresses.size();

  if (n_labels > max_labels) max_labels = n_labels;

  int n_atoms = training_structure.noa;
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


void SparseGP_DTC ::update_matrices() {
  // Combine Kuf_struc and Kuf_env.
  int n_sparse = Kuf_struc.rows();
  int n_struc_labels = Kuf_struc.cols();
  int n_env_labels = Kuf_env.cols();
  Kuf = Eigen::MatrixXd::Zero(n_sparse, n_struc_labels + n_env_labels);
  Kuf.block(0, 0, n_sparse, n_struc_labels) = Kuf_struc;
  Kuf.block(0, n_struc_labels, n_sparse, n_env_labels) = Kuf_env;

  // Combine noise_struc and noise_env.
  noise_vector = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  noise_vector.segment(0, n_struc_labels) = noise_struc;
  noise_vector.segment(n_struc_labels, n_env_labels) = noise_env;

  // Combine training labels.
  y = Eigen::VectorXd::Zero(n_struc_labels + n_env_labels);
  y.segment(0, n_struc_labels) = y_struc;
  y.segment(n_struc_labels, n_env_labels) = y_env;

  // Calculate Sigma.
  Eigen::MatrixXd sigma_inv =
      Kuu + Kuf * noise_vector.asDiagonal() * Kuf.transpose() +
      Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());

  Sigma = sigma_inv.inverse();

  // Calculate Kuu inverse.
  Kuu_inverse = Kuu.inverse();

  // Calculate alpha.
  alpha = Sigma * Kuf * noise_vector.asDiagonal() * y;
}

void SparseGP_DTC ::predict_DTC(
    StructureDescriptor test_structure, Eigen::VectorXd &mean_vector,
    Eigen::VectorXd &variance_vector,
    std::vector<Eigen::VectorXd> &mean_contributions) {

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;
  int n_sparse = sparse_environments.size();
  int n_kernels = kernels.size();

  // Store kernel matrices for each kernel.
  std::vector<Eigen::MatrixXd> kern_mats;
  for (int i = 0; i < n_kernels; i++) {
    kern_mats.push_back(Eigen::MatrixXd::Zero(n_out, n_sparse));
  }
  Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(n_out, n_sparse);

// Compute the kernel between the test structure and each sparse
// environment, parallelizing over environments.
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_sparse; i++) {
    for (int j = 0; j < n_kernels; j++) {
      kern_mats[j].col(i) +=
          kernels[j]->env_struc(sparse_environments[i], test_structure);
    }
  }

  // Sum the kernels.
  for (int i = 0; i < n_kernels; i++) {
    kern_mat += kern_mats[i];
  }

  // Compute mean contributions and total mean prediction.
  mean_contributions = std::vector<Eigen::VectorXd>{};
  for (int i = 0; i < n_kernels; i++) {
    mean_contributions.push_back(kern_mats[i] * alpha);
  }

  mean_vector = kern_mat * alpha;

  // Compute variances.
  Eigen::VectorXd V_SOR, Q_self, K_self = Eigen::VectorXd::Zero(n_out);

  // Note: Calculation of the self-kernel can be parallelized.
  for (int i = 0; i < n_kernels; i++) {
    K_self += kernels[i]->self_kernel_struc(test_structure);
  }

  Q_self = (kern_mat * Kuu_inverse * kern_mat.transpose()).diagonal();
  V_SOR = (kern_mat * Sigma * kern_mat.transpose()).diagonal();

  variance_vector = K_self - Q_self + V_SOR;
}

void SparseGP_DTC ::compute_DTC_likelihood(){
    int n_train = Kuf.cols();

    Eigen::MatrixXd Qff_plus_lambda = 
        Kuf.transpose() * Kuu_inverse * Kuf +
        noise_vector.asDiagonal() * Eigen::MatrixXd::Identity(n_train, n_train);

    double Q_det = Qff_plus_lambda.determinant();
    Eigen::MatrixXd Q_inv = Qff_plus_lambda.inverse();

    double half = 1.0 / 2.0;
    complexity_penalty = -half * log(Q_det);
    data_fit = -half * y.transpose() * Q_inv * y;
    constant_term = -half * n_train * log(2 * M_PI);
    log_marginal_likelihood = complexity_penalty + data_fit + constant_term;
}

void SparseGP_DTC ::compute_VFE_likelihood(){

}
