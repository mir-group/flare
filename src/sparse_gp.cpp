#include "sparse_gp.h"
#include <cmath>
#include <iostream>

SparseGP :: SparseGP(){}

SparseGP :: SparseGP(std::vector<Kernel *> kernels){
    this->kernels = kernels;
}

void SparseGP :: add_sparse_environment(LocalEnvironment env){
    // Compute kernels between new environment and previous sparse
    // environments.
    int n_envs = sparse_environments.size();
    int n_kernels = kernels.size();
    LocalEnvironment sparse_env;
    Eigen::VectorXd kernel_vector = Eigen::VectorXd::Zero(n_envs + 1);
    for (int i = 0; i < n_envs; i ++){
        sparse_env = sparse_environments[i];
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector(i) += kernels[j] -> env_env(sparse_env, env);
        }
    }

    // Compute self kernel.
    double self_kernel = 0;
    for (int j = 0; j < n_kernels; j ++){
        self_kernel += kernels[j] -> env_env(env, env);
    }

    // Update Kuu matrix.
    Kuu.conservativeResize(Kuu.rows()+1, Kuu.cols()+1);
    Kuu.col(Kuu.cols()-1) = kernel_vector;
    Kuu.row(Kuu.rows()-1) = kernel_vector;
    Kuu(n_envs, n_envs) = self_kernel;

    // TODO: Update Kuf matrix.

    // Store sparse environment.
    sparse_environments.push_back(env);
}

void SparseGP :: add_training_structure(StructureDescriptor training_structure){

    int n_labels = training_structure.energy.size() +
        training_structure.forces.size() + training_structure.stresses.size();
    int n_atoms = training_structure.noa;
    int n_sparse = sparse_environments.size();
    int n_kernels = kernels.size();

    // Calculate kernels between sparse environments and training structure.
    Eigen::MatrixXd kernel_block = Eigen::MatrixXd::Zero(n_sparse, n_labels);
    LocalEnvironment sparse_env;
    Eigen::VectorXd kernel_vector;
    int label_count;
    for (int i = 0; i < n_sparse; i ++){
        sparse_env = sparse_environments[i];
        kernel_vector = Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector += kernels[j] -> env_struc(sparse_env,
                training_structure);
        }

        // Update kernel block.
        label_count = 0;
        if (training_structure.energy.size() != 0){
            kernel_block(i, 0) = kernel_vector(0);
            label_count += 1;
        }

        if (training_structure.forces.size() != 0){
            kernel_block.row(i).segment(label_count, n_atoms * 3) =
                kernel_vector.segment(1, n_atoms * 3);
        }

        if (training_structure.stresses.size() != 0){
            kernel_block.row(i).tail(6) = kernel_vector.tail(6);
        }
    }

    // Add kernel block to Kuf.
    int prev_cols = Kuf.cols();
    Kuf.conservativeResize(n_sparse, prev_cols + n_labels);
    Kuf.block(0, prev_cols, n_sparse, n_labels) = kernel_block;

    // Store training structure.
    training_structures.push_back(training_structure);
}
