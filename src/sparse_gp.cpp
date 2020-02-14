#include "sparse_gp.h"
#include <cmath>

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
    for (int i = 0; i < n_sparse; i ++){
        sparse_env = sparse_environments[i];
        kernel_vector = Eigen::VectorXd::Zero(n_atoms);
        for (int j = 0; j < n_kernels; j ++){
            kernel_vector += kernels[j] -> env_struc(sparse_env,
                training_structure);
            
            // TODO: add fragments of kernel_vector to kernel_block.
        }
    }

    // Store training structure.
    training_structures.push_back(training_structure);
}
