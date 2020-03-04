#ifndef SPARSE_GP_H
#define SPARSE_GP_H

#include <vector>
#include <Eigen/Dense>

#include "local_environment.h"
#include "structure.h"
#include "kernels.h"

class SparseGP{
    public:
        Eigen::MatrixXd Kuu, Kuf, Sigma, noise_matrix;
        Eigen::VectorXd y, alpha, hyperparameters, noise;
        std::vector<Kernel *> kernels;

        std::vector<LocalEnvironment> sparse_environments;
        std::vector<StructureDescriptor> training_structures;

        double energy_norm, forces_norm, stresses_norm,
            energy_offset, forces_offset, stresses_offset;

        double sigma_e, sigma_f, sigma_s;

        SparseGP();

        SparseGP(std::vector<Kernel *>, double sigma_e, double sigma_f,
            double sigma_s);

        void add_sparse_environment(LocalEnvironment env);
        void add_training_structure(StructureDescriptor training_structure);
        void update_alpha();
        Eigen::VectorXd predict(StructureDescriptor test_structure);
        Eigen::VectorXd predict_serial(StructureDescriptor test_structure);
};

#endif