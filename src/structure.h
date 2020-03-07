#ifndef STRUCTURE_H
#define STRUCTURE_H

#include <vector>
#include <Eigen/Dense>

#include "descriptor.h"

// Structure class.
class Structure{
    public:
        Eigen::MatrixXd cell, cell_transpose, cell_transpose_inverse,
                        cell_dot, cell_dot_inverse, positions,
                        wrapped_positions;
        std::vector<int> species;
        double max_cutoff, volume;
        int noa;

        Structure();

        Structure(const Eigen::MatrixXd & cell,
                  const std::vector<int> & species,
                  const Eigen::MatrixXd & positions);

        Eigen::MatrixXd wrap_positions();

        double get_max_cutoff();
};

// Structure descriptor. Stores the atomic environments in a structure.
class StructureDescriptor : public Structure{
    public:
        std::vector<DescriptorCalculator *> descriptor_calculators;
        std::vector<LocalEnvironment> local_environments;
        double cutoff;
        std::vector<double> n_body_cutoffs;
        std::vector<double> many_body_cutoffs;

        // Make structure labels empty by default.
        Eigen::VectorXd energy;
        Eigen::VectorXd forces;
        Eigen::VectorXd stresses;

        StructureDescriptor();

        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            double cutoff);

        // n-body
        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            double cutoff, std::vector<double> n_body_cutoffs);

        // many-body
        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions, double cutoff,
                            std::vector<double> many_body_cutoffs,
                            std::vector<DescriptorCalculator *>
                                descriptor_calculators);

        // n-body + many-body
        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            double cutoff, std::vector<double> n_body_cutoffs,
                            std::vector<double> many_body_cutoffs,
                            std::vector<DescriptorCalculator *>
                                descriptor_calculators);

        void compute_environments();
        void compute_nested_environments();
        void compute_descriptors();
        void nested_descriptors();
};

#endif