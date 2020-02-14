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
        DescriptorCalculator * descriptor_calculator;
        std::vector<LocalEnvironment> environment_descriptors;
        double cutoff;
        std::vector<double> nested_cutoffs;

        // Make structure labels empty by default.
        std::vector<double> energy {};
        std::vector<double> forces {};
        std::vector<double> stresses {};

        StructureDescriptor();

        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            double cutoff);

        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            double cutoff, std::vector<double> nested_cutoffs);

        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            double cutoff,
                            DescriptorCalculator * descriptor_calculator);

        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            double cutoff, std::vector<double> nested_cutoffs,
                            DescriptorCalculator * descriptor_calculator);

        void compute_environments();
        void compute_nested_environments();
        void compute_descriptors();
        void nested_descriptors();
};

#endif