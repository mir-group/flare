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
class LocalEnvironmentDescriptor;

class StructureDescriptor : public Structure{
    public:
        DescriptorCalculator descriptor_calculator;
        std::vector<LocalEnvironmentDescriptor> environment_descriptors;
        double cutoff;

        StructureDescriptor();

        // If a descriptor calculator isn't given, store the environments
        // only (without descriptor vectors).
        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            double cutoff);

        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            DescriptorCalculator & descriptor_calculator,
                            double cutoff);

        void compute_environments();
        void compute_descriptors();
};

// Structure dataset. Stores energy, force, and stress labels.
class StructureDataset : public StructureDescriptor{
    public:
        std::vector<double> energy;
        std::vector<double> force_components;
        std::vector<double> stress_components;

        StructureDataset();

        StructureDataset(const Eigen::MatrixXd & cell,
                         const std::vector<int> & species,
                         const Eigen::MatrixXd & positions,
                         DescriptorCalculator & descriptor_calculator,
                         double cutoff,
                         std::vector<double> energy = std::vector<double>{},
                         std::vector<double> force_components =
                            std::vector<double>{},
                         std::vector<double> stress_components =
                            std::vector<double>{});

};

#endif