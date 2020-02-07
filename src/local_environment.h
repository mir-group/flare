#ifndef LOCAL_ENVIRONMENT_H
#define LOCAL_ENVIRONMENT_H

#include "structure.h"
#include "descriptor.h"

// Local environment class.
class LocalEnvironment{
    public:
        std::vector<int> environment_indices, environment_species,
            neighbor_list, two_body_indices, many_body_indices;
        std::vector<std::vector<int>> three_body_indices;
        int central_index, central_species, noa, sweep;
        std::vector<double> rs, xs, ys, zs, xrel, yrel, zrel,
            nested_cutoffs, cross_bond_dists;
        double cutoff, structure_volume;

        DescriptorCalculator * descriptor_calculator;
        Eigen::VectorXd descriptor_vals;
        Eigen::MatrixXd descriptor_force_dervs;
        Eigen::MatrixXd descriptor_stress_dervs;
        double descriptor_norm;
        Eigen::MatrixXd force_dot, stress_dot;

        LocalEnvironment();

        LocalEnvironment(const Structure & structure, int atom,
                         double cutoff);
        
        // If nested cutoffs are given, compute 2-, 3-, and many-body indices.
        LocalEnvironment(const Structure & structure, int atom,
                         double cutoff, std::vector<double> nested_cutoffs);

        // If a descriptor calculator is given as well, compute and store the
        // many-body descriptor.
        LocalEnvironment(const Structure & structure, int atom,
                         double cutoff, std::vector<double> nested_cutoffs,
                         DescriptorCalculator & descriptor_calculator);

        static void compute_environment(const Structure & structure,
                                 int noa, int atom,
                                 double cutoff, int sweep_val,
                                 std::vector<int> & environment_indices,
                                 std::vector<int> & environment_species,
                                 std::vector<int> & neighbor_list,
                                 std::vector<double> & rs,
                                 std::vector<double> & xs,
                                 std::vector<double> & ys,
                                 std::vector<double> & zs,
                                 std::vector<double> & xrel,
                                 std::vector<double> & yrel,
                                 std::vector<double> & zrel);

        void compute_nested_environment();
        void compute_descriptor();
};

// Nested environments store 2-, 3-, and many-body indices.
class NestedEnvironment : public LocalEnvironment{
    public:
        std::vector<int> two_body_indices, many_body_indices;
        std::vector<std::vector<int>> three_body_indices;
        std::vector<double> cross_bond_dists;
        double two_body_cutoff, three_body_cutoff, many_body_cutoff;

    NestedEnvironment();

    NestedEnvironment(const Structure & structure, int atom, double cutoff,
                      double two_body_cutoff = 0, double three_body_cutoff = 0,
                      double many_body_cutoff = 0);

    void compute_nested_environment();
};

// Local environment descriptor.
class LocalEnvironmentDescriptor : public LocalEnvironment{
    public:
        DescriptorCalculator * descriptor_calculator;
        Eigen::VectorXd descriptor_vals;
        Eigen::MatrixXd descriptor_force_dervs;
        Eigen::MatrixXd descriptor_stress_dervs;

        double descriptor_norm;
        Eigen::MatrixXd force_dot, stress_dot;

        LocalEnvironmentDescriptor();

        LocalEnvironmentDescriptor(const Structure & structure, int atom,
            double cutoff);

        LocalEnvironmentDescriptor(const Structure & structure, int atom,
            double cutoff, DescriptorCalculator * descriptor_calculator);

        void compute_descriptor();
};

#endif