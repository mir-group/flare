#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <vector>
#include <Eigen/Dense>

// Descriptor calculator.
class LocalEnvironment;

class DescriptorCalculator{
    protected:
        void (*radial_pointer)(double *, double *, double, int,
                               std::vector<double>);
        void (*cutoff_pointer)(double *, double, double, std::vector<double>);
    public:
        Eigen::VectorXd single_bond_vals, descriptor_vals;
        Eigen::MatrixXd single_bond_force_dervs, single_bond_stress_dervs,
            descriptor_force_dervs, descriptor_stress_dervs;
        std::string radial_basis, cutoff_function;
        std::vector<double> radial_hyps, cutoff_hyps;
        std::vector<int> descriptor_settings;

    DescriptorCalculator();

    DescriptorCalculator(
        const std::string & radial_basis, const std::string & cutoff_function,
        const std::vector<double> & radial_hyps,
        const std::vector<double> & cutoff_hyps,
        const std::vector<int> & descriptor_settings);

    virtual void compute(const LocalEnvironment & env) = 0;

    virtual ~DescriptorCalculator() = default;
};

void B2_descriptor(
Eigen::VectorXd & B2_vals,
Eigen::MatrixXd & B2_force_dervs,
Eigen::MatrixXd & B2_stress_dervs,
const Eigen::VectorXd & single_bond_vals,
const Eigen::MatrixXd & single_bond_force_dervs,
const Eigen::MatrixXd & single_bond_stress_dervs,
const LocalEnvironment & env, int nos, int N, int lmax);

class B1_Calculator : public DescriptorCalculator{
    public:
        B1_Calculator();

        B1_Calculator(
            const std::string & radial_basis,
            const std::string & cutoff_function,
            const std::vector<double> & radial_hyps,
            const std::vector<double> & cutoff_hyps,
            const std::vector<int> & descriptor_settings);

        void compute(const LocalEnvironment & env);
};

class B2_Calculator : public DescriptorCalculator{
    public:
        B2_Calculator();

        B2_Calculator(
            const std::string & radial_basis,
            const std::string & cutoff_function,
            const std::vector<double> & radial_hyps,
            const std::vector<double> & cutoff_hyps,
            const std::vector<int> & descriptor_settings);

        void compute(const LocalEnvironment & env);
};

#endif