#ifndef KERNELS_H
#define KERNELS_H

#include <vector>
#include <Eigen/Dense>

class LocalEnvironment;
class LocalEnvironmentDescriptor;
class StructureDescriptor;

class DotProductKernel{
    public:
        double signal_variance, power, sig2;

        DotProductKernel();

        DotProductKernel(double signal_variance, double power);

        double env_env(const LocalEnvironmentDescriptor & env1,
                       const LocalEnvironmentDescriptor & env2);

        Eigen::VectorXd env_struc(const LocalEnvironmentDescriptor & env1,
                                  const StructureDescriptor & struc1);

};

class TwoBodyKernel{
    public:
        double ls, ls1, ls2;
        void (*cutoff_pointer)(double *, double, double, std::vector<double>);
        std::vector<double> cutoff_hyps;

        TwoBodyKernel();

        TwoBodyKernel(double ls, const std::string & cutoff_function,
                      std::vector<double> cutoff_hyps);

        double env_env(const LocalEnvironment & env1,
                       const LocalEnvironment & env2);
        
        Eigen::VectorXd env_struc(const LocalEnvironment & env1,
                                  const StructureDescriptor & struc1);
};

class ThreeBodyKernel{
    public:
        double ls, ls1, ls2;
        void (*cutoff_pointer)(double *, double, double, std::vector<double>);
        std::vector<double> cutoff_hyps;

        ThreeBodyKernel();
        
        ThreeBodyKernel(double ls, const std::string & cutoff_function,
                        std::vector<double> cutoff_hyps);

        double env_env(const LocalEnvironment & env1,
                       const LocalEnvironment & env2);

        Eigen::VectorXd env_struc(const LocalEnvironment & env1,
                                  const StructureDescriptor & struc1);
};

#endif