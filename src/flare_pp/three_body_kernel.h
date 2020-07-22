#ifndef THREE_BODY_KERNEL_H
#define THREE_BODY_KERNEL_H

#include "kernels.h"
#include <Eigen/Dense>
#include <vector>

class LocalEnvironment;
class StructureDescriptor;

class ThreeBodyKernel : public Kernel {
public:
  double sigma, sig2, ls, ls1, ls2, ls3;
  void (*cutoff_pointer)(double *, double, double, std::vector<double>);
  std::vector<double> cutoff_hyps;

  ThreeBodyKernel();

  ThreeBodyKernel(double sigma, double ls, const std::string &cutoff_function,
                  std::vector<double> cutoff_hyps);

  double env_env(const LocalEnvironment &env1, const LocalEnvironment &env2);

  Eigen::VectorXd env_env_force(const LocalEnvironment &env1,
                                const LocalEnvironment &env2);

  Eigen::VectorXd self_kernel_env(const StructureDescriptor &struc1, int atom);

  Eigen::VectorXd self_kernel_struc(const StructureDescriptor &struc);

  Eigen::VectorXd env_struc_partial(const LocalEnvironment &env1,
                                    const StructureDescriptor &struc1,
                                    int atom);

  Eigen::VectorXd env_struc(const LocalEnvironment &env1,
                            const StructureDescriptor &struc1);

  Eigen::MatrixXd kernel_transform(Eigen::MatrixXd kernels,
                                   Eigen::VectorXd new_hyps);

  std::vector<Eigen::MatrixXd> kernel_gradient(Eigen::MatrixXd kernels,
                                               Eigen::VectorXd new_hyps);

  void env_struc_update(Eigen::VectorXd &kernel_vector, int no_elements, int i,
                        double vol_inv, double r11, double r22, double r33,
                        double fi, double fj, double fdjx1, double fdjx2,
                        double fdjy1, double fdjy2, double fdjz1, double fdjz2,
                        double xrel1, double xval1, double xrel2, double xval2,
                        double yrel1, double yval1, double yrel2, double yval2,
                        double zrel1, double zval1, double zrel2, double zval2);

  void env_env_update_1(Eigen::VectorXd &kernel_vector, int no_elements, int i,
                        double vol_inv, double r11, double r22, double r33,
                        double fi, double fj, double fdix1, double fdix2,
                        double fdjx1, double fdjx2, double fdiy1, double fdiy2,
                        double fdjy1, double fdjy2, double fdiz1, double fdiz2,
                        double fdjz1, double fdjz2, double xrel_i1,
                        double xval_i1, double xrel_i2, double xval_i2,
                        double yrel_i1, double yval_i1, double yrel_i2,
                        double yval_i2, double zrel_i1, double zval_i1,
                        double zrel_i2, double zval_i2, double xrel_j1,
                        double xval_j1, double xrel_j2, double xval_j2,
                        double yrel_j1, double yval_j1, double yrel_j2,
                        double yval_j2, double zrel_j1, double zval_j1,
                        double zrel_j2, double zval_j2);

  void env_env_update_2(Eigen::VectorXd &kernel_vector, int no_elements, int i,
                        double vol_inv, double r11, double r22, double r33,
                        double fi, double fj, double fdix1, double fdix2,
                        double fdjx1, double fdjx2, double fdiy1, double fdiy2,
                        double fdjy1, double fdjy2, double fdiz1, double fdiz2,
                        double fdjz1, double fdjz2, double xrel_i1,
                        double xval_i1, double xrel_i2, double xval_i2,
                        double yrel_i1, double yval_i1, double yrel_i2,
                        double yval_i2, double zrel_i1, double zval_i1,
                        double zrel_i2, double zval_i2, double xrel_j1,
                        double xval_j1, double xrel_j2, double xval_j2,
                        double yrel_j1, double yval_j1, double yrel_j2,
                        double yval_j2, double zrel_j1, double zval_j1,
                        double zrel_j2, double zval_j2);

  void struc_struc_update_1(
      Eigen::VectorXd &kernel_vector, int no_elements, int i, int j,
      double mult_factor, double vol_inv, double r11, double r22, double r33,
      double fi, double fj, double fdix1, double fdix2, double fdjx1,
      double fdjx2, double fdiy1, double fdiy2, double fdjy1, double fdjy2,
      double fdiz1, double fdiz2, double fdjz1, double fdjz2, double xrel_i1,
      double xval_i1, double xrel_i2, double xval_i2, double yrel_i1,
      double yval_i1, double yrel_i2, double yval_i2, double zrel_i1,
      double zval_i1, double zrel_i2, double zval_i2, double xrel_j1,
      double xval_j1, double xrel_j2, double xval_j2, double yrel_j1,
      double yval_j1, double yrel_j2, double yval_j2, double zrel_j1,
      double zval_j1, double zrel_j2, double zval_j2);

  std::vector<double> force_stress_helper_1(
      double mult_factor, double vol_inv_sq, double k_SE, double r11,
      double r22, double r33, double fi, double fj, double fdix1, double fdix2,
      double fdjx1, double fdjx2, double xrel_i1, double xrel_i2,
      double xrel_j1, double xrel_j2, double xval_i1, double xval_i2,
      double yval_i1, double yval_i2, double zval_i1, double zval_i2,
      double xval_j1, double xval_j2, double yval_j1, double yval_j2,
      double zval_j1, double zval_j2);

  void struc_struc_update_2(
      Eigen::VectorXd &kernel_vector, int no_elements, int i, int j,
      double mult_factor, double vol_inv, double r12, double r23, double r31,
      double fi, double fj, double fdix1, double fdix2, double fdjx1,
      double fdjx2, double fdiy1, double fdiy2, double fdjy1, double fdjy2,
      double fdiz1, double fdiz2, double fdjz1, double fdjz2, double xrel_i1,
      double xval_i1, double xrel_i2, double xval_i2, double yrel_i1,
      double yval_i1, double yrel_i2, double yval_i2, double zrel_i1,
      double zval_i1, double zrel_i2, double zval_i2, double xrel_j1,
      double xval_j1, double xrel_j2, double xval_j2, double yrel_j1,
      double yval_j1, double yrel_j2, double yval_j2, double zrel_j1,
      double zval_j1, double zrel_j2, double zval_j2);

  std::vector<double> force_stress_helper_2(
      double mult_factor, double vol_inv_sq, double k_SE, double r12,
      double r23, double r31, double fi, double fj, double fdix1, double fdix2,
      double fdjx1, double fdjx2, double xrel_i1, double xrel_i2,
      double xrel_j1, double xrel_j2, double xval_i1, double xval_i2,
      double yval_i1, double yval_i2, double zval_i1, double zval_i2,
      double xval_j1, double xval_j2, double yval_j1, double yval_j2,
      double zval_j1, double zval_j2);
};

#endif
