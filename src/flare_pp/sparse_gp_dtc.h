#ifndef SPARSE_GP_DTC_H
#define SPARSE_GP_DTC_H

#include "sparse_gp.h"
#include <Eigen/Dense>

class SparseGP_DTC : public SparseGP {
public:
  std::vector<Eigen::MatrixXd> Kuf_kernels, Kuu_kernels;
  Eigen::MatrixXd Sigma, Kuu_inverse;

  SparseGP_DTC();
  SparseGP_DTC(std::vector<Kernel *> kernels, double sigma_e, double sigma_f,
               double sigma_s);

  // Update matrices and vectors needed for mean and variance prediction:
  // Sigma, Kuu_inverse, and alpha.
  void update_matrices();

  // Compute the DTC mean and variance.
  void predict_DTC(StructureDescriptor test_structure,
      Eigen::VectorXd & mean_vector, Eigen::VectorXd & variance_vector,
      std::vector<Eigen::VectorXd> & mean_contributions);

  // Calculate the log marginal likelihood of the current hyperparameters and its gradient.
  // TODO: Find a way to optimize the hyperparameters.
  double compute_likelihood();

  // Change the model hyperparameters and covariance matrices.
  // With the dot product kernel, should never have to recompute kernels;
  // just rescale them.
  void set_hyperparameters(Eigen::VectorXd hyperparameters);
};

#endif
