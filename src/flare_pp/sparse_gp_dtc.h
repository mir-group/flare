#ifndef SPARSE_GP_DTC_H
#define SPARSE_GP_DTC_H

#include "sparse_gp.h"
#include <Eigen/Dense>

class SparseGP_DTC : public SparseGP {
public:
  std::vector<Eigen::MatrixXd> Kuf_kernels, Kuu_kernels;

  SparseGP_DTC();
  SparseGP_DTC(std::vector<Kernel *> kernels, double sigma_e, double sigma_f,
               double sigma_s);
};

#endif
