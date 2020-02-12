#ifndef SPARSE_GP_H
#define SPARSE_GP_H

#include <vector>
#include <Eigen/Dense>

class SparseGP{
    public:
        Eigen::MatrixXd Kuu, Kuf, Sigma;
        Eigen::VectorXd y;

        SparseGP();

}

#endif