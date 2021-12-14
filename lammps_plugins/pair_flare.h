// Jonathan Vandermause
// Pair style based on pair_eam.h

#ifdef PAIR_CLASS

PairStyle(flare, PairFLARE)

#else

#ifndef LMP_PAIR_FLARE_H
#define LMP_PAIR_FLARE_H

#include "pair.h"
#include <Eigen/Dense>
#include <cstdio>
#include <vector>

namespace LAMMPS_NS {

class PairFLARE : public Pair {
public:
  PairFLARE(class LAMMPS *);
  virtual ~PairFLARE();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual void init_style();
  double init_one(int, int);

protected:
  int power, n_species, n_max, l_max, n_descriptors, beta_size;

  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      basis_function;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function;

  std::vector<double> radial_hyps, cutoff_hyps;

  double cutoff;
  double *beta, *cutoffs;
  Eigen::MatrixXd beta_matrix, cutoff_matrix;
  std::vector<Eigen::MatrixXd> beta_matrices;

  virtual void allocate();
  virtual void read_file(char *);
  void grab(FILE *, int, double *);
};

} // namespace LAMMPS_NS

#endif
#endif
