// Yu Xie 
// Compute uncertainty per atom
// Based on pair_flare.h and compute_uncertainty_atom.h

#ifdef COMPUTE_CLASS

ComputeStyle(flare/std/atom, ComputeFlareStdAtom)

#else

#ifndef LMP_COMPUTE_FLARE_STD_ATOM_H
#define LMP_COMPUTE_FLARE_STD_ATOM_H

#include "compute.h"
#include <Eigen/Dense>
#include <cstdio>
#include <vector>

namespace LAMMPS_NS {

class ComputeFlareStdAtom : public Compute {
public:
  ComputeFlareStdAtom(class LAMMPS *, int, char **);
  ~ComputeFlareStdAtom();
  void compute_peratom();
  //void init_style();

  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();
  double init_one(int, int);
  void init();
  void init_list(int, class NeighList *);

protected:
  double *stds;
  class NeighList *list;
  int allocated=0;

  int n_species, n_max, l_max, n_descriptors, beta_size;

  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      basis_function;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_function;

  std::vector<double> radial_hyps, cutoff_hyps;

  double cutoff;
  double *beta;
  Eigen::MatrixXd beta_matrix;
  std::vector<Eigen::MatrixXd> beta_matrices;

  virtual void allocate();
  virtual void read_file(char *);
  void grab(FILE *, int, double *);

  virtual void coeff(int, char **);
};

} // namespace LAMMPS_NS

#endif
#endif
