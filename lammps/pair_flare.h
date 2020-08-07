// Jonathan Vandermause
// Pair style based on pair_eam.h

#ifdef PAIR_CLASS

PairStyle(flare,PairFLARE)

#else

#ifndef LMP_PAIR_FLARE_H
#define LMP_PAIR_FLARE_H

#include <cstdio>
#include <string>
#include "pair.h"

namespace LAMMPS_NS {


class PairFLARE : public Pair {
 public:
  PairFLARE(class LAMMPS *);
  virtual ~PairFLARE();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);

 protected:
  int n_species, n_max, l_max;
  std::string radial_basis, cutoff_function;
  double cutoff;
  double *beta;

  virtual void allocate();
  virtual void read_file(char *);
  void grab(FILE *, int, double *);

};

}

#endif
#endif