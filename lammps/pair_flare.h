// Jonathan Vandermause
// Pair style based on pair_sw.h

#ifdef PAIR_CLASS

PairStyle(flare,PairFLARE)

#else

#ifndef LMP_PAIR_FLARE_H
#define LMP_PAIR_FLARE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairFLARE : public Pair {
 public:
  PairFLARE(class LAMMPS *);
  virtual ~PairFLARE();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();

  struct Param {
    double epsilon,sigma;
    double littlea,lambda,gamma,costheta;
    double biga,bigb;
    double powerp,powerq;
    double tol;
    double cut,cutsq;
    double sigma_gamma,lambda_epsilon,lambda_epsilon2;
    double c1,c2,c3,c4,c5,c6;
    int ielement,jelement,kelement;
  };

 protected:
  double cutmax;                // max cutoff for all elements
  int nelements;                // # of unique elements
  char **elements;              // names of unique elements
  int ***elem2param;            // mapping from element triplets to parameters
  int *map;                     // mapping from atom types to elements
  int nparams;                  // # of stored parameter sets
  int maxparam;                 // max # of parameter sets
  Param *params;                // parameter set for an I-J-K interaction
  int maxshort;                 // size of short neighbor list array
  int *neighshort;              // short neighbor list array

  virtual void allocate();
  void read_file(char *);
  virtual void setup_params();
  void twobody(Param *, double, double &, int, double &);
  void threebody(Param *, Param *, Param *, double, double, double *, double *,
                 double *, double *, int, double &);
};

}

#endif
#endif