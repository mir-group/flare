// Jonathan Vandermause
// Pair style based on pair_eam.h

#ifdef PAIR_CLASS

PairStyle(flare,PairFLARE)

#else

#ifndef LMP_PAIR_FLARE_H
#define LMP_PAIR_FLARE_H

#include <cstdio>
#include "pair.h"

namespace LAMMPS_NS {


class PairFLARE : public Pair {
 public:
  friend class FixSemiGrandCanonicalMC;   // Alex Stukowski option

  // public variables so USER-ATC package can access them

  double cutmax;

  // potentials as array data

  int nrho,nr;
  int nfrho,nrhor,nz2r;
  double **frho,**rhor,**z2r;
  int *type2frho,**type2rhor,**type2z2r;

  // potentials in spline form used for force computation

  double dr,rdr,drho,rdrho,rhomax;
  double ***rhor_spline,***frho_spline,***z2r_spline;

  PairFLARE(class LAMMPS *);
  virtual ~PairFLARE();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);

  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 protected:
  int nmax;                   // allocated size of per-atom arrays
  double cutforcesq;
  double **scale;

  // per-atom arrays

  double *rho,*fp;

  // potentials as file data

  int *map;                   // which element each atom type maps to

  struct Funcfl {
    char *file;
    int nrho,nr;
    double drho,dr,cut,mass;
    double *frho,*rhor,*zr;
  };
  Funcfl *funcfl;
  int nfuncfl;

  struct Setfl {
    char **elements;
    int nelements,nrho,nr;
    double drho,dr,cut;
    double *mass;
    double **frho,**rhor,***z2r;
  };
  Setfl *setfl;

  struct Fs {
    char **elements;
    int nelements,nrho,nr;
    double drho,dr,cut;
    double *mass;
    double **frho,***rhor,***z2r;
  };
  Fs *fs;

  virtual void allocate();
  virtual void array2spline();
  void interpolate(int, double, double *, double **);
  void grab(FILE *, int, double *);

  virtual void read_file(char *);
  virtual void file2array();
};

}

#endif
#endif