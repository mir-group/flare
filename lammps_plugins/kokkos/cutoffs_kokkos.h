#ifndef CUTOFFS_KOKKOS_H
#define CUTOFFS_KOKKOS_H

// Radial cutoff functions.

#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
void quadratic_cutoff_kokkos(double &rcut0, double &rcut1, double r, double rcut) {

  if (r > rcut) {
    rcut0 = 0;
    rcut1 = 0;
    return;
  }

  double rdiff = r - rcut;
  rcut0 = rdiff * rdiff;
  rcut1 = 2 * rdiff;
}


//KOKKOS_INLINE_FUNCTION
//void cos_cutoff_kokkos(double &rcut0, double &rcut1, double r, double rcut);

KOKKOS_INLINE_FUNCTION
void cos_cutoff_kokkos(double &rcut0, double &rcut1, double r, double rcut){

  const double Pi = 3.14159265358979323846;

  // Calculate the cosine cutoff function and its gradient.
  if (r > rcut) {
    rcut0 = 0;
    rcut1 = 0;
    return;
  }

  double cutoff_val = (1. / 2.) * (cos(Pi * r / rcut) + 1);
  double cutoff_derv = -Pi * sin(Pi * r / rcut) / (2 * rcut);

  rcut0 = cutoff_val;
  rcut1 = cutoff_derv;
}

#endif
