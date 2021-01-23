#include <cutoffs_kokkos.h>
#include <cmath>
#include <iostream>
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

