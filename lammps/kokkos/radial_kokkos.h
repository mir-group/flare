#ifndef RADIAL_KOKKOS_H
#define RADIAL_KOKKOS_H

#include <functional>
#include <vector>
#include <Kokkos_Core.hpp>
#include <cutoffs_kokkos.h>


template<class ViewType>
KOKKOS_INLINE_FUNCTION
void chebyshev_kokkos(int ii, int jj, ViewType g, double r, double rcut, int N) {

  const double r1 = 0.0;
  const double r2 = rcut;

  // If r is ouside the support of the radial basis set, return.
  if ((r < r1) || (r > r2)) {
    return;
  }

  const double c = 1 / (r2 - r1);
  const double x = (r - r1) * c;

  for (int n = 0; n < N; n++) {
    if (n == 0) {
      g(ii,jj,n,0) = 1;
      g(ii,jj,n,1) = 0;
    } else if (n == 1) {
      g(ii,jj,n,0) = x;
      g(ii,jj,n,1) = c;
    }
    else {
      g(ii,jj,n,0) = 2 * x * g(ii,jj,n - 1,0) - g(ii,jj,n - 2,0);
      g(ii,jj,n,1) = 2 * g(ii,jj,n - 1,0) * c +
                        2 * x * g(ii,jj,n - 1,1) - g(ii,jj,n - 2,1);
    }
  }
}

template<class ViewType3D>
KOKKOS_INLINE_FUNCTION
void calculate_radial_kokkos(
    int ii, int jj, ViewType3D g,
    double x, double y, double z, double r, double rcut, int N) {

  // Calculate cutoff values.
  double rcut0=0.0, rcut1=0.0;
  quadratic_cutoff_kokkos(rcut0, rcut1, r, rcut);
  //cos_cutoff_kokkos(rcut0, rcut1, r, rcut);

  // Calculate radial basis values.
  chebyshev_kokkos(ii,jj, g, r, rcut, N);

  // Store the product.
  double xrel = x / r;
  double yrel = y / r;
  double zrel = z / r;

  for (int n = 0; n < N; n++) {
    double basis_val = g(ii,jj,n,0);
    double basis_deriv = g(ii,jj,n,1);
    g(ii,jj,n,0) = basis_val * rcut0;
    g(ii,jj,n,1) = basis_deriv * xrel * rcut0 +
                 basis_val * xrel * rcut1;
    g(ii,jj,n,2) = basis_deriv * yrel * rcut0 +
                  basis_val * yrel * rcut1;
    g(ii,jj,n,3) = basis_deriv * zrel * rcut0 +
                basis_val * zrel * rcut1;
  }
}

#endif
