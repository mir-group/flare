/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(flare/kk,PairFLAREKokkos<LMPDeviceType>)
PairStyle(flare/kk/device,PairFLAREKokkos<LMPDeviceType>)
PairStyle(flare/kk/host,PairFLAREKokkos<LMPHostType>)

#else

#ifndef LMP_PAIR_FLARE_KOKKOS_H
#define LMP_PAIR_FLARE_KOKKOS_H

#include "pair_flare.h"
#include <pair_kokkos.h>

namespace LAMMPS_NS {

template<class DeviceType>
class PairFLAREKokkos : public PairFLARE {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairFLAREKokkos(class LAMMPS *);
  virtual ~PairFLAREKokkos();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  virtual void init_style();

  // pair compute
  KOKKOS_INLINE_FUNCTION
  void operator()(typename Kokkos::TeamPolicy<DeviceType>::member_type, EV_FLOAT&) const;

  // short neigh list
  KOKKOS_INLINE_FUNCTION
  void operator()(const int&) const;

  // precompute g and Y
  KOKKOS_INLINE_FUNCTION
  void operator()(const int, const int) const;

  KOKKOS_INLINE_FUNCTION
  void v_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

 protected:
  typedef Kokkos::DualView<int***,DeviceType> tdual_int_3d;
  typedef typename tdual_int_3d::t_dev_const_randomread t_int_3d_randomread;
  typedef typename tdual_int_3d::t_host t_host_int_3d;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  Kokkos::View<F_FLOAT***,Kokkos::LayoutRight,DeviceType> beta;
  Kokkos::View<F_FLOAT****,Kokkos::LayoutRight,DeviceType> g, Y;

  using ScratchView1D = Kokkos::View<F_FLOAT*, Kokkos::LayoutRight, typename DeviceType::scratch_memory_space>;
  using ScratchView2D = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, typename DeviceType::scratch_memory_space>;
  using ScratchView3D = Kokkos::View<F_FLOAT***, Kokkos::LayoutRight, typename DeviceType::scratch_memory_space>;

  using ScatterFType = Kokkos::Experimental::ScatterView<F_FLOAT*[3], Kokkos::LayoutRight, typename DeviceType::memory_space>;
  ScatterFType fscatter;
  using ScatterVType = Kokkos::Experimental::ScatterView<F_FLOAT*[6], Kokkos::LayoutRight, typename DeviceType::memory_space>;
  ScatterVType vscatter;

  int need_dup;

  typename AT::t_int_1d_randomread d_type2frho;
  typename AT::t_int_2d_randomread d_type2rhor;
  typename AT::t_int_2d_randomread d_type2z2r;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  int inum, max_neighs, n_harmonics, n_radial, n_bond;
  Kokkos::View<int**,DeviceType> d_neighbors_short;
  Kokkos::View<int*,DeviceType> d_numneigh_short;


  friend void pair_virial_fdotr_compute<PairFLAREKokkos>(PairFLAREKokkos*);
};
}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use chosen neighbor list style with pair flare/kk

Self-explanatory.

*/
