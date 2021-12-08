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

struct TagSingleBond{};
struct TagB2{};
struct TagBetaB2{};
struct TagNorm2{};
struct Tagw{};
struct Tagu{};
struct TagF{};
struct TagStoreF{};

namespace LAMMPS_NS {

template<class DeviceType>
class PairFLAREKokkos : public PairFLARE {
 public:
  using MemberType = typename Kokkos::TeamPolicy<DeviceType>::member_type;
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

  KOKKOS_INLINE_FUNCTION
  void operator()(TagSingleBond, const MemberType) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagB2, const int, const int) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagBetaB2, const MemberType) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagNorm2, const MemberType) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(Tagu, const int, const int, const int) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagF, const MemberType) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagStoreF, const MemberType, EV_FLOAT&) const;

  // short neigh list
  KOKKOS_INLINE_FUNCTION
  void operator()(const int&) const;

  // precompute g and Y
  KOKKOS_INLINE_FUNCTION
  void operator()(const int, const int) const;

  KOKKOS_INLINE_FUNCTION
  void v_tally(E_FLOAT (&v)[6], const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;
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

  double maxmem = 12.0e9;
  int batch_size = 0, startatom, n_batches, approx_batch_size;


  using View1D = Kokkos::View<F_FLOAT*, Kokkos::LayoutRight, DeviceType>;
  using View2D = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, DeviceType>;
  using View3D = Kokkos::View<F_FLOAT***, Kokkos::LayoutRight, DeviceType>;
  using View4D = Kokkos::View<F_FLOAT****, Kokkos::LayoutRight, DeviceType>;
  using View5D = Kokkos::View<F_FLOAT*****, Kokkos::LayoutRight, DeviceType>;
  using gYView4D = Kokkos::View<F_FLOAT****, Kokkos::LayoutStride, DeviceType>;
  using gYView4DRA = Kokkos::View<const F_FLOAT****, Kokkos::LayoutStride, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  using ScratchViewInt2D = Kokkos::View<int**, Kokkos::LayoutRight, typename DeviceType::scratch_memory_space>;
  using ScratchView1D = Kokkos::View<F_FLOAT*, Kokkos::LayoutRight, typename DeviceType::scratch_memory_space>;
  using ScratchView2D = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, typename DeviceType::scratch_memory_space>;
  using ScratchView3D = Kokkos::View<F_FLOAT***, Kokkos::LayoutRight, typename DeviceType::scratch_memory_space>;

  using ScatterFType = Kokkos::Experimental::ScatterView<F_FLOAT*[3], Kokkos::LayoutRight, typename DeviceType::memory_space>;
  ScatterFType fscatter;
  using ScatterVType = Kokkos::Experimental::ScatterView<F_FLOAT*[6], Kokkos::LayoutRight, typename DeviceType::memory_space>;
  ScatterVType vscatter;

  int need_dup;

  View1D B2_norm2s, evdwls;
  View2D B2, beta_B2, w, cutoff_matrix_k;
  View3D beta, single_bond, u, partial_forces;
  gYView4D g, Y;
  gYView4DRA g_ra, Y_ra;
  View5D single_bond_grad;

  int B2_chunk_size;

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
