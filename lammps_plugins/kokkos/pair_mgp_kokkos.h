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

// clang-format off
#ifdef PAIR_CLASS

PairStyle(mgp/kk,PairMGPKokkos<LMPDeviceType>)
PairStyle(mgp/kk/device,PairMGPKokkos<LMPDeviceType>)
//PairStyle(mgp/kk/host,PairMGPKokkos<LMPHostType>)

#else

#ifndef LMP_PAIR_MGP_KOKKOS_H
#define LMP_PAIR_MGP_KOKKOS_H

#include "pair_mgp.h"
#include "pair_kokkos.h"
// clang-format on

template <int NEIGHFLAG, int EVFLAG> struct TagPairMGPComputeHalf {};

template <int NEIGHFLAG, int EVFLAG> struct TagPairMGPComputeFullA {};

template <int NEIGHFLAG, int EVFLAG> struct TagPairMGPComputeFullB {};

struct TagPairMGPComputeShortNeigh {};

namespace LAMMPS_NS {

template <class DeviceType> class PairMGPKokkos : public PairMGP {
public:
  struct Param {
    double epsilon, sigma;
    double littlea, lambda, gamma, costheta;
    double biga, bigb;
    double powerp, powerq;
    double tol;
    double cut, cutsq;
    double sigma_gamma, lambda_epsilon, lambda_epsilon2;
    double c1, c2, c3, c4, c5, c6;
    int ielement, jelement, kelement;
  };
  enum { EnabledNeighFlags = FULL };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairMGPKokkos(class LAMMPS *);
  virtual ~PairMGPKokkos();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  virtual void init_style();

  template <typename T, typename V> void copy_1d(V &d, T *h, int n);

  template <typename T, typename V> void copy_2d(V &d, T **h, int m, int n);

  template <typename T, typename V>
  void copy_3d(V &d, T ***h, int m, int n, int o);

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void
  operator()(TagPairMGPComputeHalf<NEIGHFLAG, EVFLAG>, const int &,
             EV_FLOAT &) const;

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void
  operator()(TagPairMGPComputeHalf<NEIGHFLAG, EVFLAG>, const int &) const;

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void
  operator()(TagPairMGPComputeFullA<NEIGHFLAG, EVFLAG>, const int &,
             EV_FLOAT &) const;

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void
  operator()(TagPairMGPComputeFullA<NEIGHFLAG, EVFLAG>, const int &) const;

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void
  operator()(TagPairMGPComputeFullB<NEIGHFLAG, EVFLAG>, const int &,
             EV_FLOAT &) const;

  template <int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION void
  operator()(TagPairMGPComputeFullB<NEIGHFLAG, EVFLAG>, const int &) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMGPComputeShortNeigh, const int &) const;

  template <int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION void
  ev_tally(EV_FLOAT &ev, const int &i, const int &j, const KK_FLOAT &epair,
           const KK_FLOAT &fpair, const KK_FLOAT &delx, const KK_FLOAT &dely,
           const KK_FLOAT &delz) const;

  template <int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION void
  ev_tally3(EV_FLOAT &ev, const int &i, const int &j, int &k,
            const KK_FLOAT &evdwl, const KK_FLOAT &ecoul, KK_FLOAT *fj,
            KK_FLOAT *fk, KK_FLOAT *drji, KK_FLOAT *drki) const;

  KOKKOS_INLINE_FUNCTION
  void ev_tally3_atom(EV_FLOAT &ev, const int &i, const KK_FLOAT &evdwl,
                      const KK_FLOAT &ecoul, KK_FLOAT *fj, KK_FLOAT *fk,
                      KK_FLOAT *drji, KK_FLOAT *drki) const;

protected:
  typedef Kokkos::DualView<int ***, DeviceType> tdual_int_3d;
  typedef typename tdual_int_3d::t_dev_const_randomread t_int_3d_randomread;
  typedef typename tdual_int_3d::t_host t_host_int_3d;

  typedef Kokkos::DualView<int **, DeviceType> tdual_int_2d;
  typedef typename tdual_int_2d::t_dev_const_randomread t_int_2d_randomread;

  using RA = Kokkos::MemoryTraits<Kokkos::RandomAccess>;
  Kokkos::View<const int **, Kokkos::LayoutRight, RA> d_map2b;
  // t_int_2d_randomread d_map2b;
  Kokkos::View<const int ***, RA> d_map3b;

  t_int_3d_randomread d_elem2param;
  typename AT::t_int_1d_randomread d_map;

  typename AT::t_float_1d_randomread d_cut2bsq, d_cut3bsq, d_lo_2body,
      d_hi_2body;
  Kokkos::View<LMP_FLOAT **, Kokkos::LayoutRight, RA> d_lo_3body, d_hi_3body,
      d_fcoeff_2body, d_fcoeff_3body;
  typename AT::t_int_1d_randomread d_grid_2body;
  Kokkos::View<int **, Kokkos::LayoutRight, RA> d_grid_3body;

  typename AT::t_float_1d_randomread d_Bd, d_Cd, d_basis;
  Kokkos::View<LMP_FLOAT **, Kokkos::LayoutRight, RA> d_Ad, d_dAd, d_d2Ad;

  typedef Kokkos::DualView<Param *, DeviceType> tdual_param_1d;
  typedef typename tdual_param_1d::t_dev t_param_1d;
  typedef typename tdual_param_1d::t_host t_host_param_1d;

  t_param_1d d_params;

  virtual void setup_params();

  KOKKOS_INLINE_FUNCTION
  void twobody(const int &, const KK_FLOAT &, KK_FLOAT &, KK_FLOAT &) const;

  KOKKOS_INLINE_FUNCTION
  void threebody(const int mapid, const KK_FLOAT r1, const KK_FLOAT r2,
                 const KK_FLOAT r12, KK_FLOAT &energy, KK_FLOAT (&force)[3]) const;
  // void threebody(const Param &, const Param &, const Param &, const KK_FLOAT
  // &, const KK_FLOAT &, KK_FLOAT *, KK_FLOAT *, KK_FLOAT *, KK_FLOAT *, const int
  // &, KK_FLOAT &) const;

  KOKKOS_INLINE_FUNCTION
  void threebodyj(const Param &, const Param &, const Param &, const KK_FLOAT &,
                  const KK_FLOAT &, KK_FLOAT *, KK_FLOAT *, KK_FLOAT *) const;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  int need_dup;
  Kokkos::Experimental::ScatterView<
      KK_FLOAT * [3], typename DAT::t_kkacc_1d_3::array_layout, DeviceType,
      Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>
      dup_f;
  Kokkos::Experimental::ScatterView<
      KK_FLOAT *, typename DAT::t_kkacc_1d::array_layout, DeviceType,
      Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>
      dup_eatom;
  Kokkos::Experimental::ScatterView<
      KK_FLOAT * [6], typename DAT::t_kkacc_1d_6::array_layout, DeviceType,
      Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>
      dup_vatom;
  Kokkos::Experimental::ScatterView<
      KK_FLOAT * [3], typename DAT::t_kkacc_1d_3::array_layout, DeviceType,
      Kokkos::Experimental::ScatterSum,
      Kokkos::Experimental::ScatterNonDuplicated>
      ndup_f;
  Kokkos::Experimental::ScatterView<
      KK_FLOAT *, typename DAT::t_kkacc_1d::array_layout, DeviceType,
      Kokkos::Experimental::ScatterSum,
      Kokkos::Experimental::ScatterNonDuplicated>
      ndup_eatom;
  Kokkos::Experimental::ScatterView<
      KK_FLOAT * [6], typename DAT::t_kkacc_1d_6::array_layout, DeviceType,
      Kokkos::Experimental::ScatterSum,
      Kokkos::Experimental::ScatterNonDuplicated>
      ndup_vatom;

  typename AT::t_int_1d_randomread d_type2frho;
  typename AT::t_int_2d_randomread d_type2rhor;
  typename AT::t_int_2d_randomread d_type2z2r;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  // NeighListKokkos<DeviceType> k_list;

  int neighflag, newton_pair;
  int nlocal, nall, eflag, vflag;

  int inum;
  Kokkos::View<int **, DeviceType> d_neighbors_short;
  Kokkos::View<int *, DeviceType> d_numneigh_short;

  friend void pair_virial_fdotr_compute<PairMGPKokkos>(PairMGPKokkos *);
};
KOKKOS_INLINE_FUNCTION int min(int i, int j) { return i < j ? i : j; }
KOKKOS_INLINE_FUNCTION int max(int i, int j) { return i > j ? i : j; }

} // namespace LAMMPS_NS

#endif
#endif

    /* ERROR/WARNING messages:

    E: Cannot use chosen neighbor list style with pair mgp/kk

    Self-explanatory.

    */
