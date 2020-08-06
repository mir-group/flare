/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Stan Moore, original SW (SNL)
                         Anders Johansson, modified to MGP (Harvard)
------------------------------------------------------------------------- */

#include "pair_mgp_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair_kokkos.h"
#include <algorithm>
#include <cmath>
#include <utility>

using namespace LAMMPS_NS;
using namespace MathConst;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairMGPKokkos<DeviceType>::PairMGPKokkos(LAMMPS *lmp) : PairMGP(lmp) {
  respa_enable = 0;

  atomKK = (AtomKokkos *)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read =
      X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template <class DeviceType> PairMGPKokkos<DeviceType>::~PairMGPKokkos() {
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->destroy_kokkos(k_vatom, vatom);
    eatom = NULL;
    vatom = NULL;
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void PairMGPKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL)
    no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, 6, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space, datamask_read);
  if (eflag || vflag)
    atomKK->modified(execution_space, datamask_modify);
  else
    atomKK->modified(execution_space, F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  nall = atom->nlocal + atom->nghost;

  const int inum = list->inum;
  const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType> *k_list =
      static_cast<NeighListKokkos<DeviceType> *>(list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterDuplicated>(f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterDuplicated>(d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_eatom = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<
        Kokkos::Experimental::ScatterSum,
        Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }

  copymode = 1;

  EV_FLOAT ev;
  EV_FLOAT ev_all;

  // build short neighbor list

  int max_neighs = d_neighbors.extent(1);

  if ((d_neighbors_short.extent(1) != max_neighs) ||
      (d_neighbors_short.extent(0) != ignum)) {
    d_neighbors_short = Kokkos::View<int **, DeviceType>("MGP::neighbors_short",
                                                         ignum, max_neighs);
  }
  if (d_numneigh_short.extent(0) != ignum)
    d_numneigh_short =
        Kokkos::View<int *, DeviceType>("MGP::numneighs_short", ignum);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType, TagPairMGPComputeShortNeigh>(
          0, neighflag == FULL ? ignum : inum),
      *this);

  // loop over neighbor list of my atoms

  if (neighflag == HALF) {
    if (evflag)
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairMGPComputeHalf<HALF, 1>>(0,
                                                                          inum),
          *this, ev);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPairMGPComputeHalf<HALF, 0>>(0,
                                                                          inum),
          *this);
    ev_all += ev;
  } else if (neighflag == HALFTHREAD) {
    if (evflag)
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairMGPComputeHalf<HALFTHREAD, 1>>(
              0, inum),
          *this, ev);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPairMGPComputeHalf<HALFTHREAD, 0>>(
              0, inum),
          *this);
    ev_all += ev;
  } else if (neighflag == FULL) {
    if (evflag)
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairMGPComputeFullA<FULL, 1>>(
              0, inum),
          *this, ev);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPairMGPComputeFullA<FULL, 0>>(
              0, inum),
          *this);
    ev_all += ev;

    /*
    if (evflag)
      Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairMGPComputeFullB<FULL, 1>>(
              0, ignum),
          *this, ev);
    else
      Kokkos::parallel_for(
          Kokkos::RangePolicy<DeviceType, TagPairMGPComputeFullB<FULL, 0>>(
              0, ignum),
          *this);
    ev_all += ev;
    */
  }

  if (need_dup)
    Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global)
    eng_vdwl += ev_all.evdwl;
  if (vflag_global) {
    virial[0] += ev_all.v[0];
    virial[1] += ev_all.v[1];
    virial[2] += ev_all.v[2];
    virial[3] += ev_all.v[3];
    virial[4] += ev_all.v[4];
    virial[5] += ev_all.v[5];
  }

  if (eflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr)
    pair_virial_fdotr_compute(this);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f = decltype(dup_f)();
    dup_eatom = decltype(dup_eatom)();
    dup_vatom = decltype(dup_vatom)();
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::
operator()(TagPairMGPComputeShortNeigh, const int &ii) const {
  const int i = d_ilist[ii];
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  const int jnum = d_numneigh[i];
  int inside = 0;
  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i, jj);
    j &= NEIGHMASK;

    const X_FLOAT delx = xtmp - x(j, 0);
    const X_FLOAT dely = ytmp - x(j, 1);
    const X_FLOAT delz = ztmp - x(j, 2);
    const F_FLOAT rsq = delx * delx + dely * dely + delz * delz;

    if (rsq < cutmax * cutmax) {
      d_neighbors_short(i, inside) = j;
      inside++;
    }
  }
  d_numneigh_short(i) = inside;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::
operator()(TagPairMGPComputeHalf<NEIGHFLAG, EVFLAG>, const int &ii,
           EV_FLOAT &ev) const {

  // The f array is duplicated for OpenMP, atomic for CUDA, and neither for
  // Serial

  auto v_f =
      ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value, decltype(dup_f),
                        decltype(ndup_f)>::get(dup_f, ndup_f);
  auto a_f = v_f.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  F_FLOAT delr1[3], delr2[3], delr12[3], fj[3], fk[3], ftriplet[3];
  F_FLOAT evdwl = 0.0;
  F_FLOAT fpair = 0.0;

  const int i = d_ilist[ii];
  const tagint itag = tag[i];
  const int itype = type[i];
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  // two-body interactions, skip half of them

  const int jnum = d_numneigh_short[i];

  F_FLOAT fxtmpi = 0.0;
  F_FLOAT fytmpi = 0.0;
  F_FLOAT fztmpi = 0.0;

  if (compute2b) {
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors_short(i, jj);
      j &= NEIGHMASK;
      const tagint jtag = tag[j];

      if (itag > jtag) {
        if ((itag + jtag) % 2 == 0)
          continue;
      } else if (itag < jtag) {
        if ((itag + jtag) % 2 == 1)
          continue;
      } else {
        if (x(j, 2) < ztmp)
          continue;
        if (x(j, 2) == ztmp && x(j, 1) < ytmp)
          continue;
        if (x(j, 2) == ztmp && x(j, 1) == ytmp && x(j, 0) < xtmp)
          continue;
      }

      const int jtype = type[j];

      const X_FLOAT delx = xtmp - x(j, 0);
      const X_FLOAT dely = ytmp - x(j, 1);
      const X_FLOAT delz = ztmp - x(j, 2);
      const F_FLOAT rsq = delx * delx + dely * dely + delz * delz;

      const int mapid = d_map2b(itype, jtype);
      if (rsq >= d_cut2bsq(mapid))
        continue;

      twobody(mapid, rsq, fpair, evdwl);

      fxtmpi += delx * fpair;
      fytmpi += dely * fpair;
      fztmpi += delz * fpair;
      a_f(j, 0) -= delx * fpair;
      a_f(j, 1) -= dely * fpair;
      a_f(j, 2) -= delz * fpair;

      if (EVFLAG) {
        if (eflag)
          ev.evdwl += 2 * evdwl;
        if (vflag_either || eflag_atom)
          this->template ev_tally<NEIGHFLAG>(ev, i, j, evdwl, fpair, delx, dely,
                                             delz);
      }
    }
  }

  if (compute3b) {
    const int jnumm1 = jnum - 1;

    for (int jj = 0; jj < jnumm1; jj++) {
      int j = d_neighbors_short(i, jj);
      j &= NEIGHMASK;
      const int jtype = type[j];
      // const int ijparam = d_elem2param(itype, jtype, jtype);
      delr1[0] = x(j, 0) - xtmp;
      delr1[1] = x(j, 1) - ytmp;
      delr1[2] = x(j, 2) - ztmp;
      const F_FLOAT rsq1 =
          delr1[0] * delr1[0] + delr1[1] * delr1[1] + delr1[2] * delr1[2];
      // if (rsq1 >= d_cut2bsq(d_map2b(itype, jtype)))
      // continue;

      F_FLOAT fxtmpj = 0.0;
      F_FLOAT fytmpj = 0.0;
      F_FLOAT fztmpj = 0.0;

      for (int kk = jj + 1; kk < jnum; kk++) {
        int k = d_neighbors_short(i, kk);
        k &= NEIGHMASK;
        const int ktype = type[k];
        const int mapid1 = d_map3b(itype, jtype, ktype),
                  mapid2 = d_map3b(itype, ktype, jtype);

        const F_FLOAT cutoff = d_cut3bsq[mapid1];

        if (rsq1 >= cutoff)
          continue;

        delr2[0] = x(k, 0) - xtmp;
        delr2[1] = x(k, 1) - ytmp;
        delr2[2] = x(k, 2) - ztmp;
        const F_FLOAT rsq2 =
            delr2[0] * delr2[0] + delr2[1] * delr2[1] + delr2[2] * delr2[2];

        if (rsq2 >= cutoff)
          continue;

        delr12[0] = x(k, 0) - x(j, 0);
        delr12[1] = x(k, 1) - x(j, 1);
        delr12[2] = x(k, 2) - x(j, 2);
        const F_FLOAT rsq12 = delr12[0] * delr12[0] + delr12[1] * delr12[1] +
                              delr12[2] * delr12[2];
        if (rsq12 >= cutoff)
          continue;

        const F_FLOAT r1 = sqrt(rsq1), r2 = sqrt(rsq2), r12 = sqrt(rsq12);

        F_FLOAT evdwl3;
        threebody(mapid1, r1, r2, r12, evdwl3, ftriplet);

        F_FLOAT f_d1, f_d2;

        // I don't understand the 1.5
        f_d1 = -0.5 * ftriplet[0] / r1; // divided by length
        f_d2 = -0.5 * ftriplet[1] / r2; // divided by length

        fj[0] = f_d1 * delr1[0]; // delr1, delr2, not unit vector
        fj[1] = f_d1 * delr1[1];
        fj[2] = f_d1 * delr1[2];
        fk[0] = f_d2 * delr2[0];
        fk[1] = f_d2 * delr2[1];
        fk[2] = f_d2 * delr2[2];

        fxtmpi -= 3 * (fj[0] + fk[0]);
        fytmpi -= 3 * (fj[1] + fk[1]);
        fztmpi -= 3 * (fj[2] + fk[2]);

        fxtmpj += 3 * fj[0];
        fytmpj += 3 * fj[1];
        fztmpj += 3 * fj[2];

        a_f(k, 0) += 3 * fk[0];
        a_f(k, 1) += 3 * fk[1];
        a_f(k, 2) += 3 * fk[2];

        if (EVFLAG) {
          if (eflag)
            ev.evdwl += evdwl3;
          if (vflag_either || eflag_atom)
            this->template ev_tally3<NEIGHFLAG>(ev, i, j, k, evdwl3, 0.0, fj,
                                                fk, delr1, delr2);
        }
      }

      a_f(j, 0) += fxtmpj;
      a_f(j, 1) += fytmpj;
      a_f(j, 2) += fztmpj;
    }
  }

  a_f(i, 0) += fxtmpi;
  a_f(i, 1) += fytmpi;
  a_f(i, 2) += fztmpi;
}

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::
operator()(TagPairMGPComputeHalf<NEIGHFLAG, EVFLAG>, const int &ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG, EVFLAG>(
      TagPairMGPComputeHalf<NEIGHFLAG, EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::
operator()(TagPairMGPComputeFullA<NEIGHFLAG, EVFLAG>, const int &ii,
           EV_FLOAT &ev) const {

  F_FLOAT delr1[3], delr2[3], delr12[3], fj[3], fk[3], ftriplet[3];
  F_FLOAT evdwl = 0.0;
  F_FLOAT fpair = 0.0;

  const int i = d_ilist[ii];

  const tagint itag = tag[i];
  const int itype = type[i];
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  // two-body interactions

  const int jnum = d_numneigh_short[i];

  F_FLOAT fxtmpi = 0.0;
  F_FLOAT fytmpi = 0.0;
  F_FLOAT fztmpi = 0.0;

  if (compute2b) {
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors_short(i, jj);
      j &= NEIGHMASK;
      const tagint jtag = tag[j];

      const int jtype = type[j];

      const X_FLOAT delx = xtmp - x(j, 0);
      const X_FLOAT dely = ytmp - x(j, 1);
      const X_FLOAT delz = ztmp - x(j, 2);
      const F_FLOAT rsq = delx * delx + dely * dely + delz * delz;

      const int mapid = d_map2b(itype, jtype);
      if (rsq >= d_cut2bsq(mapid))
        continue;

      twobody(mapid, rsq, fpair, evdwl);

      fxtmpi += delx * fpair;
      fytmpi += dely * fpair;
      fztmpi += delz * fpair;

      if (EVFLAG) {
        if (eflag)
          ev.evdwl += evdwl;
        if (vflag_either || eflag_atom)
          this->template ev_tally<NEIGHFLAG>(ev, i, j, evdwl, fpair, delx, dely,
                                             delz);
      }
    }
  }

  if (compute3b) {
    const int jnumm1 = jnum - 1;

    for (int jj = 0; jj < jnumm1; jj++) {
      int j = d_neighbors_short(i, jj);
      j &= NEIGHMASK;
      const int jtype = type[j];
      // const int ijparam = d_elem2param(itype, jtype, jtype);
      delr1[0] = x(j, 0) - xtmp;
      delr1[1] = x(j, 1) - ytmp;
      delr1[2] = x(j, 2) - ztmp;
      const F_FLOAT rsq1 =
          delr1[0] * delr1[0] + delr1[1] * delr1[1] + delr1[2] * delr1[2];

      // if (rsq1 >= d_cut2bsq(d_map2b(itype, jtype)))
      // continue;

      for (int kk = jj + 1; kk < jnum; kk++) {
        int k = d_neighbors_short(i, kk);
        k &= NEIGHMASK;
        const int ktype = type[k];
        const int mapid1 = d_map3b(itype, jtype, ktype),
                  mapid2 = d_map3b(itype, ktype, jtype);
        // const int ikparam = d_elem2param(itype, ktype, ktype);
        // const int ijkparam = d_elem2param(itype, jtype, ktype);

        const F_FLOAT cutoff = d_cut3bsq[mapid1];

        if (rsq1 >= cutoff)
          continue;

        delr2[0] = x(k, 0) - xtmp;
        delr2[1] = x(k, 1) - ytmp;
        delr2[2] = x(k, 2) - ztmp;
        const F_FLOAT rsq2 =
            delr2[0] * delr2[0] + delr2[1] * delr2[1] + delr2[2] * delr2[2];

        if (rsq2 >= cutoff)
          continue;

        delr12[0] = x(k, 0) - x(j, 0);
        delr12[1] = x(k, 1) - x(j, 1);
        delr12[2] = x(k, 2) - x(j, 2);
        const F_FLOAT rsq12 = delr12[0] * delr12[0] + delr12[1] * delr12[1] +
                              delr12[2] * delr12[2];
        if (rsq12 >= cutoff)
          continue;

        const F_FLOAT r1 = sqrt(rsq1), r2 = sqrt(rsq2), r12 = sqrt(rsq12);

        F_FLOAT evdwl3;
        threebody(mapid1, r1, r2, r12, evdwl3, ftriplet);

        // threebody(d_params[ijparam], d_params[ikparam], d_params[ijkparam],
        //           rsq1, rsq2, delr1, delr2, fj, fk, eflag, evdwl);

        F_FLOAT f_d1, f_d2;

        f_d1 = -1.5 * ftriplet[0] / r1; // divided by length
        f_d2 = -1.5 * ftriplet[1] / r2; // divided by length

        fj[0] = f_d1 * delr1[0]; // delr1, delr2, not unit vector
        fj[1] = f_d1 * delr1[1];
        fj[2] = f_d1 * delr1[2];
        fk[0] = f_d2 * delr2[0];
        fk[1] = f_d2 * delr2[1];
        fk[2] = f_d2 * delr2[2];

        fxtmpi -= 2 * (fj[0] + fk[0]);
        fytmpi -= 2 * (fj[1] + fk[1]);
        fztmpi -= 2 * (fj[2] + fk[2]);

        if (EVFLAG) {
          if (eflag) {
            ev.evdwl += evdwl3;
          }
          if (vflag_either || eflag_atom) {
            // this->template ev_tally<NEIGHFLAG>(ev, i, j, evdwl3, f_d1,
            // delr1[0], delr1[1], delr1[2]);
            // this->template ev_tally<NEIGHFLAG>(ev, i, k, evdwl3, f_d2,
            // delr2[0], delr2[1], delr2[2]);
            this->template ev_tally3<NEIGHFLAG>(ev, i, j, k, evdwl3, 0.0, fj,
                                                fk, delr1, delr2);
            // TODO: check this, probably a factor off
            // ev_tally3_atom(ev, i, evdwl3, 0.0, fj, fk, delr1, delr2);
          }
        }
      }
    }
  }

  f(i, 0) += fxtmpi;
  f(i, 1) += fytmpi;
  f(i, 2) += fztmpi;
}

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::
operator()(TagPairMGPComputeFullA<NEIGHFLAG, EVFLAG>, const int &ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG, EVFLAG>(
      TagPairMGPComputeFullA<NEIGHFLAG, EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::
operator()(TagPairMGPComputeFullB<NEIGHFLAG, EVFLAG>, const int &ii,
           EV_FLOAT &ev) const {

  return; // DON'T THINK THIS IS NEEDED
  /*

  F_FLOAT delr1[3], delr2[3], fj[3], fk[3];
  F_FLOAT evdwl = 0.0;

  const int i = d_ilist[ii];

  const int itype = type[i];
  const X_FLOAT xtmpi = x(i, 0);
  const X_FLOAT ytmpi = x(i, 1);
  const X_FLOAT ztmpi = x(i, 2);

  const int jnum = d_numneigh_short[i];

  F_FLOAT fxtmpi = 0.0;
  F_FLOAT fytmpi = 0.0;
  F_FLOAT fztmpi = 0.0;

  if (compute3b) {
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors_short(i, jj);
      j &= NEIGHMASK;
      if (j >= nlocal)
        continue;
      const int jtype = type[j];
      const int jiparam = d_elem2param(jtype, itype, itype);
      const X_FLOAT xtmpj = x(j, 0);
      const X_FLOAT ytmpj = x(j, 1);
      const X_FLOAT ztmpj = x(j, 2);

      delr1[0] = xtmpi - xtmpj;
      delr1[1] = ytmpi - ytmpj;
      delr1[2] = ztmpi - ztmpj;
      const F_FLOAT rsq1 =
          delr1[0] * delr1[0] + delr1[1] * delr1[1] + delr1[2] * delr1[2];

      if (rsq1 >= d_params[jiparam].cutsq)
        continue;

      const int j_jnum = d_numneigh_short[j];

      for (int kk = 0; kk < j_jnum; kk++) {
        int k = d_neighbors_short(j, kk);
        k &= NEIGHMASK;
        if (k == i)
          continue;
        const int ktype = type[k];
        const int jkparam = d_elem2param(jtype, ktype, ktype);
        const int jikparam = d_elem2param(jtype, itype, ktype);

        delr2[0] = x(k, 0) - xtmpj;
        delr2[1] = x(k, 1) - ytmpj;
        delr2[2] = x(k, 2) - ztmpj;
        const F_FLOAT rsq2 =
            delr2[0] * delr2[0] + delr2[1] * delr2[1] + delr2[2] * delr2[2];

        if (rsq2 >= d_params[jkparam].cutsq)
          continue;

        if (vflag_atom)
          threebody(d_params[jiparam], d_params[jkparam], d_params[jikparam],
                    rsq1, rsq2, delr1, delr2, fj, fk, eflag, evdwl);
        else
          threebodyj(d_params[jiparam], d_params[jkparam], d_params[jikparam],
                     rsq1, rsq2, delr1, delr2, fj);

        fxtmpi += fj[0];
        fytmpi += fj[1];
        fztmpi += fj[2];

        if (EVFLAG)
          if (vflag_atom || eflag_atom)
            ev_tally3_atom(ev, i, evdwl, 0.0, fj, fk, delr1, delr2);
      }
    }
  }

  f(i, 0) += fxtmpi;
  f(i, 1) += fytmpi;
  f(i, 2) += fztmpi;
  */
}

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::
operator()(TagPairMGPComputeFullB<NEIGHFLAG, EVFLAG>, const int &ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG, EVFLAG>(
      TagPairMGPComputeFullB<NEIGHFLAG, EVFLAG>(), ii, ev);
}

template <class DeviceType>
template <typename T, typename V>
void PairMGPKokkos<DeviceType>::copy_1d(V &d, T *h, int n) {
  // Kokkos::realloc(d, n);
  // auto h_view = Kokkos::create_mirror(d);
  Kokkos::DualView<T *, DeviceType> tmp("pair::tmp", n);
  auto h_view = tmp.h_view;

  for (int i = 0; i < n; i++) {
    h_view(i) = h[i];
  }

  // Kokkos::deep_copy(d, h_view);

  tmp.template modify<LMPHostType>();
  tmp.template sync<DeviceType>();

  d = tmp.template view<DeviceType>();
}

template <class DeviceType>
template <typename T, typename V>
void PairMGPKokkos<DeviceType>::copy_2d(V &d, T **h, int m, int n) {
  Kokkos::View<T **, Kokkos::LayoutRight> tmp("pair::tmp", m, n);
  auto h_view = Kokkos::create_mirror(tmp);
  // typename Kokkos::DualView<T **, Kokkos::LayoutRight, DeviceType> tmp(
  //    "pair::tmp", m, n);
  // auto h_view = tmp.h_view;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      h_view(i, j) = h[i][j];
    }
  }

  Kokkos::deep_copy(tmp, h_view);

  d = tmp;

  // tmp.template modify<LMPHostType>();
  // tmp.template sync<DeviceType>();

  // d = tmp.template view<DeviceType>();
}

template <class DeviceType>
template <typename T, typename V>
void PairMGPKokkos<DeviceType>::copy_3d(V &d, T ***h, int m, int n, int o) {
  Kokkos::View<T ***> tmp("pair::tmp", m, n, o);
  auto h_view = Kokkos::create_mirror(tmp);
  // Kokkos::DualView<T ***, DeviceType> tmp("pair::tmp", m, n, o);
  // auto h_view = tmp.h_view;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < o; k++) {
        h_view(i, j, k) = h[i][j][k];
      }
    }
  }

  Kokkos::deep_copy(tmp, h_view);

  d = tmp;

  // tmp.template modify<LMPHostType>();
  // tmp.template sync<DeviceType>();

  // d = tmp.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template <class DeviceType>
void PairMGPKokkos<DeviceType>::coeff(int narg, char **arg) {
  // let CPU implementation read coefficients and set everything up
  PairMGP::coeff(narg, arg);

  // copy all coefficients etc. to device

  int n = atom->ntypes;

  if (compute2b) {
    copy_1d(d_cut2bsq, cut2bsq, n_2body);
    copy_1d(d_lo_2body, lo_2body, n_2body);
    copy_1d<LMP_FLOAT>(d_hi_2body, hi_2body, n_2body);
    copy_1d(d_grid_2body, grid_2body, n_2body);
    copy_2d(d_map2b, map2b, n + 1, n + 1);
  }

  if (compute3b) {
    copy_1d(d_cut3bsq, cut3bsq, n_3body);
    copy_3d<int>(d_map3b, map3b, n + 1, n + 1, n + 1);

    copy_2d<LMP_FLOAT>(d_lo_3body, lo_3body, n_3body, 3);
    copy_2d<LMP_FLOAT>(d_hi_3body, hi_3body, n_3body, 3);

    copy_2d(d_grid_3body, grid_3body, n_3body, 3);
  }

  copy_1d(d_Bd, Bd, 4);
  copy_1d(d_Cd, Cd, 4);
  copy_1d(d_basis, basis, 4);

  Kokkos::View<LMP_FLOAT **, Kokkos::LayoutRight> k_Ad("Ad", 4, 4),
      k_dAd("dAd", 4, 4), k_d2Ad("d2Ad", 4, 4);
  auto h_Ad = Kokkos::create_mirror(k_Ad), h_dAd = Kokkos::create_mirror(k_dAd),
       h_d2Ad = Kokkos::create_mirror(k_d2Ad);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      h_Ad(i, j) = Ad[i][j];
      h_dAd(i, j) = dAd[i][j];
      h_d2Ad(i, j) = d2Ad[i][j];
    }
  }

  Kokkos::deep_copy(k_Ad, h_Ad);
  Kokkos::deep_copy(k_dAd, h_dAd);
  Kokkos::deep_copy(k_d2Ad, h_d2Ad);
  d_Ad = k_Ad;
  d_dAd = k_dAd;
  d_d2Ad = k_d2Ad;

  // copy fcoeff_(2|3)body. these are ragged, not rectangular,
  // but kokkos views should be rectangular. I choose to pad.

  if (compute2b) {
    int maxlen = *std::max_element(grid_2body, grid_2body + n_2body) + 2;
    Kokkos::View<LMP_FLOAT **, Kokkos::LayoutRight> k_fcoeff_2body(
        "pair::fcoeff_2body", n_2body, maxlen);
    auto h_fcoeff_2body = Kokkos::create_mirror(k_fcoeff_2body);

    for (int i = 0; i < n_2body; i++) {
      for (int j = 0; j < grid_2body[i] + 2; j++) {
        h_fcoeff_2body(i, j) = fcoeff_2body[i][j];
      }
    }
    Kokkos::deep_copy(k_fcoeff_2body, h_fcoeff_2body);
    d_fcoeff_2body = k_fcoeff_2body;
  }

  if (compute3b) {
    int maxlen = 0;
    for (int i = 0; i < n_3body; i++) {
      int len = (grid_3body[i][0] + 2) * (grid_3body[i][1] + 2) *
                (grid_3body[i][2] + 2);
      if (len > maxlen)
        maxlen = len;
    }
    Kokkos::View<LMP_FLOAT **, Kokkos::LayoutRight> k_fcoeff_3body(
        "pair::fcoeff_3body", n_3body, maxlen);
    auto h_fcoeff_3body = Kokkos::create_mirror(k_fcoeff_3body);
    for (int i = 0; i < n_3body; i++) {
      int len = (grid_3body[i][0] + 2) * (grid_3body[i][1] + 2) *
                (grid_3body[i][2] + 2);
      for (int j = 0; j < len; j++) {
        h_fcoeff_3body(i, j) = fcoeff_3body[i][j];
      }
    }
    Kokkos::deep_copy(k_fcoeff_3body, h_fcoeff_3body);
    d_fcoeff_3body = k_fcoeff_3body;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template <class DeviceType> void PairMGPKokkos<DeviceType>::init_style() {
  PairMGP::init_style();

  // irequest = neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->kokkos_host =
      Kokkos::Impl::is_same<DeviceType, LMPHostType>::value &&
      !Kokkos::Impl::is_same<DeviceType, LMPDeviceType>::value;
  neighbor->requests[irequest]->kokkos_device =
      Kokkos::Impl::is_same<DeviceType, LMPDeviceType>::value;

  // always request a full neighbor list

  if (neighflag == FULL || neighflag == HALF || neighflag == HALFTHREAD) {
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
    if (neighflag == FULL)
      neighbor->requests[irequest]->ghost = 1;
    else
      neighbor->requests[irequest]->ghost = 0;
  } else {
    error->all(FLERR, "Cannot use chosen neighbor list style with pair sw/kk");
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void PairMGPKokkos<DeviceType>::setup_params() {
  /*
  PairMGP::setup_params();

  // sync elem2param and params

  tdual_int_3d k_elem2param =
      tdual_int_3d("pair:elem2param", nelements, nelements, nelements);
  t_host_int_3d h_elem2param = k_elem2param.h_view;

  tdual_param_1d k_params = tdual_param_1d("pair:params", nparams);
  t_host_param_1d h_params = k_params.h_view;

  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++)
      for (int k = 0; k < nelements; k++)
        h_elem2param(i, j, k) = elem2param[i][j][k];

  for (int m = 0; m < nparams; m++)
    h_params[m] = params[m];

  k_elem2param.template modify<LMPHostType>();
  k_elem2param.template sync<DeviceType>();
  k_params.template modify<LMPHostType>();
  k_params.template sync<DeviceType>();

  d_elem2param = k_elem2param.template view<DeviceType>();
  d_params = k_params.template view<DeviceType>();
  */
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
PairMGPKokkos<DeviceType>::twobody(const int &mapid, const F_FLOAT &rsq,
                                   F_FLOAT &force, F_FLOAT &energy) const {
  LMP_FLOAT r = sqrt(rsq);

  LMP_FLOAT a = d_lo_2body(mapid), b = d_hi_2body(mapid);
  int orders = d_grid_2body(mapid);
  auto coefs = Kokkos::subview(d_fcoeff_2body, mapid, Kokkos::ALL);

  // printf("proc = %d, mapid = %d, a = %g, b = %g, orders = %d\n", comm->me,
  //       mapid, a, b, orders);
  // for (int i = 0; i < orders + 2; i++) {
  //  printf("%.2f, ", coefs(i));
  //}
  // printf("\n");

  force = 0.0;
  energy = 0.0;

  // some coefficients
  int i, j, ii, i0; // made i0 int
  F_FLOAT dinv, u, tt;
  dinv = (orders - 1.0) / (b - a);
  u = (r - a) * dinv;
  i0 = floor(u);
  ii = max(min(i0, orders - 2), 0); // fmax fmin -> max min
  tt = u - ii;

  // printf("i0=%d, ii=%d, dinv=%.8f, u=%.8f, tt=%.8f, ", i0, ii, dinv, u, tt);

  // interpolation points
  F_FLOAT tp[4];
  for (j = 0; j < 4; j++) {
    tp[j] = pow(tt, 3 - j);
  }

  // value of cubic spline function
  F_FLOAT Phi[4], dPhi[4];
  F_FLOAT dt;
  int k;

  if (tt < 0) {
    for (i = 0; i < 4; i++) {
      Phi[i] = d_dAd(i, 3) * tt + d_Ad(i, 3);
    }
  } else if (tt > 1) {
    dt = tt - 1;
    for (i = 0; i < 4; i++) {
      Phi[i] = d_Bd(i) * dt + d_Cd(i);
    }
  } else {
    for (i = 0; i < 4; i++) {
      Phi[i] = 0;
      for (k = 0; k < 4; k++) {
        Phi[i] += d_Ad(i, k) * tp[k];
      }
    }
  }

  // value of derivative of spline
  for (i = 0; i < 4; i++) {
    dPhi[i] = 0;
    for (k = 0; k < 4; k++) {
      dPhi[i] += d_dAd(i, k) * tp[k];
    }
    dPhi[i] *= dinv;
  }

  // added by coefficients
  int N = orders + 2;
  F_FLOAT pc = 0;
  F_FLOAT ppc = 0;

  for (j = 0; j < 4; j++) {
    energy += Phi[j] * coefs(ii + j);
    force += dPhi[j] * coefs(ii + j);
    // printf("%.8g ", dPhi[j]);
  }

  force = -2 * force / r;
  // printf("mapid = %d, a = %g, b = %g, orders = %d, r = %g, f = %g, e = %g\n",
  // mapid, a, b, orders, r, force, energy);
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::threebody(
    const int mapid, const F_FLOAT r1, const F_FLOAT r2, const F_FLOAT r12,
    F_FLOAT &energy, F_FLOAT (&force)[3]) const {
  F_FLOAT *a = &d_lo_3body(mapid, 0), *b = &d_hi_3body(mapid, 0);
  int *orders = &d_grid_3body(mapid, 0);
  F_FLOAT *coefs = &d_fcoeff_3body(mapid, 0);
  // auto a = Kokkos::subview(d_lo_3body, mapid, Kokkos::ALL),
  //      b = Kokkos::subview(d_hi_3body, mapid, Kokkos::ALL);
  // auto orders = Kokkos::subview(d_grid_3body, mapid, Kokkos::ALL);
  // auto coefs = Kokkos::subview(d_fcoeff_3body, mapid, Kokkos::ALL);

  energy = 0.0;
  for (int i = 0; i < 3; i++) {
    force[i] = 0;
  }

  const int dim = 3;
  int i;
  int j;
  F_FLOAT point[3] = {r1, r2, r12};
  F_FLOAT dinv[dim];
  F_FLOAT u[dim];
  int i0[dim];
  int ii[dim];
  F_FLOAT tt[dim];

  // coefficients
  for (i = 0; i < dim; i++) {
    dinv[i] = (orders[i] - 1.0) / (b[i] - a[i]);
    u[i] = (point[i] - a[i]) * dinv[i];
    i0[i] = floor(u[i]);
    ii[i] = max(min(i0[i], orders[i] - 2), 0);
    tt[i] = u[i] - ii[i];
  }

  // points
  F_FLOAT tp[dim][4];
  for (i = 0; i < dim; i++) {
    for (j = 0; j < 4; j++) {
      tp[i][j] = pow(tt[i], 3 - j);
    }
  }

  F_FLOAT Phi[dim][4], dPhi[dim][4];
  F_FLOAT dt;
  int k;

  for (j = 0; j < dim; j++) {

    // evaluate spline function
    if (tt[j] < 0) {
      for (i = 0; i < 4; i++) {
        Phi[j][i] = d_dAd(i, 3) * tt[j] + d_Ad(i, 3);
      }
    } else if (tt[j] > 1) {
      dt = tt[j] - 1;
      for (i = 0; i < 4; i++) {
        Phi[j][i] = d_Bd(i) * dt + d_Cd(i);
      }
    } else {
      for (i = 0; i < 4; i++) {
        Phi[j][i] = 0;
        for (k = 0; k < 4; k++) {
          Phi[j][i] += d_Ad(i, k) * tp[j][k];
        }
      }
    }

    // evaluate derivatives
    for (i = 0; i < 4; i++) {
      dPhi[j][i] = 0;
      for (k = 0; k < 4; k++) {
        dPhi[j][i] += d_dAd(i, k) * tp[j][k];
      }
      dPhi[j][i] *= dinv[j];
    }
  }

  // added by coefficients
  int N[dim];
  for (i = 0; i < dim; i++) {
    N[i] = orders[i] + 2;
  }
  F_FLOAT c, pc, ppc;
  F_FLOAT dpc, dppc1, dppc2;

  for (i = 0; i < 4; i++) {
    ppc = 0;
    dppc1 = 0;
    dppc2 = 0;
    for (j = 0; j < 4; j++) {
      pc = 0;
      dpc = 0;
      for (k = 0; k < 4; k++) {
        c = coefs[((ii[0] + i) * N[1] + ii[1] + j) * N[2] + ii[2] + k];
        pc += Phi[2][k] * c;
        dpc += dPhi[2][k] * c;
      }
      ppc += Phi[1][j] * pc;
      dppc1 += dPhi[1][j] * pc;
      dppc2 += Phi[1][j] * dpc;
    }
    energy += Phi[0][i] * ppc;
    force[0] += dPhi[0][i] * ppc;
    force[1] += Phi[0][i] * dppc1;
    force[2] += Phi[0][i] * dppc2;
  }
  // printf("r1 = %g, r2 = %g, r12 = %g, e = %g, f = %.16g %.16g %.16g\n", r1,
  // r1, a12, energy, force[0], force[1], force[2]);
}

/* ---------------------------------------------------------------------- */

/*
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::threebody(
    const Param &paramij, const Param &paramik, const Param &paramijk,
    const F_FLOAT &rsq1, const F_FLOAT &rsq2, F_FLOAT *delr1, F_FLOAT *delr2,
    F_FLOAT *fj, F_FLOAT *fk, const int &eflag, F_FLOAT &eng) const {
  F_FLOAT r1, rinvsq1, rainv1, gsrainv1, gsrainvsq1, expgsrainv1;
  F_FLOAT r2, rinvsq2, rainv2, gsrainv2, gsrainvsq2, expgsrainv2;
  F_FLOAT rinv12, cs, delcs, delcssq, facexp, facrad, frad1, frad2;
  F_FLOAT facang, facang12, csfacang, csfac1, csfac2;

  r1 = sqrt(rsq1);
  rinvsq1 = 1.0 / rsq1;
  rainv1 = 1.0 / (r1 - paramij.cut);
  gsrainv1 = paramij.sigma_gamma * rainv1;
  gsrainvsq1 = gsrainv1 * rainv1 / r1;
  expgsrainv1 = exp(gsrainv1);

  r2 = sqrt(rsq2);
  rinvsq2 = 1.0 / rsq2;
  rainv2 = 1.0 / (r2 - paramik.cut);
  gsrainv2 = paramik.sigma_gamma * rainv2;
  gsrainvsq2 = gsrainv2 * rainv2 / r2;
  expgsrainv2 = exp(gsrainv2);

  rinv12 = 1.0 / (r1 * r2);
  cs = (delr1[0] * delr2[0] + delr1[1] * delr2[1] + delr1[2] * delr2[2]) *
       rinv12;
  delcs = cs - paramijk.costheta;
  delcssq = delcs * delcs;

  facexp = expgsrainv1 * expgsrainv2;

  // facrad = sqrt(paramij.lambda_epsilon*paramik.lambda_epsilon) *
  //          facexp*delcssq;

  facrad = paramijk.lambda_epsilon * facexp * delcssq;
  frad1 = facrad * gsrainvsq1;
  frad2 = facrad * gsrainvsq2;
  facang = paramijk.lambda_epsilon2 * facexp * delcs;
  facang12 = rinv12 * facang;
  csfacang = cs * facang;
  csfac1 = rinvsq1 * csfacang;

  fj[0] = delr1[0] * (frad1 + csfac1) - delr2[0] * facang12;
  fj[1] = delr1[1] * (frad1 + csfac1) - delr2[1] * facang12;
  fj[2] = delr1[2] * (frad1 + csfac1) - delr2[2] * facang12;

  csfac2 = rinvsq2 * csfacang;

  fk[0] = delr2[0] * (frad2 + csfac2) - delr1[0] * facang12;
  fk[1] = delr2[1] * (frad2 + csfac2) - delr1[1] * facang12;
  fk[2] = delr2[2] * (frad2 + csfac2) - delr1[2] * facang12;

  if (eflag)
    eng = facrad;
}
*/

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::threebodyj(
    const Param &paramij, const Param &paramik, const Param &paramijk,
    const F_FLOAT &rsq1, const F_FLOAT &rsq2, F_FLOAT *delr1, F_FLOAT *delr2,
    F_FLOAT *fj) const {
  F_FLOAT r1, rinvsq1, rainv1, gsrainv1, gsrainvsq1, expgsrainv1;
  F_FLOAT r2, rainv2, gsrainv2, expgsrainv2;
  F_FLOAT rinv12, cs, delcs, delcssq, facexp, facrad, frad1;
  F_FLOAT facang, facang12, csfacang, csfac1;

  r1 = sqrt(rsq1);
  rinvsq1 = 1.0 / rsq1;
  rainv1 = 1.0 / (r1 - paramij.cut);
  gsrainv1 = paramij.sigma_gamma * rainv1;
  gsrainvsq1 = gsrainv1 * rainv1 / r1;
  expgsrainv1 = exp(gsrainv1);

  r2 = sqrt(rsq2);
  rainv2 = 1.0 / (r2 - paramik.cut);
  gsrainv2 = paramik.sigma_gamma * rainv2;
  expgsrainv2 = exp(gsrainv2);

  rinv12 = 1.0 / (r1 * r2);
  cs = (delr1[0] * delr2[0] + delr1[1] * delr2[1] + delr1[2] * delr2[2]) *
       rinv12;
  delcs = cs - paramijk.costheta;
  delcssq = delcs * delcs;

  facexp = expgsrainv1 * expgsrainv2;

  // facrad = sqrt(paramij.lambda_epsilon*paramik.lambda_epsilon) *
  //          facexp*delcssq;

  facrad = paramijk.lambda_epsilon * facexp * delcssq;
  frad1 = facrad * gsrainvsq1;
  facang = paramijk.lambda_epsilon2 * facexp * delcs;
  facang12 = rinv12 * facang;
  csfacang = cs * facang;
  csfac1 = rinvsq1 * csfacang;

  fj[0] = delr1[0] * (frad1 + csfac1) - delr2[0] * facang12;
  fj[1] = delr1[1] * (frad1 + csfac1) - delr2[1] * facang12;
  fj[2] = delr1[2] * (frad1 + csfac1) - delr2[2] * facang12;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION void
PairMGPKokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
                                    const F_FLOAT &epair, const F_FLOAT &fpair,
                                    const F_FLOAT &delx, const F_FLOAT &dely,
                                    const F_FLOAT &delz) const {
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for CUDA,
  // and neither for Serial

  auto v_eatom =
      ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value,
                        decltype(dup_eatom),
                        decltype(ndup_eatom)>::get(dup_eatom, ndup_eatom);
  auto a_eatom =
      v_eatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  auto v_vatom =
      ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value,
                        decltype(dup_vatom),
                        decltype(ndup_vatom)>::get(dup_vatom, ndup_vatom);
  auto a_vatom =
      v_vatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  if (eflag_atom) {
    const E_FLOAT epairhalf = 0.5 * epair;
    a_eatom[i] += epairhalf;
    if (NEIGHFLAG != FULL)
      a_eatom[j] += epairhalf;
  }

  if (VFLAG) {
    const E_FLOAT v0 = delx * delx * fpair;
    const E_FLOAT v1 = dely * dely * fpair;
    const E_FLOAT v2 = delz * delz * fpair;
    const E_FLOAT v3 = delx * dely * fpair;
    const E_FLOAT v4 = delx * delz * fpair;
    const E_FLOAT v5 = dely * delz * fpair;

    if (vflag_global) {
      if (NEIGHFLAG != FULL) {
        ev.v[0] += v0;
        ev.v[1] += v1;
        ev.v[2] += v2;
        ev.v[3] += v3;
        ev.v[4] += v4;
        ev.v[5] += v5;
      } else {
        ev.v[0] += 0.5 * v0;
        ev.v[1] += 0.5 * v1;
        ev.v[2] += 0.5 * v2;
        ev.v[3] += 0.5 * v3;
        ev.v[4] += 0.5 * v4;
        ev.v[5] += 0.5 * v5;
      }
    }

    if (vflag_atom) {
      a_vatom(i, 0) += 0.5 * v0;
      a_vatom(i, 1) += 0.5 * v1;
      a_vatom(i, 2) += 0.5 * v2;
      a_vatom(i, 3) += 0.5 * v3;
      a_vatom(i, 4) += 0.5 * v4;
      a_vatom(i, 5) += 0.5 * v5;

      if (NEIGHFLAG != FULL) {
        a_vatom(j, 0) += 0.5 * v0;
        a_vatom(j, 1) += 0.5 * v1;
        a_vatom(j, 2) += 0.5 * v2;
        a_vatom(j, 3) += 0.5 * v3;
        a_vatom(j, 4) += 0.5 * v4;
        a_vatom(j, 5) += 0.5 * v5;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   tally eng_vdwl and virial into global and per-atom accumulators
   called by SW and hbond potentials, newton_pair is always on
   virial = riFi + rjFj + rkFk = (rj-ri) Fj + (rk-ri) Fk = drji*fj + drki*fk
 ------------------------------------------------------------------------- */

template <class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::ev_tally3(
    EV_FLOAT &ev, const int &i, const int &j, int &k, const F_FLOAT &evdwl,
    const F_FLOAT &ecoul, F_FLOAT *fj, F_FLOAT *fk, F_FLOAT *drji,
    F_FLOAT *drki) const {
  F_FLOAT epairthird, v[6];

  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for CUDA,
  // and neither for Serial

  auto v_eatom =
      ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value,
                        decltype(dup_eatom),
                        decltype(ndup_eatom)>::get(dup_eatom, ndup_eatom);
  auto a_eatom =
      v_eatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  auto v_vatom =
      ScatterViewHelper<NeedDup<NEIGHFLAG, DeviceType>::value,
                        decltype(dup_vatom),
                        decltype(ndup_vatom)>::get(dup_vatom, ndup_vatom);
  auto a_vatom =
      v_vatom.template access<AtomicDup<NEIGHFLAG, DeviceType>::value>();

  if (eflag_atom) {
    epairthird = THIRD * (evdwl + ecoul);
    a_eatom[i] += epairthird;
    if (NEIGHFLAG != FULL) {
      a_eatom[j] += epairthird;
      a_eatom[k] += epairthird;
    }
  }

  if (VFLAG) {
    v[0] = drji[0] * fj[0] + drki[0] * fk[0];
    v[1] = drji[1] * fj[1] + drki[1] * fk[1];
    v[2] = drji[2] * fj[2] + drki[2] * fk[2];
    v[3] = drji[0] * fj[1] + drki[0] * fk[1];
    v[4] = drji[0] * fj[2] + drki[0] * fk[2];
    v[5] = drji[1] * fj[2] + drki[1] * fk[2];

    if (vflag_global) {
      ev.v[0] += v[0];
      ev.v[1] += v[1];
      ev.v[2] += v[2];
      ev.v[3] += v[3];
      ev.v[4] += v[4];
      ev.v[5] += v[5];
    }

    if (vflag_atom) {
      a_vatom(i, 0) += THIRD * v[0];
      a_vatom(i, 1) += THIRD * v[1];
      a_vatom(i, 2) += THIRD * v[2];
      a_vatom(i, 3) += THIRD * v[3];
      a_vatom(i, 4) += THIRD * v[4];
      a_vatom(i, 5) += THIRD * v[5];

      if (NEIGHFLAG != FULL) {
        a_vatom(j, 0) += THIRD * v[0];
        a_vatom(j, 1) += THIRD * v[1];
        a_vatom(j, 2) += THIRD * v[2];
        a_vatom(j, 3) += THIRD * v[3];
        a_vatom(j, 4) += THIRD * v[4];
        a_vatom(j, 5) += THIRD * v[5];

        a_vatom(k, 0) += THIRD * v[0];
        a_vatom(k, 1) += THIRD * v[1];
        a_vatom(k, 2) += THIRD * v[2];
        a_vatom(k, 3) += THIRD * v[3];
        a_vatom(k, 4) += THIRD * v[4];
        a_vatom(k, 5) += THIRD * v[5];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   tally eng_vdwl and virial into global and per-atom accumulators
   called by SW and hbond potentials, newton_pair is always on
   virial = riFi + rjFj + rkFk = (rj-ri) Fj + (rk-ri) Fk = drji*fj + drki*fk
 ------------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairMGPKokkos<DeviceType>::ev_tally3_atom(
    EV_FLOAT &ev, const int &i, const F_FLOAT &evdwl, const F_FLOAT &ecoul,
    F_FLOAT *fj, F_FLOAT *fk, F_FLOAT *drji, F_FLOAT *drki) const {
  F_FLOAT epairthird, v[6];

  const int VFLAG = vflag_either;

  if (eflag_atom) {
    epairthird = THIRD * (evdwl + ecoul);
    d_eatom[i] += epairthird;
  }

  if (VFLAG) {
    v[0] = drji[0] * fj[0] + drki[0] * fk[0];
    v[1] = drji[1] * fj[1] + drki[1] * fk[1];
    v[2] = drji[2] * fj[2] + drki[2] * fk[2];
    v[3] = drji[0] * fj[1] + drki[0] * fk[1];
    v[4] = drji[0] * fj[2] + drki[0] * fk[2];
    v[5] = drji[1] * fj[2] + drki[1] * fk[2];

    if (vflag_atom) {
      d_vatom(i, 0) += THIRD * v[0];
      d_vatom(i, 1) += THIRD * v[1];
      d_vatom(i, 2) += THIRD * v[2];
      d_vatom(i, 3) += THIRD * v[3];
      d_vatom(i, 4) += THIRD * v[4];
      d_vatom(i, 5) += THIRD * v[5];
    }
  }
}

namespace LAMMPS_NS {
template class PairMGPKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class PairMGPKokkos<LMPHostType>;
#endif
} // namespace LAMMPS_NS
