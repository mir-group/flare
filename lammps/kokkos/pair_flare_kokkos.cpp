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
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include <cmath>
#include "kokkos.h"
#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "error.h"
#include "atom_masks.h"
#include "math_const.h"

#include <radial_kokkos.h>
#include <y_grad_kokkos.h>
#include <lammps_descriptor_kokkos.h>
#include <pair_flare_kokkos.h>

using namespace LAMMPS_NS;
using namespace MathConst;
namespace Kokkos {
  template <>
  struct reduction_identity<t_scalar3<double>> {
    KOKKOS_FORCEINLINE_FUNCTION static t_scalar3<double> sum() {
      return t_scalar3<double>();
    }
  };
}

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairFLAREKokkos<DeviceType>::PairFLAREKokkos(LAMMPS *lmp) : PairFLARE(lmp)
{
  respa_enable = 0;


  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<class DeviceType>
PairFLAREKokkos<DeviceType>::~PairFLAREKokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    eatom = NULL;
    vatom = NULL;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairFLAREKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  nall = atom->nlocal + atom->nghost;

  const int inum = list->inum;
  const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }

  copymode = 1;

  EV_FLOAT ev;
  EV_FLOAT ev_all;

  // build short neighbor list

  max_neighs = d_neighbors.extent(1);
  // TODO: check inum/ignum here
  int n_atoms = neighflag == FULL ? inum : inum;

  if ((d_neighbors_short.extent(1) != max_neighs) ||
     (d_neighbors_short.extent(0) != n_atoms)) {
    d_neighbors_short = Kokkos::View<int**,DeviceType>("FLARE::neighbors_short",n_atoms,max_neighs);
  }
  if (d_numneigh_short.extent(0)!=n_atoms)
    d_numneigh_short = Kokkos::View<int*,DeviceType>("FLARE::numneighs_short",n_atoms);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,n_atoms), *this);

  {
    int gsize = ScratchView3D::shmem_size(max_neighs, n_max, 4);

    int Ysize = ScratchView3D::shmem_size(max_neighs, n_harmonics, 4);

    int single_bond_size = ScratchView1D::shmem_size(n_bond);
    int single_bond_grad_size = ScratchView3D::shmem_size(n_bond, max_neighs, 3);
    int B2_size = ScratchView1D::shmem_size(n_descriptors);
    int B2_grad_size = ScratchView3D::shmem_size(n_descriptors, max_neighs, 3);

    int force_size = ScratchView2D::shmem_size(max_neighs,3);

    //fscatter = ScatterFType(f);


//  _                           _                                       _
// | |    __ _ _   _ _ __   ___| |__     ___ ___  _ __ ___  _ __  _   _| |_ ___
// | |   / _` | | | | '_ \ / __| '_ \   / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
// | |__| (_| | |_| | | | | (__| | | | | (_| (_) | | | | | | |_) | |_| | ||  __/
// |_____\__,_|\__,_|_| |_|\___|_| |_|  \___\___/|_| |_| |_| .__/ \__,_|\__\___|
//                                                         |_|
    // TODO: Check team size for CUDA, maybe figure out how it works
    auto policy = Kokkos::TeamPolicy<DeviceType>(n_atoms, Kokkos::AUTO(), 8).set_scratch_size(
        1, Kokkos::PerTeam(gsize + Ysize + single_bond_size + single_bond_grad_size
                           + 2*B2_size + B2_grad_size + force_size));
    // compute forces and energy
    Kokkos::parallel_reduce(policy, *this, ev);
    Kokkos::fence();
    atomKK->modified(execution_space,F_MASK);
    atomKK->sync(Host,F_MASK);
    /*
    atomKK->k_f.template modify<DeviceType>();
    atomKK->k_f.template sync<LMPHostType>();
    atomKK->modified(execution_space,ALL_MASK);
    */
    //Kokkos::Experimental::contribute(f, fscatter);
  }
  if (evflag)
    ev_all += ev;


  if (need_dup)
    //Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev_all.evdwl;
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

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f            = decltype(dup_f)();
    dup_eatom        = decltype(dup_eatom)();
    dup_vatom        = decltype(dup_vatom)();
  }
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(const int& ii) const {
    const int i = d_ilist[ii];
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);

    const int jnum = d_numneigh[i];
    int inside = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;

      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutoff*cutoff) {
        d_neighbors_short(i,inside) = j;
        inside++;
      }
    }
    d_numneigh_short(i) = inside;
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(typename Kokkos::TeamPolicy<DeviceType>::member_type team_member, EV_FLOAT& ev) const {

  int ii = team_member.league_rank();

  printf("Hello from thread %d of %d on team %d of %d\n", team_member.team_rank(), team_member.team_size(),
                                                          team_member.league_rank(), team_member.league_size());

  F_FLOAT delr1[3],delr2[3],fj[3],fk[3];
  const int i = d_ilist[ii];

  const int itype = type[i] - 1;
  const tagint itag = tag[i];
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);


  // two-body interactions

  const int jnum = d_numneigh_short[i];

  ScratchView3D g(team_member.team_scratch(1), max_neighs, n_max, 4);
  ScratchView3D Y(team_member.team_scratch(1), max_neighs, n_harmonics, 4);

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, jnum), [&] (int& jj){
    int j = d_neighbors_short(i,jj);
    j &= NEIGHMASK;

    const X_FLOAT delx = x(j,0) - xtmp;
    const X_FLOAT dely = x(j,1) - ytmp;
    const X_FLOAT delz = x(j,2) - ztmp;
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    calculate_radial_kokkos(jj, g, delx, dely, delz, sqrt(rsq), cutoff, n_max);
    get_Y_kokkos(jj, Y, delx, dely, delz, l_max);
    /*
    printf("i = %d, j = %d, Y =", i, j);
    for(int h = 0; h < n_harmonics; h++){
      printf(" %g", Y(jj, h, 0));
    }
    printf("\n");
    */
  });
  team_member.team_barrier();

  ScratchView1D single_bond(team_member.team_scratch(1), n_bond);
  ScratchView3D single_bond_grad(team_member.team_scratch(1), n_bond, max_neighs, 3);

  single_bond_kokkos(
    i,
    team_member,
    single_bond,
    single_bond_grad,
    g, Y,
    type,
    n_species, n_max, n_harmonics, d_neighbors_short, jnum, NEIGHMASK);
  team_member.team_barrier();

  ScratchView1D B2(team_member.team_scratch(1), n_descriptors);
  ScratchView3D B2_grad(team_member.team_scratch(1), n_descriptors, max_neighs, 3);

  //printf("i = %d, B2 =", i);
  B2_descriptor_kokkos(team_member, B2, B2_grad,
                   single_bond,
                   single_bond_grad, n_species,
                   n_max, l_max, jnum);
  team_member.team_barrier();
  //printf("\n");
  /*
  Kokkos::single(Kokkos::PerTeam(team_member), [&](){
    printf("i = %d, B2 =", i);
    for(int d = 0; d < n_descriptors; d++){
      printf(" %g", B2(d));
    }

    printf("\n");
    for(int jj = 0; jj < jnum; jj++){

      int j = d_neighbors_short(i,jj);
      j &= NEIGHMASK;
      printf("i = %d, j = %d, B2grad =", i, j);
      for(int d = 0; d < n_descriptors; d++){
        printf("\n%g %g %g", B2_grad(d,jj,0), B2_grad(d,jj,1), B2_grad(d,jj,2));
      }
      for(int d = 0; d < n_bond; d++){
        //printf("\n%g %g %g", single_bond_grad(d,jj,0), single_bond_grad(d,jj,1), single_bond_grad(d,jj,2));
      }
      printf("\n");
    }
    printf("\n");
  });
  */

  ScratchView1D beta_B2(team_member.team_scratch(1), n_descriptors);
  ScratchView2D partial_forces(team_member.team_scratch(1), max_neighs, 3);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_descriptors), [&] (int &i){
      double tmp = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, n_descriptors), [&](int &j, double &tmp){
          tmp += beta(itype, i, j)*B2(j);
      }, tmp);
      beta_B2(i) = tmp;
  });

  double B2_norm_squared = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int &i, double &norm2){
      norm2 += B2(i)*B2(i);
  }, B2_norm_squared);
  team_member.team_barrier();

  double evdwl = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int &i, double &evdwl){
      evdwl += B2(i)*beta_B2(i);
  }, evdwl);
  evdwl /= B2_norm_squared;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 3*jnum), [&] (int &k){
      int j = k/3;
      int c = k - 3*j;
      double tmp = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, n_descriptors), [&](int &d, double &tmp){
          tmp += B2_grad(d, j, c)*beta_B2(d);
      }, tmp);
      partial_forces(j,c) = -tmp;
  });
  team_member.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 3*jnum), [&] (int &k){
      int j = k/3;
      int c = k - 3*j;
      double tmp = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, n_descriptors), [&](int &d, double &tmp){
          tmp += B2_grad(d, j, c)*B2(d);
      }, tmp);
      Kokkos::single(Kokkos::PerThread(team_member), [&](){
          partial_forces(j,c) += evdwl*tmp;
          partial_forces(j,c) *= 2/B2_norm_squared;
      });
  });
  team_member.team_barrier();
  /*
  Kokkos::single(Kokkos::PerTeam(team_member), [&](){
      printf("i = %d, evdwl = %g\n", i, evdwl);
      printf("Fs = ");
      for(int jj = 0; jj < jnum; jj++){
        int j = d_neighbors_short(i,jj);
        j &= NEIGHMASK;
        printf("%d %g %g %g |", j, partial_forces(jj,0), partial_forces(jj,1), partial_forces(jj,2));
      }
      printf("\n");
  });
  */

  //printf("atom %d has %d neighs\n", i, jnum);

  if (eflag){
    Kokkos::single(Kokkos::PerTeam(team_member), [&](){ev.evdwl += evdwl;});
  }

  /*
    const X_FLOAT delx = xtmp - x(j,0);
    const X_FLOAT dely = ytmp - x(j,1);
    const X_FLOAT delz = ztmp - x(j,2);
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
    */

  //auto a_f = fscatter.access();
  t_scalar3<double> fsum;

  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, jnum), [&] (const int jj, t_scalar3<double> &ftmp){
    int j = d_neighbors_short(i,jj);
    j &= NEIGHMASK;

    const F_FLOAT fx = -partial_forces(jj,0);
    const F_FLOAT fy = -partial_forces(jj,1);
    const F_FLOAT fz = -partial_forces(jj,2);

    ftmp.x += fx;
    ftmp.y += fy;
    ftmp.z += fz;

    // ScatterView version that gives identical results:
    // a_f(j,0) -= fx;    // a_f(j,1) -= fy;    // a_f(j,2) -= fz;

    Kokkos::atomic_add(&f(j,0), -fx);
    Kokkos::atomic_add(&f(j,1), -fy);
    Kokkos::atomic_add(&f(j,2), -fz);

    printf("i = %d, j = %d, f = %g %g %g\n", i, j, fx, fy, fz);

    // TODO
    //if (vflag_either || eflag_atom) this->template ev_tally<FULL>(ev,i,j,evdwl,fpair,delx,dely,delz);
  }, fsum);
  team_member.team_barrier();

  Kokkos::single(Kokkos::PerTeam(team_member), [&](){
      //a_f(i,0) += fsum.x;      //a_f(i,1) += fsum.y;      //a_f(i,2) += fsum.z;
      Kokkos::atomic_add(&f(i,0), fsum.x);
      Kokkos::atomic_add(&f(i,1), fsum.y);
      Kokkos::atomic_add(&f(i,2), fsum.z);
      printf("i = %d, Fsum = %g %g %g\n", i, fsum.x, fsum.y, fsum.z);
  });
}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType>
void PairFLAREKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairFLARE::coeff(narg,arg);

  n_harmonics = (l_max+1)*(l_max+1);
  n_radial = n_species * n_max;
  n_bond = n_radial * n_harmonics;
  n_descriptors = (n_radial * (n_radial + 1) / 2) * (l_max + 1);

  beta = Kokkos::View<F_FLOAT***, Kokkos::LayoutRight, typename DeviceType::memory_space>("beta", n_species, n_descriptors, n_descriptors);
  auto beta_h = Kokkos::create_mirror_view(beta);
  for(int s = 0; s < n_species; s++){
    for(int i = 0; i < n_descriptors; i++){
      for(int j = 0; j < n_descriptors; j++){
        beta_h(s,i,j) = beta_matrices[s](i,j);
      }
    }
  }
  Kokkos::deep_copy(beta, beta_h);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairFLAREKokkos<DeviceType>::init_style()
{
  PairFLARE::init_style();

  // irequest = neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = std::is_same<DeviceType,LMPHostType>::value &&
    !std::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = std::is_same<DeviceType,LMPDeviceType>::value;

  // always request a full neighbor list

  //if (neighflag == FULL) { // TODO: figure this out
  //if (neighflag == HALF || neighflag == HALFTHREAD) { // TODO: figure this out
  if (neighflag == FULL || neighflag == HALF || neighflag == HALFTHREAD) {
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
    if (neighflag == FULL)
      neighbor->requests[irequest]->ghost = 1;
    else
      neighbor->requests[irequest]->ghost = 0;
  } else {
    error->all(FLERR,"Cannot use chosen neighbor list style with pair flare/kk");
  }
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto v_eatom = ScatterViewHelper<typename NeedDup<HALFTHREAD,DeviceType>::value,decltype(dup_eatom),decltype(ndup_eatom)>::get(dup_eatom,ndup_eatom);
  auto a_eatom = v_eatom.template access<typename AtomicDup<HALFTHREAD,DeviceType>::value>();

  auto v_vatom = ScatterViewHelper<typename NeedDup<HALFTHREAD,DeviceType>::value,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<typename AtomicDup<HALFTHREAD,DeviceType>::value>();

  if (eflag_atom) {
    const E_FLOAT epairhalf = 0.5 * epair;
    a_eatom[i] += epairhalf;
    if (NEIGHFLAG != FULL)
      a_eatom[j] += epairhalf;
  }

  if (VFLAG) {
    const E_FLOAT v0 = delx*delx*fpair;
    const E_FLOAT v1 = dely*dely*fpair;
    const E_FLOAT v2 = delz*delz*fpair;
    const E_FLOAT v3 = delx*dely*fpair;
    const E_FLOAT v4 = delx*delz*fpair;
    const E_FLOAT v5 = dely*delz*fpair;

    if (vflag_global) {
      if (NEIGHFLAG != FULL) {
        ev.v[0] += v0;
        ev.v[1] += v1;
        ev.v[2] += v2;
        ev.v[3] += v3;
        ev.v[4] += v4;
        ev.v[5] += v5;
      } else {
        ev.v[0] += 0.5*v0;
        ev.v[1] += 0.5*v1;
        ev.v[2] += 0.5*v2;
        ev.v[3] += 0.5*v3;
        ev.v[4] += 0.5*v4;
        ev.v[5] += 0.5*v5;
      }
    }

    if (vflag_atom) {
      a_vatom(i,0) += 0.5*v0;
      a_vatom(i,1) += 0.5*v1;
      a_vatom(i,2) += 0.5*v2;
      a_vatom(i,3) += 0.5*v3;
      a_vatom(i,4) += 0.5*v4;
      a_vatom(i,5) += 0.5*v5;

      if (NEIGHFLAG != FULL) {
        a_vatom(j,0) += 0.5*v0;
        a_vatom(j,1) += 0.5*v1;
        a_vatom(j,2) += 0.5*v2;
        a_vatom(j,3) += 0.5*v3;
        a_vatom(j,4) += 0.5*v4;
        a_vatom(j,5) += 0.5*v5;
      }
    }
  }
}



namespace LAMMPS_NS {
template class PairFLAREKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairFLAREKokkos<LMPHostType>;
#endif
}

