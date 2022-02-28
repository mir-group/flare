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
   Contributing author: Anders Johansson (Harvard)
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
#include <pair_flare_kokkos.h>

using namespace LAMMPS_NS;
using namespace MathConst;
namespace Kokkos {
  template <>
  struct reduction_identity<s_FEV_FLOAT> {
    KOKKOS_FORCEINLINE_FUNCTION static s_FEV_FLOAT sum() {
      return s_FEV_FLOAT();
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

  copymode = 1;

  EV_FLOAT ev_all;

  // build short neighbor list

  max_neighs = d_neighbors.extent(1);
  // TODO: check inum/ignum here
  int n_atoms = neighflag == FULL ? inum : inum;


#ifdef LMP_KOKKOS_GPU
  int vector_length = 32;
#define TEAM_SIZE 4
#define SINGLE_BOND_TEAM_SIZE 16
#else
  int vector_length = 8;
#define TEAM_SIZE Kokkos::AUTO()
#define SINGLE_BOND_TEAM_SIZE Kokkos::AUTO()
#endif

  // Divide the atoms into batches.
  // Goal: First batch needs to be biggest to avoid extra allocs.
  {
    double beta_mem = n_species * n_descriptors * n_descriptors * 8;
    double neigh_mem = 1.0*n_atoms * max_neighs * 4;
    double lmp_atom_mem = ignum * (18 * 8 + 4 * 4); // 2xf, v, x, virial, tag, type, mask, image
    double mem_per_atom = 8 * (
        2*n_bond // single_bond, u
        + 3*n_descriptors // B2, betaB2, w
        + 2 // evdwls, B2_norm2s
        + 0.5 // numneigh_short
        + max_neighs * (
            n_max*4 // g
            + n_harmonics*4 // Y
            + n_max*n_harmonics*3 // single_bond_grad
            + 3 // partial_forces
            + 0.5 // neighs_short
          )
        );
    size_t availmem, totalmem;
    double avail_double = maxmem - beta_mem;
    availmem = avail_double;
    approx_batch_size = std::min<int>(availmem/ mem_per_atom, n_atoms);

    if(approx_batch_size < 1) error->all(FLERR,"Not enough memory for even a single atom!");

    n_batches = std::ceil(1.0*n_atoms / approx_batch_size);
    approx_batch_size = n_atoms / n_batches;

    //printf("maxmem = %g | betamem = %g | neighmem = %g | lmp_atom_mem = %g  | mem_per_atom = %g | approx_batch_size = %d | n_batches = %d | remainder = %d\n", maxmem, beta_mem, neigh_mem, lmp_atom_mem, mem_per_atom, approx_batch_size, n_batches, n_atoms -n_batches* approx_batch_size);

  }
  int remainder = n_atoms - n_batches*approx_batch_size;

  vscatter = ScatterVType(d_vatom);
  fscatter = ScatterFType(f);


  startatom = 0;
  for(int batch_idx = 0; batch_idx < n_batches; batch_idx++){
    batch_size = approx_batch_size + (remainder-- > 0 ? 1 : 0);
    int stopatom = startatom + batch_size;
    //printf("BATCH: %d from %d to %d\n", batch_idx, startatom, stopatom);

    // reallocate per-atom views
    if (single_bond.extent(0) < batch_size){
      single_bond = View3D();
      single_bond = View3D(Kokkos::ViewAllocateWithoutInitializing("FLARE: single_bond"), batch_size, n_radial, n_harmonics);
      B2 = View2D();
      B2 = View2D(Kokkos::ViewAllocateWithoutInitializing("FLARE: B2"), batch_size, n_descriptors);
      beta_B2 = View2D();
      beta_B2 = View2D(Kokkos::ViewAllocateWithoutInitializing("FLARE: beta*B2"), batch_size, n_descriptors);
      B2_norm2s = View1D(); evdwls = View1D(); w = View2D();
      B2_norm2s = View1D(Kokkos::ViewAllocateWithoutInitializing("FLARE: B2_norm2s"), batch_size);
      evdwls = View1D(Kokkos::ViewAllocateWithoutInitializing("FLARE: evdwls"), batch_size);
      w = View2D(Kokkos::ViewAllocateWithoutInitializing("FLARE: w"), batch_size, n_descriptors);
      u = View3D();
      u = View3D(Kokkos::ViewAllocateWithoutInitializing("FLARE: u"), batch_size, n_radial, n_harmonics);

      d_numneigh_short = decltype(d_numneigh_short)();
      d_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("FLARE::numneighs_short") ,batch_size);
    }

    // reallocate per-neighbor views
    if(g.extent(0) < batch_size || g.extent(1) < max_neighs){
      Kokkos::LayoutStride glayout(batch_size, max_neighs*n_max*4,
                                   max_neighs, 1,
                                   n_max, 4*max_neighs,
                                   4, max_neighs);
      Kokkos::LayoutStride Ylayout(batch_size, max_neighs*n_harmonics*4,
                                   max_neighs, 1,
                                   n_harmonics, 4*max_neighs,
                                   4, max_neighs);
      g = gYView4D(); Y = gYView4D();
      g = gYView4D(Kokkos::ViewAllocateWithoutInitializing("FLARE: g"), glayout);
      Y = gYView4D(Kokkos::ViewAllocateWithoutInitializing("FLARE: Y"), Ylayout);
      g_ra = g;
      Y_ra = Y;

      single_bond_grad = View5D();
      single_bond_grad = View5D(Kokkos::ViewAllocateWithoutInitializing("FLARE: single_bond_grad"), batch_size, max_neighs, 3, n_max, n_harmonics);
      partial_forces = View3D();
      partial_forces = View3D(Kokkos::ViewAllocateWithoutInitializing("FLARE: partial forces"), batch_size, max_neighs, 3);

      d_neighbors_short = decltype(d_neighbors_short)();
      d_neighbors_short = Kokkos::View<int**,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("FLARE::neighbors_short") ,batch_size,max_neighs);
    }

    // compute short neighbor list
      Kokkos::parallel_for("FLARE: Short neighlist", Kokkos::RangePolicy<DeviceType>(0,batch_size), *this);

    // compute basis functions Rn and Ylm
      Kokkos::parallel_for("FLARE: R and Y",
          Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
                          {0,0}, {batch_size, max_neighs}, {1,max_neighs}),
          *this
      );

    // compute single bond and its gradient
    // dnlm, dnlmj
      int g_size = ScratchView2D::shmem_size(n_max, 4);
      int Y_size = ScratchView2D::shmem_size(n_harmonics, 4);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagSingleBond>(batch_size, SINGLE_BOND_TEAM_SIZE, vector_length).set_scratch_size(
          0, Kokkos::PerThread(g_size + Y_size));
      Kokkos::deep_copy(single_bond, 0.0);
      //Kokkos::deep_copy(single_bond_grad, 0.0);
      Kokkos::parallel_for("FLARE: single bond",
          policy,
          *this
      );

    // compute B2
    // pn1n2l = dn1lm dn2lm
      Kokkos::parallel_for("FLARE: B2",
          Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>, TagB2>(
                          {0,0}, {batch_size, n_descriptors}),
          *this
      );

    // compute beta*B2
      B2_chunk_size = std::min(1000, n_descriptors);
      int B2_size = ScratchView1D::shmem_size(B2_chunk_size);
      Kokkos::parallel_for("FLARE: beta*B2",
          Kokkos::TeamPolicy<DeviceType, TagBetaB2>(batch_size, TEAM_SIZE, vector_length).set_scratch_size(
            0, Kokkos::PerTeam(B2_size)
          ),
          *this
      );

    // compute B2 squared norms and evdwls and w
      Kokkos::parallel_for("FLARE: B2 norm2 evdwl w",
          Kokkos::TeamPolicy<DeviceType, TagNorm2>(batch_size, TEAM_SIZE, vector_length),
          *this
      );

    // compute u
    // un1lm = dn2lm(wn1n2l + wn2n1l) ~ 2*dn2lm*wn1n2l
      Kokkos::parallel_for("FLARE: u",
          Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>, Tagu>(
                          {0,0,0}, {batch_size, n_radial, n_harmonics}),
          *this
      );

    // compute partial forces
      int u_size = ScratchView2D::shmem_size(n_radial, n_harmonics);
      Kokkos::parallel_for("FLARE: partial forces",
          Kokkos::TeamPolicy<DeviceType, TagF>(batch_size, TEAM_SIZE, vector_length).set_scratch_size(
            0, Kokkos::PerTeam(u_size)
          ),
          *this
      );

    // sum and store total forces, ev_tally
      EV_FLOAT ev;
      Kokkos::parallel_reduce("FLARE: total forces, ev_tally",
          Kokkos::TeamPolicy<DeviceType, TagStoreF>(batch_size, TEAM_SIZE, vector_length),
          *this,
          ev
      );
      if (evflag)
        ev_all += ev;

    startatom = stopatom;
  }
  if (eflag_global) eng_vdwl += ev_all.evdwl;
  if (vflag_global) {
    virial[0] += ev_all.v[0];
    virial[1] += ev_all.v[1];
    virial[2] += ev_all.v[2];
    virial[3] += ev_all.v[3];
    virial[4] += ev_all.v[4];
    virial[5] += ev_all.v[5];
  }
  Kokkos::Experimental::contribute(d_vatom, vscatter);
  Kokkos::Experimental::contribute(f, fscatter);


  if (eflag_atom) {
    // if (need_dup)
    //   Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    //if (need_dup)
      //Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;

}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(const int ii, const int jj) const {

  const int i = d_ilist[ii+startatom];
  const int j = d_neighbors_short(ii,jj);
  const int jnum = d_numneigh_short(ii);
  if(jj >= jnum) return;

  const X_FLOAT delx = x(j,0) - x(i,0);
  const X_FLOAT dely = x(j,1) - x(i,1);
  const X_FLOAT delz = x(j,2) - x(i,2);
  const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

  calculate_radial_kokkos(ii, jj, g, delx, dely, delz, sqrt(rsq), cutoff_matrix_k(type[i]-1, type[j]-1), n_max);
  get_Y_kokkos(ii, jj, Y, delx, dely, delz, l_max);
  /*
  printf("i = %d, j = %d, Y =", i, j);
  for(int h = 0; h < n_harmonics; h++){
    printf(" %g", Y(jj, h, 0));
  }
  printf("\n");
  */
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagSingleBond, const MemberType team_member) const{
  int ii = team_member.league_rank();

  const int jnum = d_numneigh_short(ii);

  ScratchView2D gscratch(team_member.thread_scratch(0), 4, n_max);
  ScratchView2D Yscratch(team_member.thread_scratch(0), 4, n_harmonics);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, jnum), [&] (int jj){

      int j = d_neighbors_short(ii,jj);
      j &= NEIGHMASK;
      int s = type[j] - 1;


      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, 4*n_max), [&] (int nc){
          //int n = nc / 4;
          //int c = nc -4*n;
          int c = nc / n_max;
          int n = nc - c*n_max;
          gscratch(c, n) = g_ra(ii, jj, n, c);
      });
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, 4*n_harmonics), [&] (int lmc){
          //int lm = lmc / 4;
          //int c = lmc - 4 * lm;
          int c = lmc / n_harmonics;
          int lm = lmc - c*n_harmonics;
          Yscratch(c, lm) = Y_ra(ii, jj, lm, c);
      });

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, n_max*n_harmonics), [&] (int nlm){
          int n = nlm / n_harmonics;
          int lm = nlm - n_harmonics*n;

          int radial_index = s*n_max + n;
          double g_val = gscratch(0,n);
          double gx_val = gscratch(1,n);
          double gy_val = gscratch(2,n);
          double gz_val = gscratch(3,n);


          double h_val = Yscratch(0,lm);
          double hx_val = Yscratch(1,lm);
          double hy_val = Yscratch(2,lm);
          double hz_val = Yscratch(3,lm);

          double bond = g_val * h_val;
          double bond_x = gx_val * h_val + g_val * hx_val;
          double bond_y = gy_val * h_val + g_val * hy_val;
          double bond_z = gz_val * h_val + g_val * hz_val;

          // Update single bond basis arrays.
          Kokkos::atomic_add(&single_bond(ii, radial_index, lm),bond); // TODO: bad?

          single_bond_grad(ii,jj,0,n,lm) = bond_x;
          single_bond_grad(ii,jj,1,n,lm) = bond_y;
          single_bond_grad(ii,jj,2,n,lm) = bond_z;
      });
  });
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagB2, const int ii, const int nnl) const{
  int x = nnl/(l_max+1);
  int l = nnl-x*(l_max+1);
  double np12 = n_radial + 0.5;
  int n1 = -std::sqrt(np12*np12 - 2*x) + np12;
  int n2 = x - n1*(np12 - 1 - 0.5*n1);

  double tmp = 0.0;
  for(int m = 0; m < 2*l+1; m++){
    int lm = l*l + m;
    tmp += single_bond(ii, n1, lm) * single_bond(ii, n2, lm);
  }
  B2(ii, nnl) = tmp;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagBetaB2, const MemberType team_member) const{
  int ii = team_member.league_rank();
  const int i = d_ilist[ii+startatom];

  const int itype = type[i] - 1;

  ScratchView1D B2scratch(team_member.team_scratch(0), B2_chunk_size);

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int nnl){
      beta_B2(ii,nnl) = 0.0;
  });

  // do mat-vec product in chunks to enable level 0 scratch
  // even when descriptors are too big
  for(int starti = 0; starti < n_descriptors; starti += B2_chunk_size){
    int stopi = starti + B2_chunk_size;
    stopi = n_descriptors < stopi ? n_descriptors : stopi;
//        Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
//            if(ii==0) printf("%d %d %d\n", n_descriptors, starti, stopi);
//        });
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, stopi - starti), [&] (int nnl){
        B2scratch(nnl) = B2(ii, nnl + starti);
    });
    team_member.team_barrier();

    // TODO: team-wise GEMV?
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_descriptors), [&] (int x){
        F_FLOAT tmp = 0.0;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, stopi - starti), [&](int y, F_FLOAT &tmp){
            tmp += beta(itype, x, y+starti)*B2scratch(y);
        }, tmp);
        Kokkos::single(Kokkos::PerThread(team_member), [&] () {
            beta_B2(ii, x) += tmp;
        });
    });
    team_member.team_barrier();
  }
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagNorm2, const MemberType team_member) const{
  int ii = team_member.league_rank();
  double empty_thresh = 1e-8;

  F_FLOAT tmp = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int x, F_FLOAT &tmp){
      tmp += B2(ii, x) * B2(ii, x);
  }, tmp);
  B2_norm2s(ii) = tmp;

  tmp = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int x, F_FLOAT &tmp){
      tmp += B2(ii, x) * beta_B2(ii, x);
  }, tmp);
  evdwls(ii) = tmp/B2_norm2s(ii);

  if (d_numneigh_short(ii) == 0) {
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int x){
        w(ii, x) = 0;
    });
    evdwls(ii) = 0;
  } else {
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, n_descriptors), [&] (int x){
        w(ii, x) = 2*(evdwls(ii) * B2(ii,x) - beta_B2(ii,x))/B2_norm2s(ii);
    });
  }
  if (eflag_atom){
    const int i = d_ilist[ii+startatom];
    d_eatom[i] = evdwls(ii);
  }

}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(Tagu, const int ii, const int n1, const int lm) const{
  int l = sqrt(1.0*lm);
  //int l = Kokkos::Experimental::sqrt(lm);

  F_FLOAT un1lm = 0.0;
  for(int n2 = 0; n2 < n_radial; n2++){
    int i = n2 > n1 ? n1 : n2;
    int j = n2 > n1 ? n2 : n1;
    int n1n2 = j + i*(n_radial - 0.5*(i+1));
    int n1n2l = n1n2*(l_max+1)+l;

    un1lm += single_bond(ii, n2, lm) * w(ii, n1n2l) * (1 + (n1 == n2 ? 1 : 0));
  }
  u(ii, n1, lm) = un1lm;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagF, const MemberType team_member) const{
  int ii = team_member.league_rank();
  const int i = d_ilist[ii+startatom];
  const int jnum = d_numneigh_short(ii);

  ScratchView2D uscratch(team_member.team_scratch(0), n_radial, n_harmonics);
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, n_bond), [&] (int nlm){
      int n = nlm / n_harmonics;
      int lm = nlm - n*n_harmonics;
      uscratch(n, lm) = u(ii, n, lm);
  });
  team_member.team_barrier();

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 3*jnum), [&] (int &k){
      int jj = k/3;
      int c = k - 3*jj;

      int j = d_neighbors_short(ii,jj);
      j &= NEIGHMASK;
      int s = type[j] - 1;

      F_FLOAT tmp = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, n_max*n_harmonics), [&](int nlm, F_FLOAT &tmp){
          int n = nlm / n_harmonics;
          int lm = nlm - n*n_harmonics;
          int radial_index = s*n_max + n;
          tmp += single_bond_grad(ii, jj, c, n, lm)*uscratch(radial_index, lm);
      }, tmp);
      partial_forces(ii,jj,c) = tmp;
  });
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(TagStoreF, const MemberType team_member, EV_FLOAT &ev) const{
  int ii = team_member.league_rank();
  const int i = d_ilist[ii+startatom];
  const int jnum = d_numneigh_short(ii);
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);

  auto a_f = fscatter.access();
  s_FEV_FLOAT fvsum;

  Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team_member, jnum), [&] (const int jj, s_FEV_FLOAT &fvtmp){
      int j = d_neighbors_short(ii,jj);

      const F_FLOAT fx = -partial_forces(ii,jj,0);
      const F_FLOAT fy = -partial_forces(ii,jj,1);
      const F_FLOAT fz = -partial_forces(ii,jj,2);

      fvtmp.f[0] += fx;
      fvtmp.f[1] += fy;
      fvtmp.f[2] += fz;

      a_f(j,0) -= fx;
      a_f(j,1) -= fy;
      a_f(j,2) -= fz;

      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);

      //printf("i = %d, j = %d, f = %g %g %g\n", i, j, fx, fy, fz);

      if (vflag_either) v_tally(fvtmp.v,i,j,fx,fy,fz,delx,dely,delz);
  }, fvsum);
  team_member.team_barrier();

  Kokkos::single(Kokkos::PerTeam(team_member), [&](){
      a_f(i,0) += fvsum.f[0];
      a_f(i,1) += fvsum.f[1];
      a_f(i,2) += fvsum.f[2];
      if(eflag) ev.evdwl += evdwls(ii);
      if(vflag_global){
        ev.v[0] += fvsum.v[0];
        ev.v[1] += fvsum.v[1];
        ev.v[2] += fvsum.v[2];
        ev.v[3] += fvsum.v[3];
        ev.v[4] += fvsum.v[4];
        ev.v[5] += fvsum.v[5];
      }
      //printf("i = %d, Fsum = %g %g %g\n", i, fsum.x, fsum.y, fsum.z);
  });
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::operator()(const int& ii) const {
    const int i = d_ilist[ii+startatom];
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);

    const int si = type[i] - 1;

    const int jnum = d_numneigh[i];
    int inside = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;

      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      const double paircut = cutoff_matrix_k(si, type[j]-1);

      if (rsq < paircut*paircut) {
        d_neighbors_short(ii,inside) = j;
        inside++;
      }
    }
    d_numneigh_short(ii) = inside;
}


/* ---------------------------------------------------------------------- */




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
  beta_matrices.clear();

  cutoff_matrix_k = View2D("cutoff_matrix", n_species, n_species);
  auto cutoff_matrix_h = Kokkos::create_mirror_view(cutoff_matrix_k);

  for(int si = 0; si < n_species; si++){
    for(int sj = 0; sj < n_species; sj++){
      cutoff_matrix_h(si,sj) = cutoff_matrix(si,sj);
    }
  }
  Kokkos::deep_copy(cutoff_matrix_k, cutoff_matrix_h);
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

  if (neighflag == FULL) { // TODO: figure this out
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->ghost = 0;
  } else {
    error->all(FLERR,"Cannot use chosen neighbor list style with pair flare/kk");
  }

  // get available memory from environment variable,
  // defaults to 16 GB set in the header file
  char *memstr = std::getenv("MAXMEM");
  if (memstr != NULL) {
    maxmem = std::atof(memstr) * 1.0e9;
  }
  printf("FLARE will use up to %.2f GB of device memory, controlled by MAXMEM environment variable\n", maxmem/1.0e9);
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairFLAREKokkos<DeviceType>::v_tally(E_FLOAT (&v)[6], const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{
  const int VFLAG = vflag_either;

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto a_vatom = vscatter.access();

  if (VFLAG) {
    const E_FLOAT v0 = delx*fx;
    const E_FLOAT v1 = dely*fy;
    const E_FLOAT v2 = delz*fz;
    const E_FLOAT v3 = delx*fy;
    const E_FLOAT v4 = delx*fz;
    const E_FLOAT v5 = dely*fz;

    if (vflag_global) {
        v[0] += v0;
        v[1] += v1;
        v[2] += v2;
        v[3] += v3;
        v[4] += v4;
        v[5] += v5;
    }

    if (vflag_atom) {
      a_vatom(i,0) += 0.5*v0;
      a_vatom(i,1) += 0.5*v1;
      a_vatom(i,2) += 0.5*v2;
      a_vatom(i,3) += 0.5*v3;
      a_vatom(i,4) += 0.5*v4;
      a_vatom(i,5) += 0.5*v5;

      a_vatom(j,0) += 0.5*v0;
      a_vatom(j,1) += 0.5*v1;
      a_vatom(j,2) += 0.5*v2;
      a_vatom(j,3) += 0.5*v3;
      a_vatom(j,4) += 0.5*v4;
      a_vatom(j,5) += 0.5*v5;
    }
  }
}



namespace LAMMPS_NS {
template class PairFLAREKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairFLAREKokkos<LMPHostType>;
#endif
}

