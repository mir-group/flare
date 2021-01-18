/* -------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
-------------------------------------------------------------------- */

/* --------------------------------------------------------------------
   Contributing authors: Lixin Sun (Harvard) Yu Xie (Harvard)
-------------------------------------------------------------------- */

#include "pair_mgp.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ----------------------------------------------------------------- */

PairMGP::PairMGP(LAMMPS *lmp) : Pair(lmp) {
  // no restart info
  restartinfo = 0;
  // give a warning if bonds/angles/dihedrals are used with mmf
  manybody_flag = 1;
  single_enable = 1;
  // not sure this is good
  reinitflag = 0;

  setflag = NULL;
  map2b = NULL;
  map3b = NULL;
  compute2b = true;
  compute3b = true;

  elements = NULL;
  n_2body = 0;
  n_3body = 0;

  lo_2body = NULL;
  hi_2body = NULL;
  lo_3body = NULL;
  hi_3body = NULL;
  grid_2body = NULL;
  grid_3body = NULL;
  ecoeff_2body = NULL;
  fcoeff_2body = NULL;
  ecoeff_3body = NULL;
  fcoeff_3body = NULL;

  cutmax = 0.0;
  cutsq = NULL;
  cut2bsq = NULL;
  cut3bsq = NULL;
  cutshortsq = 0;

  maxshort = 20;
  neighshort = NULL;

  Ad[0][0] = -1.0 / 6.0;
  Ad[0][1] = 3.0 / 6.0;
  Ad[0][2] = -3.0 / 6.0;
  Ad[0][3] = 1.0 / 6.0;
  Ad[1][0] = 3.0 / 6.0;
  Ad[1][1] = -6.0 / 6.0;
  Ad[1][2] = 0.0 / 6.0;
  Ad[1][3] = 4.0 / 6.0;
  Ad[2][0] = -3.0 / 6.0;
  Ad[2][1] = 3.0 / 6.0;
  Ad[2][2] = 3.0 / 6.0;
  Ad[2][3] = 1.0 / 6.0;
  Ad[3][0] = 1.0 / 6.0;
  Ad[3][1] = 0.0 / 6.0;
  Ad[3][2] = 0.0 / 6.0;
  Ad[3][3] = 0.0 / 6.0;
  // {-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0},
  // { 3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0},
  // {-3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0},
  // { 1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0}

  // initialize with 0.0
  for (int j = 0; j < 4; j++) {
    Bd[j] = 0.0;
    Cd[j] = 0.0;
    basis[j] = 0.0;
    for (int i = 0; i < 4; i++) {
      dAd[j][i] = 0.0;
      d2Ad[j][i] = 0.0;
    }
  }


  // calculate values
  for (int j = 0; j < 4; j++) {
    for (int i = 1; i < 4; i++) {
      dAd[j][i] = Ad[j][i - 1] * (4 - i);
    }
  }

  for (int j = 0; j < 4; j++) {
    for (int i = 1; i < 4; i++) {
      d2Ad[j][i] = dAd[j][i - 1] * (4 - i);
    }
  }

  for (int i = 1; i < 4; i++) {
    Bd[i] = 3 * Ad[i][0] + 2 * Ad[i][1] + Ad[i][2];
    Cd[i] = Ad[i][0] + Ad[i][1] + Ad[i][2] + Ad[i][3];
  }
}

/* --------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
-------------------------------------------------------------------- */

PairMGP::~PairMGP() {
  int me;
  // why comm->me does not work here?
  MPI_Comm_rank(world, &me);

  if (copymode)
    return;

  if (elements) {
    for (int i = 1; i <= atom->ntypes; i++) {
      if (elements[i])
        delete[] elements[i];
    }
    delete[] elements;
  }

  memory->destroy(lo_2body);
  memory->destroy(hi_2body);
  memory->destroy(grid_2body);

  memory->destroy(lo_3body);
  memory->destroy(hi_3body);
  memory->destroy(grid_3body);

  // more to do
  if (fcoeff_2body) {
    for (int i = 0; i < n_2body; i++) {
      memory->destroy(fcoeff_2body[i]);
      //   memory->destroy(ecoeff_2body[i]);
    }
    memory->sfree(fcoeff_2body);
    // memory->sfree(ecoeff_2body);
  }

  if (fcoeff_3body) {
    for (int i = 0; i < n_3body; i++) {
      memory->destroy(fcoeff_3body[i]);
      //     memory->destroy(ecoeff_3body[i]);
    }
    memory->sfree(fcoeff_3body);
    // memory->sfree(ecoeff_3body);
  }

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
  }
  memory->destroy(cut2bsq);
  memory->destroy(cut3bsq);
  memory->destroy(map2b);
  memory->destroy(map3b);
}

/* ----------------------------------------------------------------- */

void PairMGP::compute(int eflag, int vflag) {
  int me;
  MPI_Comm_rank(world, &me);

  int i, j, k, m, inum, jnum, itype, jtype, ktype;
  double xtmp, ytmp, ztmp, del[3];
  double evdwl = 0;
  double rsq, r, p, rhoip, rhojp, z2, z2p, recip, phip, psip, phi;
  double cutoff;
  int *ilist, *jlist, *numneigh, **firstneigh;
  tagint *tag = atom->tag;
  tagint itag, jtag;

  // viral setting
  evdwl = 0.0;
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = eflag_global = eflag_atom = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  // int nall = nlocal + atom->nghost;
  int ntype_p1 = atom->ntypes + 1;

  // TODO: double check this. the flat should not be used.
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute 2-body forces on each atom
  // loop over neighbors of my atoms

  for (int ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    int numshort = 0;
    for (int jj = 0; jj < jnum; jj++) {
      double fpair = 0;
      j = jlist[jj];

      j &= NEIGHMASK;
      jtype = type[j];
      jtag = tag[j];

      del[0] = x[j][0] - xtmp;
      del[1] = x[j][1] - ytmp;
      del[2] = x[j][2] - ztmp;
      rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];

      if (compute3b) {
        // accumulate short list for 3body interactions
        if (rsq < cutshortsq) {
          neighshort[numshort++] = j;
          if (numshort >= maxshort) {
            maxshort += maxshort / 2;
            memory->grow(neighshort, maxshort, "pair:neighshort");
          }
        }
      }

      if (compute2b) {
        int mapid = map2b[itype][jtype];
        if (mapid != -1) {
          if (rsq < cut2bsq[mapid]) {

            jtype = type[j];
            //            if (itag > jtag) {
            //              if ((itag+jtag) % 2 == 0) continue;
            //            } else if (itag < jtag) {
            //              if ((itag+jtag) % 2 == 1) continue;
            //            } else {
            //              if (x[j][2] < ztmp) continue;
            //              if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
            //              if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] <
            //              xtmp) continue;
            //            }

            r = sqrt(rsq);

            if (r < lo_2body[mapid])
              error->all(FLERR, "MGP lower bound: The interatomic distance is smaller than the 2b lower bound.");

            // TODO: pay attention to the sign of the force

            // energy and forces are both computed
            eval_cubic_splines_1d(lo_2body[mapid], hi_2body[mapid],
                                  grid_2body[mapid], fcoeff_2body[mapid], r,
                                  &evdwl, &fpair);

            evdwl *= 2;
            fpair *= 2 / r; // check the prefactor of 2

            double fx, fy, fz;
            fx = del[0] * fpair;
            fy = del[1] * fpair;
            fz = del[2] * fpair;

            f[i][0] += fx;
            f[i][1] += fy;
            f[i][2] += fz;
            //            f[j][0] -= f0;
            //            f[j][1] -= f1;
            //            f[j][2] -= f2;

            // viral compute
            if (evflag)
              ev_tally_xyz_full(i, evdwl, 0.0, -fx, -fy, -fz, del[0], del[1],
                                del[2]);
            //            if (evflag) {
            //                ev_tally_full(i,evdwl,0.0,-fpair,del[0],del[1],del[2]);
            //                // if (screen) fprintf(screen, "MGP compute%d: 2b
            //                forces on %d from %d %lf %lf %lf distance %lf %lf
            //                %lf\n",
            //                //         me, tag[i], tag[j], del[0]*fpair,
            //                del[1]*fpair, del[2]*fpair, del[0], del[1],
            //                del[2]);
            //            }
          }
        }
      }

    } // j

    if (compute3b) {
      // three-body interactions
      // skip immediately if I-J is not within cutoff
      // double fxtmp=0, fytmp=0, fztmp=0;
      double delr1[3], delr2[3], delr12[3];
      double rsq1, rsq2, rsq12, rij, rik, rjk;
      double fij[3], fik[3], fjk[3];
      double fxtmp = 0.0, fytmp = 0.0, fztmp = 0.0;
      double fjxtmp = 0.0, fjytmp = 0.0, fjztmp = 0.0;

      for (int jj = 0; jj < numshort; jj++) {
        j = neighshort[jj];
        jtype = type[j];

        delr1[0] = x[j][0] - xtmp;
        delr1[1] = x[j][1] - ytmp;
        delr1[2] = x[j][2] - ztmp;
        rsq1 = delr1[0] * delr1[0] + delr1[1] * delr1[1] + delr1[2] * delr1[2];

        for (int kk = jj + 1; kk < numshort; kk++) {
          double ftriplet[3] = {0, 0, 0};
          k = neighshort[kk];
          ktype = type[k];

          int mapid1 = map3b[itype][jtype][ktype];
          int mapid2 = map3b[itype][ktype][jtype];
          // 3 body coef not found, then continue
          if (mapid1 == -1 || mapid2 == -1)
            error->all(FLERR, "MGP coeff: mapid is not found.");

          // TODO: double check whether the cutoffs are consistent
          cutoff = cut3bsq[mapid1];
          if (rsq1 >= cutoff)
            continue;

          delr2[0] = x[k][0] - xtmp;
          delr2[1] = x[k][1] - ytmp;
          delr2[2] = x[k][2] - ztmp;
          rsq2 =
              delr2[0] * delr2[0] + delr2[1] * delr2[1] + delr2[2] * delr2[2];
          if (rsq2 >= cutoff)
            continue;

          delr12[0] = x[k][0] - x[j][0];
          delr12[1] = x[k][1] - x[j][1];
          delr12[2] = x[k][2] - x[j][2];
          rsq12 = delr12[0] * delr12[0] + delr12[1] * delr12[1] +
                  delr12[2] * delr12[2];
          if (rsq12 >= cutoff)
            continue;

          // compute bonds
          rij = sqrt(rsq1);
          rik = sqrt(rsq2);
          rjk = sqrt(rsq12);
          if (rij < lo_3body[mapid1][0] || rik < lo_3body[mapid1][1] || rjk < lo_3body[mapid1][2])
              error->all(FLERR, "MGP lower bound: The interatomic distance is smaller than the 3b lower bound.");

          // compute spline
          eval_cubic_splines_3d(lo_3body[mapid1], hi_3body[mapid1],
                                grid_3body[mapid1], fcoeff_3body[mapid1], rij,
                                rik, rjk, &evdwl, ftriplet);

          //           double factor1, factor2, f_d1, f_d2;
          //           factor1 = 1/r2 - 1/r1 * cos_angle;
          //           factor2 = 1/r1 - 1/r2 * cos_angle;
          //
          //           f_d1 = (ftriplet[0] + ftriplet[2] * factor1) / r1; //
          //           diveded by length f_d2 = (ftriplet[1] + ftriplet[2] *
          //           factor2) / r2; // diveded by length
          double f_ij, f_ik, f_jk;

          f_ij = 3 * ftriplet[0] / rij;
          f_ik = 3 * ftriplet[1] / rik;
          //          f_jk = 3 * ftriplet[2] / rjk;

          fij[0] = f_ij * delr1[0]; // delr1, delr2, not unit vector
          fij[1] = f_ij * delr1[1];
          fij[2] = f_ij * delr1[2];
          fik[0] = f_ik * delr2[0];
          fik[1] = f_ik * delr2[1];
          fik[2] = f_ik * delr2[2];
          //          fjk[0] = f_jk * delr12[0];
          //          fjk[1] = f_jk * delr12[1];
          //          fjk[2] = f_jk * delr12[2];
          f[i][0] += (fij[0] + fik[0]);
          f[i][1] += (fij[1] + fik[1]);
          f[i][2] += (fij[2] + fik[2]);
          //          f[j][0] -= fij[0] - fjk[0];
          //          f[j][0] -= fij[1] - fjk[1];
          //          f[j][0] -= fij[2] - fjk[2];
          //          f[k][0] -= fik[0] + fjk[0];
          //          f[k][1] -= fik[1] + fjk[1];
          //          f[k][2] -= fik[2] + fjk[2];

          // calculate stress
          // TO DO, test, the fj, fk sign may be wrong
          //          if (evflag)
          //          ev_tally3(i,j,k,evdwl,0.0,fij,fik,delr1,delr2);
          if (evflag) {
//            ev_tally_full(i,evdwl,0.0,-f_ij,delr1[0],delr1[1],delr1[2]);
//            ev_tally_full(i,evdwl,0.0,-f_ik,delr2[0],delr2[1],delr2[2]);

            ev_tally_xyz_full(i, evdwl, 0.0, -fij[0], -fij[1], -fij[2],
                              delr1[0], delr1[1], delr1[2]);
            ev_tally_xyz_full(i, evdwl, 0.0, -fik[0], -fik[1], -fik[2],
                              delr2[0], delr2[1], delr2[2]);
          }

          //          if (evflag) {
          //           ev_tally_full(i,evdwl,0.0,-f_ij,delr1[0],delr1[1],delr1[2]);
          //           // check if it should be evdwl3, check prefactor 3
          //           ev_tally_full(i,evdwl,0.0,-f_ik,delr2[0],delr2[1],delr2[2]);
          // ev_tally3(i,j,k, evdwl, 0.0, fij, fik, delr1, delr2);
          // forces on j, force on k, fji, fki
          // check ev_tally3 default atomic directions!!!!!!!!! refer to
          // pair_tersoff_table: line 439 if (screen) fprintf(screen, "MGP
          // compute%d: 3b forces on %d from %d %lf %lf %lf distance %lf %lf %lf
          // \n",
          //         me, tag[i], tag[j], fj[0], fj[1], fj[2], delr1[0],
          //         delr1[1], delr1[2]);
          // if (screen) fprintf(screen, "MGP compute%d: 3b forces on %d from %d
          // %lf %lf %lf distance %lf %lf %lf \n",
          //         me, tag[i], tag[k], fk[0], fk[1], fk[2], delr2[0],
          //         delr2[1], delr2[2]);
          //          }

        } // k

      } // j
    }
  }

  // TODO, check out this function
  if (vflag_fdotr)
    virial_fdotr_compute();
}

/* --------------------------------------------------------------------
   allocate all arrays
-------------------------------------------------------------------- */

void PairMGP::allocate() {
  int me = comm->me;
  allocated = 1;

  int np = atom->ntypes + 1;
  memory->create(setflag, np, np, "pair:setflag");
  memory->create(cutsq, np, np, "pair:cutsq");
  memory->create(neighshort, maxshort, "pair:neighshort");

  memset(&setflag[0][0], 0, np * np * sizeof(int));
  memset(&cutsq[0][0], 0, np * np * sizeof(double));
}

/* -------------------------------------------------------------------
   global settings
-------------------------------------------------------------------- */

void PairMGP::settings(int narg, char ** /*arg*/) {
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

/* --------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
-------------------------------------------------------------------- */

void PairMGP::coeff(int narg, char **arg) {
  int me = comm->me;

  int n3 = 3 + atom->ntypes;
  if (narg < n3)
    error->all(FLERR, "MGP coeff: Incorrect args for pair coefficients");

  // TO DO, this should be compatible with other potentials?
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "MGP coeff: Incorrect args for pair coefficients");

  if (!allocated)
    allocate();

  // parse pair of atom types
  elements = new char *[atom->ntypes + 1];

  int n;
  for (int i = 3; i < n3; i++) {
    n = strlen(arg[i]);
    elements[i - 2] = new char[n];
    strcpy(elements[i - 2], arg[i]);
    if (me == 0 && screen)
      fprintf(screen, "MGP coeff: type %d is element %s\n", i - 2,
              elements[i - 2]);
  }

  if (narg >= (4 + atom->ntypes)) {
    if (strcmp(arg[n3], "yes") == 0)
      compute2b = true;
    else if (strcmp(arg[n3], "no") == 0)
      compute2b = false;
    else {
      error->all(FLERR, "MGP coeff:Incorrect args for pair coefficients. The "
                        "argument should be either yes or no");
    }
  }

  if (narg > (4 + atom->ntypes)) {
    if (strcmp(arg[n3 + 1], "yes") == 0)
      compute3b = true;
    else if (strcmp(arg[n3 + 1], "no") == 0)
      compute3b = false;
    else {
      error->all(FLERR, "MGP coeff:Incorrect args for pair coefficients. The "
                        "argument should be either yes or no");
    }
  }

  // read_file(arg[2]);
  if (me == 0)
    read_file(arg[2]);
  bcast_table();

  if (me == 0 && screen)
    fprintf(screen, "MGP coeff: done reading coefficients\n");
}

/* -------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
-------------------------------------------------------------------- */

void PairMGP::grab(FILE *fptr, int n, double *list) {
  char *ptr;
  char line[MAXLINE];

  int i = 0;
  while (i < n) {
    fgets(line, MAXLINE, fptr);
    ptr = strtok(line, " \t\n\r\f");
    list[i++] = atof(ptr);
    while ((ptr = strtok(NULL, " \t\n\r\f")))
      list[i++] = atof(ptr);
  }
}

/* ----------------------------------------------------------------- */

// double PairMGP::single(int i, int j, int itype, int jtype,
//                       double rsq, double /*factor_coul*/, double
//                       /*factor_lj*/, double &fforce) {
//  return 0;
//}
//

/* -------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
-------------------------------------------------------------------- */

void PairMGP::read_file(char *filename) {

  // variables that need to be set in the class member
  if (screen)
    fprintf(screen, "MGP read_file: reading potential parameters...\n");

  FILE *fptr;
  char line[MAXLINE];

  fptr = utils::open_potential(filename,lmp,nullptr);
  if (fptr == NULL) {
    char str[128];
    snprintf(str, 128, "Cannot open MMF potential file %s", filename);
    error->one(FLERR, str);
  }

  fgets(line, MAXLINE, fptr); // first line is comment
  fgets(line, MAXLINE, fptr); // second line is comment
  fgets(line, MAXLINE, fptr); // third line is comment
  fgets(line, MAXLINE, fptr);
  sscanf(line, "%d %d", &n_2body, &n_3body);
  if (screen)
    fprintf(screen,
            "MGP read_file: the file contains %d 2b potential and %d 3b "
            "potential\n",
            n_2body, n_3body);

  int ntype_p1 = atom->ntypes + 1;
  if (n_2body > 0) {
    memory->create(lo_2body, n_2body, "pair:lo_2body");
    memory->create(hi_2body, n_2body, "pair:hi_2body");
    memory->create(grid_2body, n_2body, "pair:grid_2body");
    memory->create(map2b, ntype_p1, ntype_p1, "pair:map2b");
    memory->create(cut2bsq, n_2body, "pair:cut2bsq");

    for (int i = 1; i < ntype_p1; i++) {
      for (int j = 1; j < ntype_p1; j++) {
        map2b[i][j] = -1;
      }
    }

    // ????????????????????/pair: 3body?
    fcoeff_2body = (double **)memory->smalloc(n_2body * sizeof(double *),
                                              "pair:fcoeff_3body");
    // ecoeff_2body = (double **) memory->smalloc(n_2body*sizeof(double
    // *),"pair:ecoeff_3body");
  } else {
    compute2b = false;
  }

  if (n_3body > 0) {
    memory->create(lo_3body, n_3body, 3, "pair:lo_3body");
    memory->create(hi_3body, n_3body, 3, "pair:hi_3body");
    memory->create(grid_3body, n_3body, 3, "pair:grid_3body");
    memory->create(map3b, ntype_p1, ntype_p1, ntype_p1, "pair:map3b");
    memory->create(cut3bsq, n_3body, "pair:cut3bsq");
    for (int i = 1; i < ntype_p1; i++) {
      for (int j = 1; j < ntype_p1; j++) {
        for (int k = 1; k < ntype_p1; k++) {
          map3b[i][j][k] = -1;
        }
      }
    }
    fcoeff_3body = (double **)memory->smalloc(n_3body * sizeof(double *),
                                              "pair:coeff_2body");
    // ecoeff_3body = (double **) memory->smalloc(n_3body*sizeof(double
    // *),"pair:ecoeff_2body");
  } else {
    compute3b = false;
  }

  cutmax = 0;
  for (int idx = 0; idx < n_2body; idx++) {
    char ele1[10], ele2[10];
    double a, b;
    int order;
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s %s %lg %lg %d", &ele1, &ele2, &a, &b, &order);

    bool type1[atom->ntypes + 1];
    bool type2[atom->ntypes + 1];
    for (int itype = 1; itype <= atom->ntypes; itype++) {
      if (strcmp(elements[itype], ele1) == 0)
        type1[itype] = true;
      else
        type1[itype] = false;
      if (strcmp(elements[itype], ele2) == 0)
        type2[itype] = true;
      else
        type2[itype] = false;
    }

    double rsq = b * b;
    cut2bsq[idx] = rsq;
    for (int itype = 1; itype <= atom->ntypes; itype++) {
      for (int jtype = 1; jtype <= atom->ntypes; jtype++) {
        if (type1[itype] && type2[jtype]) {
          map2b[itype][jtype] = idx;
          map2b[jtype][itype] = idx;
          if (compute2b) {
            setflag[itype][jtype] = 1;
            setflag[jtype][itype] = 1;
          }
          cutsq[itype][jtype] = rsq;
          cutsq[jtype][itype] = rsq;
        }
      }
    }

    if (cutmax < rsq)
      cutmax = rsq;

    lo_2body[idx] = a;
    hi_2body[idx] = b;
    grid_2body[idx] = order;
    memory->create(fcoeff_2body[idx], order + 2, "pair:2body_coeff");
    // memory->create(ecoeff_2body[idx],order+2,"pair:2body_ecoeff");

    grab(fptr, order + 2, fcoeff_2body[idx]);
    // grab(fptr, order+2, ecoeff_2body[idx]);
  }

  for (int idx = 0; idx < n_3body; idx++) {
    if (screen)
      fprintf(screen, "MGP read_file: reading the %d 3b       potential\n",
              idx);
    char ele1[10], ele2[10], ele3[10];
    double a[3], b[3];
    int order[3];

    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s %s %s %lg %lg %lg %lg %lg %lg %d %d %d", &ele1, &ele2,
           &ele3, &a[0], &a[1], &a[2], &b[0], &b[1], &b[2], &order[0],
           &order[1], &order[2]);

    bool type1[atom->ntypes + 1];
    bool type2[atom->ntypes + 1];
    bool type3[atom->ntypes + 1];
    for (int itype = 1; itype <= atom->ntypes; itype++) {
      if (strcmp(elements[itype], ele1) == 0)
        type1[itype] = true;
      else
        type1[itype] = false;
      if (strcmp(elements[itype], ele2) == 0)
        type2[itype] = true;
      else
        type2[itype] = false;
      if (strcmp(elements[itype], ele3) == 0)
        type3[itype] = true;
      else
        type3[itype] = false;
    }

    double rsq = b[0] * b[0];
    cut3bsq[idx] = rsq;
    for (int itype = 1; itype <= atom->ntypes; itype++) {
      for (int jtype = 1; jtype <= atom->ntypes; jtype++) {
        for (int ktype = 1; ktype <= atom->ntypes; ktype++) {
          if (type1[itype] && type2[jtype] && type3[ktype]) {
            map3b[itype][jtype][ktype] = idx;
            // no permutation sym
            // map3b[itype][ktype][jtype] = idx;
            if (compute3b) {
              setflag[itype][jtype] = 1;
              setflag[jtype][itype] = 1;
              setflag[itype][ktype] = 1;
              setflag[ktype][itype] = 1;
              setflag[jtype][ktype] = 1;
              setflag[ktype][jtype] = 1;
            }
          }
        }
      }
    }

    if (cutmax < rsq)
      cutmax = rsq;
    if (rsq > cutshortsq)
      cutshortsq = rsq;

    lo_3body[idx][0] = a[0];
    lo_3body[idx][1] = a[1];
    lo_3body[idx][2] = a[2];
    hi_3body[idx][0] = b[0];
    hi_3body[idx][1] = b[1];
    hi_3body[idx][2] = b[2];
    grid_3body[idx][0] = order[0];
    grid_3body[idx][1] = order[1];
    grid_3body[idx][2] = order[2];

    int len = (order[0] + 2) * (order[1] + 2) * (order[2] + 2);
    memory->create(fcoeff_3body[idx], len, "pair:3body_coeff");
    // memory->create(ecoeff_3body[idx], len, "pair:3body_ecoeff");

    grab(fptr, len, fcoeff_3body[idx]);
    // grab(fptr, len, ecoeff_3body[idx]);
  }

  fclose(fptr);

  cutmax = sqrt(cutmax);

  // if (screen) fprintf(screen, "MGP read_file: sanity check on setflag\n");
  for (int i = 1; i < ntype_p1; i++) {
    for (int j = 1; j < ntype_p1; j++) {
      if (setflag[i][j] == 0) {
        char str[128]; // read files, warning if some potential is not covered
        snprintf(str, 128,
                 "MGP read_file: no MGP potential for type %d %d is defined", i,
                 j);
        error->all(FLERR, str);
        // setflag[i][j] == 1; // can be covered by other potentials
      }
    }
  }

  // if (screen) fprintf(screen, "MGP read_file: sanity check on 2b
  // potential\n");
  if (compute2b) {
    for (int i = 1; i < ntype_p1; i++) {
      for (int j = 1; j < ntype_p1; j++) {
        if (map2b[i][j] == -1) {
          char str[128];
          snprintf(
              str, 128,
              "MGP read_file: 2b MGP potential for type %d %d is not defined",
              i, j);
          error->warning(FLERR, str);
        }
      }
    }
  }

  // if (screen) fprintf(screen, "MGP read_file: sanity check on 3b
  // potential\n");
  if (compute3b) {
    for (int i = 1; i < ntype_p1; i++) {
      for (int j = 1; j < ntype_p1; j++) {
        for (int k = 1; k < ntype_p1; k++) {
          if (map3b[i][j][k] == -1) {
            char str[128];
            snprintf(str, 128,
                     "MGP read_file: 3b MGP potential for type %d %d %d are "
                     "not defined",
                     i, j, k);
            error->warning(FLERR, str);
          }
        }
      }
    }
  }

  if (screen)
    fprintf(screen, "MGP read_file: done\n");
}

void PairMGP::bcast_table() {
  int me; // = comm->me;
  MPI_Comm_rank(world, &me);

  MPI_Bcast(&n_2body, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_3body, 1, MPI_INT, 0, world);

  int ntype_p1 = atom->ntypes + 1;
  if (me > 0) {
    if (n_2body > 0) {
      memory->create(lo_2body, n_2body, "pair:lo_2body");
      memory->create(hi_2body, n_2body, "pair:hi_2body");
      memory->create(grid_2body, n_2body, "pair:grid_2body");
      memory->create(cut2bsq, n_2body, "pair:cut2bsq");
      memory->create(map2b, ntype_p1, ntype_p1, "pair:map2b");
      fcoeff_2body = (double **)memory->smalloc(n_2body * sizeof(double *),
                                                "pair:fcoeff_2body");
      // ecoeff_2body = (double **) memory->smalloc(n_2body*sizeof(double
      // *),"pair:ecoeff_2body");
    }
    if (n_3body > 0) {
      memory->create(lo_3body, n_3body, 3, "pair:lo_3body");
      memory->create(hi_3body, n_3body, 3, "pair:hi_3body");
      memory->create(grid_3body, n_3body, 3, "pair:grid_3body");
      memory->create(cut3bsq, n_3body, "pair:cut3bsq");
      memory->create(map3b, ntype_p1, ntype_p1, ntype_p1, "pair:map3b");
      fcoeff_3body = (double **)memory->smalloc(n_3body * sizeof(double *),
                                                "pair:fcoeff_3body");
      // ecoeff_3body = (double **) memory->smalloc(n_3body*sizeof(double
      // *),"pair:ecoeff_3body");
    }
  }

  MPI_Bcast(cutsq[0], ntype_p1 * ntype_p1, MPI_DOUBLE, 0, world);
  MPI_Bcast(setflag[0], ntype_p1 * ntype_p1, MPI_INT, 0, world);
  MPI_Bcast(&cutmax, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cutshortsq, 1, MPI_DOUBLE, 0, world);

  char temp[100];
  if (n_2body > 0) {
    MPI_Bcast(lo_2body, n_2body, MPI_DOUBLE, 0, world);
    MPI_Bcast(hi_2body, n_2body, MPI_DOUBLE, 0, world);
    MPI_Bcast(grid_2body, n_2body, MPI_INT, 0, world);
    MPI_Bcast(cut2bsq, n_2body, MPI_DOUBLE, 0, world);
    MPI_Bcast(map2b[0], ntype_p1 * ntype_p1, MPI_INT, 0, world);
    for (int idx = 0; idx < n_2body; idx++) {
      if (me > 0) {
        sprintf(temp, "pair:fcoeff_2body%d", idx);
        memory->create(fcoeff_2body[idx], grid_2body[idx] + 2, temp);
        // sprintf(temp, "pair:ecoeff_2body%d", idx);
        // memory->create(fcoeff_2body[idx],grid_2body[idx]+2,temp);
      }
      MPI_Bcast(fcoeff_2body[idx], grid_2body[idx] + 2, MPI_DOUBLE, 0, world);
      // MPI_Bcast(ecoeff_2body[idx],order+2,MPI_DOUBLE,0,world);
    }
  }

  if (n_3body > 0) {
    MPI_Bcast(lo_3body[0], n_3body * 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(hi_3body[0], n_3body * 3, MPI_DOUBLE, 0, world);
    MPI_Bcast(grid_3body[0], n_3body * 3, MPI_INT, 0, world);
    MPI_Bcast(cut3bsq, n_3body, MPI_DOUBLE, 0, world);
    MPI_Bcast(map3b[0][0], ntype_p1 * ntype_p1 * ntype_p1, MPI_INT, 0, world);
    for (int idx = 0; idx < n_3body; idx++) {
      int *order = grid_3body[idx];
      int len = (order[0] + 2) * (order[1] + 2) * (order[2] + 2);
      if (me > 0) {
        sprintf(temp, "pair:fcoeff_3body%d", idx);
        memory->create(fcoeff_3body[idx], len, temp);
        // sprintf(temp, "pair:ecoeff_3body%d", idx);
        // memory->create(fcoeff_3body[idx], len, temp);
      }
      MPI_Bcast(fcoeff_3body[idx], len, MPI_DOUBLE, 0, world);
      // MPI_Bcast(ecoeff_3body[idx],len,MPI_DOUBLE,0,world);
    }
  }
}

/* -------------------------------------------------------------------
   init specific to this pair style
-------------------------------------------------------------------- */

void PairMGP::init_style() {
  if (force->newton_pair == 1) {
    if (comm->me == 0)
      error->warning(FLERR, "Pair style MGP requires newton pair off");
    //          error->warning(FLERR,"Pair style MGP is not using newton pair.
    //          But it does not need the newton pair flag to be off");
  }

  // need a full neighbor list
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ------------------------------------------------------------------
 *     init for one type pair i,j and corresponding j,i
 * ---------------------------------------------------------------- */

double PairMGP::init_one(int i, int j) {
  if (setflag[i][j] == 0)
    error->all(FLERR, "All pair coeffs are not set");
  return cutmax;
  // return sqrt(cutsq[i][j]);
}

void PairMGP::eval_cubic_splines_1d(double a, double b, int orders,
                                    double *coefs, double r, double *val,
                                    double *dval) {
  // some coefficients
  int i, j, ii;
  double dinv, u, i0, tt;
  *val = 0;
  dinv = (orders - 1.0) / (b - a);
  u = (r - a) * dinv;
  i0 = floor(u);
  ii = fmax(fmin(i0, orders - 2), 0);
  tt = u - ii;

  // interpolation points
  double tp[4];
  for (j = 0; j < 4; j++) {
    tp[j] = pow(tt, 3 - j);
  }

  // value of cubic spline function
  double Phi[4], dPhi[4];
  double dt;
  int k;

  if (tt < 0) {
    for (i = 0; i < 4; i++) {
      Phi[i] = dAd[i][3] * tt + Ad[i][3];
    }
  } else if (tt > 1) {
    dt = tt - 1;
    for (i = 0; i < 4; i++) {
      Phi[i] = Bd[i] * dt + Cd[i];
    }
  } else {
    for (i = 0; i < 4; i++) {
      Phi[i] = 0;
      for (k = 0; k < 4; k++) {
        Phi[i] += Ad[i][k] * tp[k];
      }
    }
  }

  // value of derivative of spline
  for (i = 0; i < 4; i++) {
    dPhi[i] = 0;
    for (k = 0; k < 4; k++) {
      dPhi[i] += dAd[i][k] * tp[k];
    }
    dPhi[i] *= dinv;
  }

  // added by coefficients
  int N = orders + 2;
  double pc = 0;
  double ppc = 0;

  for (j = 0; j < 4; j++) {
    *val += Phi[j] * coefs[ii + j];
    *dval += dPhi[j] * coefs[ii + j];
  }
}

void PairMGP::eval_cubic_splines_3d(double *a, double *b, int *orders,
                                    double *coefs, double r1, double r2,
                                    double a12, double *val, double *dval) {
  int dim = 3;
  int i;
  int j;
  double point[3] = {r1, r2, a12};
  double dinv[dim];
  double u[dim];
  double i0[dim];
  int ii[dim];
  double tt[dim];

  *val = 0;
  // coefficients
  for (i = 0; i < dim; i++) {
    dinv[i] = (orders[i] - 1.0) / (b[i] - a[i]);
    u[i] = (point[i] - a[i]) * dinv[i];
    i0[i] = floor(u[i]);
    ii[i] = fmax(fmin(i0[i], orders[i] - 2), 0);
    tt[i] = u[i] - ii[i];
  }

  // points
  double tp[dim][4];
  for (i = 0; i < dim; i++) {
    for (j = 0; j < 4; j++) {
      tp[i][j] = pow(tt[i], 3 - j);
    }
  }

  double Phi[dim][4], dPhi[dim][4];
  double dt;
  int k;

  for (j = 0; j < dim; j++) {

    // evaluate spline function
    if (tt[j] < 0) {
      for (i = 0; i < 4; i++) {
        Phi[j][i] = dAd[i][3] * tt[j] + Ad[i][3];
      }
    } else if (tt[j] > 1) {
      dt = tt[j] - 1;
      for (i = 0; i < 4; i++) {
        Phi[j][i] = Bd[i] * dt + Cd[i];
      }
    } else {
      for (i = 0; i < 4; i++) {
        Phi[j][i] = 0;
        for (k = 0; k < 4; k++) {
          Phi[j][i] += Ad[i][k] * tp[j][k];
        }
      }
    }

    // evaluate derivatives
    for (i = 0; i < 4; i++) {
      dPhi[j][i] = 0;
      for (k = 0; k < 4; k++) {
        dPhi[j][i] += dAd[i][k] * tp[j][k];
      }
      dPhi[j][i] *= dinv[j];
    }
  }

  // added by coefficients
  int N[dim];
  for (i = 0; i < dim; i++) {
    N[i] = orders[i] + 2;
  }
  double c, pc, ppc;
  double dpc, dppc1, dppc2;

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
    *val += Phi[0][i] * ppc;
    dval[0] += dPhi[0][i] * ppc;
    dval[1] += Phi[0][i] * dppc1;
    dval[2] += Phi[0][i] * dppc2;
  }
}
