#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "pair_flare.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

// flare++ modules
#include "y_grad.h"
#include "cutoffs.h"
#include "radial.h"
#include "lammps_descriptor.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairFLARE::PairFLARE(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;

  beta = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairFLARE::~PairFLARE()
{
  if (copymode) return;

  memory->destroy(beta);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairFLARE::compute(int eflag, int vflag)
{
  int i,j,ii,jj,m,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,p,rhoip,rhojp,z2,z2p,recip,phip,psip,phi;
  double *coeff;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Test the single bond function.
  std::cout << "Testing single bond function." << std::endl;
  int atom_index = 0;
  std::function<void(std::vector<double> &, std::vector<double> &, double,
        int, std::vector<double>)> basis_function = chebyshev;
  std::function<void(std::vector<double> &, double, double,
        std::vector<double>)> cutoff_function = quadratic_cutoff;
  double cutoff = 5.0;
  int n_species = 2;
  int N = 5;
  int lmax = 3;
  std::vector<double> radial_hyps = {0, cutoff};
  std::vector<double> cutoff_hyps;
  Eigen::VectorXd single_bond_vals;
  Eigen::MatrixXd environment_force_dervs;
  Eigen::MatrixXd central_force_dervs;

  single_bond(x, atom_index, type, inum, ilist, numneigh, firstneigh,
    basis_function, cutoff_function, cutoff, n_species, N, lmax,
    radial_hyps, cutoff_hyps, single_bond_vals, environment_force_dervs,
    central_force_dervs);

  std::cout << "Single bond values:" << std::endl;
  std::cout << single_bond_vals << std::endl;

  std::cout << "Environment derivatives rows:" << std::endl;
  std::cout << environment_force_dervs.rows() << std::endl;

  std::cout << "Environment derivatives columns:" << std::endl;
  std::cout << environment_force_dervs.cols() << std::endl;

  std::cout << "Central derivatives:" << std::endl;
  std::cout << central_force_dervs << std::endl;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    std::cout << "Printing i" << std::endl;
    std::cout << i << std::endl;
    std::cout << "Printing itype" << std::endl;
    std::cout << itype << std::endl;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairFLARE::allocate()
{
  // Based on tersoff.cpp.

  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");

  // Set the diagonal of setflag to 1 (otherwise pair.cpp will throw an error)
  for (int i = 1; i <= n; i++)
      setflag[i][i] = 1;

  // Create cutsq array (used in pair.cpp)
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairFLARE::settings(int narg, char **/*arg*/)
{
  // "flare" should be the only word after "pair_style" in the input file.
  if (narg > 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void PairFLARE::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != 3) error->all(FLERR,"Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // TODO: Parse the beta file. Set n_species, n_max, l_max, radial basis
  //    set, cutoff function, and beta array.

}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairFLARE::init_style()
{
  // Need a full neighbor list. Default is a half-neighbor list.
  // See tersoff.cpp for a potential that requires a full neighbor list and newton on.
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairFLARE::init_one(int i, int j)
{
  // init_one is called for each i, j pair in pair.cpp after calling init_style.

    // TODO: Set cutoff based on beta file.
    return 5.0;
//   return cutoff;
}

/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void PairFLARE::read_file(char *filename)
{
  // TODO: Implement based on pair_eam.cpp.
}


/* ----------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
------------------------------------------------------------------------- */

void PairFLARE::grab(FILE *fptr, int n, double *list)
{
  char *ptr;
  char line[MAXLINE];

  int i = 0;
  while (i < n) {
    fgets(line,MAXLINE,fptr);
    ptr = strtok(line," \t\n\r\f");
    list[i++] = atof(ptr);
    while ((ptr = strtok(NULL," \t\n\r\f"))) list[i++] = atof(ptr);
  }
}
