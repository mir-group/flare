#include "compute_flare_std_atom.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <omp.h>

// flare++ modules
#include "cutoffs.h"
#include "lammps_descriptor.h"
#include "radial.h"
#include "y_grad.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

ComputeFlareStdAtom::ComputeFlareStdAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  stds(nullptr)
{
  if (narg < 4) error->all(FLERR, "Illegal compute flare/std/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;
  timeflag = 1;
  comm_reverse = 1;

  // restartinfo = 0;
  // manybody_flag = 1;

  setflag = 0;
  cutsq = NULL;

  beta = NULL;
  coeff(narg, arg);

  nmax = 0;
  desc_derv = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

ComputeFlareStdAtom::~ComputeFlareStdAtom() {
  if (copymode)
    return;

  memory->destroy(beta);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }

  memory->destroy(stds);
  memory->destroy(desc_derv);
}

/* ----------------------------------------------------------------------
   init specific to this compute command
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::init() {
  // Require newton on.
//  if (force->newton_pair == 0)
//    error->all(FLERR, "Compute command requires newton pair on");

  // Request a full neighbor list.
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
}

void ComputeFlareStdAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}


/* ---------------------------------------------------------------------- */

void ComputeFlareStdAtom::compute_peratom() {
  if (atom->nmax > nmax) {
    memory->destroy(stds);
    nmax = atom->nmax;
    memory->create(stds,nmax,"flare/std/atom:stds");
    vector_atom = stds;
  }

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;
  int ntotal = nlocal;
  if (force->newton) ntotal += atom->nghost;

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  for (int ii = 0; ii < ntotal; ii++) {
    stds[ii] = 0.0;
  }

#pragma omp parallel
{
  double delx, dely, delz, xtmp, ytmp, ztmp, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  int inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int beta_init, beta_counter;
  double B2_norm_squared, B2_val_1, B2_val_2;

  Eigen::VectorXd single_bond_vals, B2_vals, B2_env_dot, beta_p, partial_forces, u;
  Eigen::MatrixXd single_bond_env_dervs, B2_env_dervs;
  double empty_thresh = 1e-8;

  #pragma omp for
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int itype = type[i];
    int jnum = numneigh[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];

    // Count the atoms inside the cutoff.
    int n_inner = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      int s = type[j] - 1;
      double cutoff_val = cutoff_matrix(itype-1, s);

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < (cutoff_val * cutoff_val)) {
        n_inner++;
      }

    }

    // Compute covariant descriptors.
    single_bond_multiple_cutoffs(x, type, jnum, n_inner, i, xtmp, ytmp, ztmp,
                                 jlist, basis_function, cutoff_function,
                                 n_species, n_max, l_max, radial_hyps,
                                 cutoff_hyps, single_bond_vals,
                                 single_bond_env_dervs, cutoff_matrix);

    // Compute invariant descriptors.
    B2_descriptor(B2_vals, B2_norm_squared,
                  single_bond_vals, n_species, n_max, l_max);

    double variance = 0.0;
    double sig = hyperparameters(0);
    double sig2 = sig * sig;

    // Continue if the environment is empty.
    if (B2_norm_squared < empty_thresh)
      continue;

    if (use_map) {
      Eigen::VectorXd Q_desc;
      double K_self;
      if (normalized) {
        K_self = 1.0;
        double B2_norm = pow(B2_norm_squared, 0.5);
        Q_desc = beta_matrices[itype - 1].transpose() * B2_vals / B2_norm;
      } else {
        K_self = B2_norm_squared; // only power 1 is supported
        Q_desc = beta_matrices[itype - 1].transpose() * B2_vals;
      }
      variance = K_self - Q_desc.dot(Q_desc);
    } else {
      Eigen::VectorXd kernel_vec = Eigen::VectorXd::Zero(n_clusters);
      double K_self;
      double B2_norm = pow(B2_norm_squared, 0.5);
      Eigen::VectorXd normed_B2 = B2_vals / B2_norm;
      int cum_types = 0;
      for (int s = 0; s < n_types; s++) {
        if (type[i] - 1 == s) {
          if (normalized) {
            kernel_vec.segment(cum_types, n_clusters_by_type[s]) = (normed_sparse_descriptors[s] * normed_B2).array().pow(power);
            K_self = 1.0;
          } else {
            // the normed_sparse_descriptors is non-normalized in this case
            kernel_vec.segment(cum_types, n_clusters_by_type[s]) = (normed_sparse_descriptors[s] * B2_vals).array().pow(power);
            K_self = pow(B2_norm_squared, power);
          }
        }
        cum_types += n_clusters_by_type[s];
      }
      Eigen::VectorXd L_inv_kv = L_inv_blocks[0] * kernel_vec;
      double Q_self = sig2 * L_inv_kv.transpose() * L_inv_kv;

      variance = K_self - Q_self;
    }

    // Compute the normalized variance, it could be negative
    if (variance >= 0.0) {
      stds[i] = pow(variance, 0.5);
    } else {
      stds[i] = - pow(abs(variance), 0.5);
    }

  }
} // #pragma
}

/* ---------------------------------------------------------------------- */

int ComputeFlareStdAtom::pack_reverse_comm(int n, int first, double *buf)
{
    // TODO: add desc_derv to this
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (int comp = 0; comp < 3; comp++) {
      buf[m++] = stds[i];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeFlareStdAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int comp = 0; comp < 3; comp++) {
      stds[j] += buf[m++];
    }
  }

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeFlareStdAtom::memory_usage()
{
  double bytes = nmax * 3 * (1 + n_descriptors) * sizeof(double);
  return bytes;
}



/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "compute:setflag");

  // Set the diagonal of setflag to 1 (otherwise pair.cpp will throw an error)
  for (int i = 1; i <= n; i++)
    setflag[i][i] = 1;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::coeff(int narg, char **arg) {
  if (!allocated)
    allocate();

  // Should be exactly 3 arguments following "compute" in the input file.
  if (narg == 4) {
    read_file(arg[3]);
    use_map = true;
  } else if (narg == 5) {
    read_L_inverse(arg[3]);
    read_sparse_descriptors(arg[4]);
    use_map = false;
  } else {
    error->all(FLERR, "Incorrect args for compute coefficients");
  }

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

//double ComputeFlareStdAtom::init_one(int i, int j) {
//  // init_one is called for each i, j pair in pair.cpp after calling init_style.
//  if (setflag[i][j] == 0)
//    error->all(FLERR, "All pair coeffs are not set");
//  return cutoff;
//}

/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::parse_cutoff_matrix(int n_species, FILE *fptr){
  int me = comm->me;

  // Parse the cutoffs.
  int n_cutoffs = n_species * n_species;
  memory->create(cutoffs, n_cutoffs, "compute:cutoffs");
  if (me == 0)
    grab(fptr, n_cutoffs, cutoffs);
  MPI_Bcast(cutoffs, n_cutoffs, MPI_DOUBLE, 0, world);

  // Create cutsq array (used in pair.cpp)
  memory->create(cutsq, n_species + 1, n_species + 1, "compute:cutsq");
  memset(&cutsq[0][0], 0, (n_species + 1) * (n_species + 1) * sizeof(double));

  // Fill in the cutoff matrix.
  cutoff = -1;
  cutoff_matrix = Eigen::MatrixXd::Zero(n_species, n_species);
  int cutoff_count = 0;
  for (int i = 0; i < n_species; i++){
    for (int j = 0; j < n_species; j++){
      double cutoff_val = cutoffs[cutoff_count];
      cutoff_matrix(i, j) = cutoff_val;
      cutsq[i + 1][j + 1] = cutoff_val * cutoff_val;
      if (cutoff_val > cutoff) cutoff = cutoff_val;
      cutoff_count ++;
    }
  }
}


void ComputeFlareStdAtom::read_file(char *filename) {
  int me = comm->me;
  char line[MAXLINE], radial_string[MAXLINE], cutoff_string[MAXLINE], kernel_string[MAXLINE];
  int radial_string_length, cutoff_string_length, kernel_string_length;
  FILE *fptr;

  // Check that the potential file can be opened.
  if (me == 0) {
    fptr = utils::open_potential(filename,lmp,nullptr);
    if (fptr == NULL) {
      char str[128];
      snprintf(str, 128, "Cannot open variance file %s", filename);
      error->one(FLERR, str);
    }
  }

  if (me == 0) {
    fgets(line, MAXLINE, fptr);

    fgets(line, MAXLINE, fptr); // hyperparameters
    sscanf(line, "%i", &n_hyps);
  }

  MPI_Bcast(&n_hyps, 1, MPI_INT, 0, world);
  hyperparameters = Eigen::VectorXd::Zero(n_hyps);
  if (me == 0) {
    fgets(line, MAXLINE, fptr); // hyperparameters
    double sig, en, fn, sn;
    sscanf(line, "%lg %lg %lg %lg", &sig, &en, &fn, &sn);
    hyperparameters(0) = sig;
    hyperparameters(1) = en;
    hyperparameters(2) = fn;
    hyperparameters(3) = sn;

    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", kernel_string); // kernel name
    kernel_string_length = strlen(kernel_string);

    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", radial_string); // Radial basis set
    radial_string_length = strlen(radial_string);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%i %i %i %i", &n_species, &n_max, &l_max, &beta_size);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", cutoff_string); // Cutoff function
    cutoff_string_length = strlen(cutoff_string);
  }

  MPI_Bcast(hyperparameters.data(), n_hyps, MPI_DOUBLE, 0, world);
  MPI_Bcast(&n_species, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&l_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&beta_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&cutoff, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&radial_string_length, 1, MPI_INT, 0, world);
  MPI_Bcast(&cutoff_string_length, 1, MPI_INT, 0, world);
  MPI_Bcast(&kernel_string_length, 1, MPI_INT, 0, world);
  MPI_Bcast(radial_string, radial_string_length + 1, MPI_CHAR, 0, world);
  MPI_Bcast(cutoff_string, cutoff_string_length + 1, MPI_CHAR, 0, world);
  MPI_Bcast(kernel_string, kernel_string_length + 1, MPI_CHAR, 0, world);

  // Parse the cutoffs and fill in the cutoff matrix
  parse_cutoff_matrix(n_species, fptr);

  // Set number of descriptors.
  int n_radial = n_max * n_species;
  n_descriptors = (n_radial * (n_radial + 1) / 2) * (l_max + 1);

  // Check the relationship between the power spectrum and beta.
  int beta_check = n_descriptors * n_descriptors;
  if (beta_check != beta_size)
    error->all(FLERR, "Beta size doesn't match the number of descriptors.");

  // Set the radial basis.
  if (!strcmp(radial_string, "chebyshev")) {
    basis_function = chebyshev;
    radial_hyps = std::vector<double>{0, cutoff};
  } else {
    error->all(FLERR, "Please use chebyshev radial basis function.");
  }

  // Set the cutoff function.
  if (!strcmp(cutoff_string, "quadratic")) {
    cutoff_function = quadratic_cutoff;
  } else if (!strcmp(cutoff_string, "cosine")) {
    cutoff_function = cos_cutoff;
  } else {
    error->all(FLERR, "Please use quadratic or cosine cutoff function.");
  }

  // Set the kernel
  if (!strcmp(kernel_string, "NormalizedDotProduct")) {
    normalized = true;
  } else {
    normalized = false;
  }

  // Parse the beta vectors.
  //memory->create(beta, beta_size * n_species * n_species, "compute:beta");
  memory->create(beta, beta_size * n_species, "compute:beta");

  if (me == 0)
  //  grab(fptr, beta_size * n_species * n_species, beta);
    grab(fptr, beta_size * n_species, beta);
  //MPI_Bcast(beta, beta_size * n_species * n_species, MPI_DOUBLE, 0, world);
  MPI_Bcast(beta, beta_size * n_species, MPI_DOUBLE, 0, world);

  // Fill in the beta matrix.
  // TODO: Remove factor of 2 from beta.
  int n_size = n_species * n_descriptors;
  int beta_count = 0;
  double beta_val;
  for (int k = 0; k < n_species; k++) {
    beta_matrix = Eigen::MatrixXd::Zero(n_descriptors, n_descriptors);
    for (int i = 0; i < n_descriptors; i++) {
      for (int j = 0; j < n_descriptors; j++) {
        beta_matrix(i, j) = beta[beta_count];
        beta_count++;
      }
    }
    beta_matrices.push_back(beta_matrix);
  }

}

void ComputeFlareStdAtom::read_L_inverse(char *filename) {
  int me = comm->me;
  char line[MAXLINE], radial_string[MAXLINE], cutoff_string[MAXLINE], kernel_string[MAXLINE];
  int radial_string_length, cutoff_string_length, kernel_string_length;
  FILE *fptr;

  // Check that the potential file can be opened.
  if (me == 0) {
    fptr = utils::open_potential(filename,lmp,nullptr);
    if (fptr == NULL) {
      char str[128];
      snprintf(str, 128, "Cannot open variance file %s", filename);
      error->one(FLERR, str);
    }
  }

  int tmp, nwords;
  if (me == 0) {
    fgets(line, MAXLINE, fptr); // skip the first line

    fgets(line, MAXLINE, fptr); // power
    sscanf(line, "%i %s", &power, kernel_string);
    kernel_string_length = strlen(kernel_string);

    fgets(line, MAXLINE, fptr); // hyperparameters
    sscanf(line, "%i", &n_hyps);
  }
  MPI_Bcast(&power, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_hyps, 1, MPI_INT, 0, world);

  hyperparameters = Eigen::VectorXd::Zero(n_hyps);
  if (me == 0) {
    fgets(line, MAXLINE, fptr); // hyperparameters
    double sig, en, fn, sn;
    sscanf(line, "%lg %lg %lg %lg", &sig, &en, &fn, &sn);
    hyperparameters(0) = sig;
    hyperparameters(1) = en;
    hyperparameters(2) = fn;
    hyperparameters(3) = sn;

    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", radial_string); // Radial basis set
    radial_string_length = strlen(radial_string);

    fgets(line, MAXLINE, fptr);
    sscanf(line, "%i %i %i %i", &n_species, &n_max, &l_max, &n_kernels);

    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", cutoff_string); // Cutoff function
    cutoff_string_length = strlen(cutoff_string);
  }

  MPI_Bcast(hyperparameters.data(), n_hyps, MPI_DOUBLE, 0, world);
  MPI_Bcast(&n_species, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&l_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_kernels, 1, MPI_INT, 0, world);
  MPI_Bcast(&cutoff, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&kernel_string_length, 1, MPI_INT, 0, world);
  MPI_Bcast(&radial_string_length, 1, MPI_INT, 0, world);
  MPI_Bcast(&cutoff_string_length, 1, MPI_INT, 0, world);
  MPI_Bcast(radial_string, radial_string_length + 1, MPI_CHAR, 0, world);
  MPI_Bcast(cutoff_string, cutoff_string_length + 1, MPI_CHAR, 0, world);
  MPI_Bcast(kernel_string, kernel_string_length + 1, MPI_CHAR, 0, world);

  // Parse the cutoffs and fill in the cutoff matrix
  parse_cutoff_matrix(n_species, fptr);

  // Parse number of sparse envs
  if (me == 0) {
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%i", &n_clusters);
  }
  MPI_Bcast(&n_clusters, 1, MPI_INT, 0, world);

  // Set number of descriptors.
  int n_radial = n_max * n_species;
  n_descriptors = (n_radial * (n_radial + 1) / 2) * (l_max + 1);

  // Check the relationship between the power spectrum and beta.
  int Linv_size = n_clusters * (n_clusters + 1) / 2;
//  if (beta_check != beta_size)
//    error->all(FLERR, "Beta size doesn't match the number of descriptors.");

  // Set the radial basis.
  if (!strcmp(radial_string, "chebyshev")) {
    basis_function = chebyshev;
    radial_hyps = std::vector<double>{0, cutoff};
  } else {
    error->all(FLERR, "Please use chebyshev radial basis function.");
  }

  // Set the cutoff function.
  if (!strcmp(cutoff_string, "quadratic")) {
    cutoff_function = quadratic_cutoff;
  } else if (!strcmp(cutoff_string, "cosine")) {
    cutoff_function = cos_cutoff;
  } else {
    error->all(FLERR, "Please use quadratic or cosine cutoff function.");
  }

  // Set the kernel
  if (!strcmp(kernel_string, "NormalizedDotProduct")) {
    normalized = true;
  } else {
    normalized = false;
  }

  // Parse the beta vectors.
  memory->create(beta, Linv_size, "compute:L_inv");
  if (me == 0)
    grab(fptr, Linv_size, beta);
  MPI_Bcast(beta, Linv_size, MPI_DOUBLE, 0, world);

  // Fill in the beta matrix.
  for (int i = 0; i < n_kernels; i++) {
    int count = 0;
    Eigen::MatrixXd Linv = Eigen::MatrixXd::Zero(n_clusters, n_clusters);
    for (int j = 0; j < n_clusters; j++) {
      for (int k = 0; k <= j; k++) {
        Linv(j, k) = beta[count];
        count++;
      }
    }
    L_inv_blocks.push_back(Linv);
  }
  memory->destroy(beta);
}

void ComputeFlareStdAtom::read_sparse_descriptors(char *filename) {
  int me = comm->me;
  char line[MAXLINE], radial_string[MAXLINE], cutoff_string[MAXLINE];
  int radial_string_length, cutoff_string_length;
  FILE *fptr;

  // Check that the potential file can be opened.
  if (me == 0) {
    fptr = utils::open_potential(filename,lmp,nullptr);
    if (fptr == NULL) {
      char str[128];
      snprintf(str, 128, "Cannot open variance file %s", filename);
      error->one(FLERR, str);
    }
  }

  int kernel_ind = 0;
  if (me == 0) {
    fgets(line, MAXLINE, fptr); // skip the first line

    fgets(line, MAXLINE, fptr); // hyperparameters
    int n_kern = 0;
    sscanf(line, "%i", &n_kern);
    if (n_kern != n_kernels) {
      error->all(FLERR, "n_kernals in sparse_descriptors and L_inv not match");
    }
  }

  for (int i = 0; i < n_kernels; i++) {
    if (me == 0) {
      fgets(line, MAXLINE, fptr);
      int n_clst = 0;
      sscanf(line, "%i %i %i", &kernel_ind, &n_clst, &n_types);
      if (n_clst != n_clusters) {
        error->all(FLERR, "n_clusters in sparse_descriptors and L_inv not match");
      }
    }
    MPI_Bcast(&n_types, 1, MPI_INT, 0, world);

    memory->create(n_clusters_by_type, n_types, "compute:n_clusters_by_type");
    for (int s = 0; s < n_types; s++) {
      int n_clst_by_type;
      if (me == 0) {
        fgets(line, MAXLINE, fptr);
        sscanf(line, "%i", &n_clst_by_type);
      }
      MPI_Bcast(&n_clst_by_type, 1, MPI_INT, 0, world);

      n_clusters_by_type[s] = n_clst_by_type;

      // Check the relationship between the power spectrum and beta.
      int sparse_desc_size = n_clst_by_type * n_descriptors;

      // Parse the beta vectors.
      memory->create(beta, sparse_desc_size, "compute:sparse_desc");
      if (me == 0)
        grab(fptr, sparse_desc_size, beta);
      MPI_Bcast(beta, sparse_desc_size, MPI_DOUBLE, 0, world);

      // Fill in the beta matrix.
      int count = 0;
      Eigen::MatrixXd sparse_desc = Eigen::MatrixXd::Zero(n_clst_by_type, n_descriptors);
      for (int j = 0; j < n_clst_by_type; j++) {
        for (int k = 0; k < n_descriptors; k++) {
          sparse_desc(j, k) = beta[count];
          count++;
        }
      }
      memory->destroy(beta);
      normed_sparse_descriptors.push_back(sparse_desc);
    }
  }

}


/* ----------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::grab(FILE *fptr, int n, double *list) {
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
