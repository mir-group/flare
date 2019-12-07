// Spherical harmonics.
void get_Y(double * Y, double * Yx, double * Yy, double * Yz,
           double x, double y, double z, int l);

// Radial basis sets.
void equispaced_gaussians(double * basis_vals, double * basis_derivs,
                          double r, int N, double * radial_hyps);

// Radial cutoff functions.
void cos_cutoff(double * rcut_vals, double r, double rcut,
                double * cutoff_hyps);

void hard_cutoff(double * rcut_vals, double r, double rcut,
                 double * cutoff_hyps);

void calculate_radial(
    double * comb_vals, double * comb_x, double * comb_y, double * comb_z,
    void (*basis_function)(double *, double *, double, int, double *),
    void (*cutoff_function)(double *, double, double, double *),
    double x, double y, double z, double r, double rcut, int N,
    double * radial_hyps, double * cutoff_hyps);

// Single bond basis functions.
void single_bond_update(
double * single_bond_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, double *),
void (*cutoff_function)(double *, double, double, double *),
double x, double y, double z, double r, double rcut, int N, int lmax,
double * radial_hyps, double * cutoff_hyps);

void single_bond_sum(
double * single_bond_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, double *),
void (*cutoff_function)(double *, double, double, double *),
double * xs, double * ys, double * zs, double * rs, int * species,
int noa, double rcut, int N, int lmax,
double * radial_hyps, double * cutoff_hyps);

// Rotationally invariant descriptors.
void B2_descriptor(
double * descriptor_vals, double * environment_dervs, double * central_dervs,
void (*basis_function)(double *, double *, double, int, double *),
void (*cutoff_function)(double *, double, double, double *),
double * xs, double * ys, double * zs, double * rs, int * species,
int nos, int noa, double rcut, int N, int lmax,
double * radial_hyps, double * cutoff_hyps);
