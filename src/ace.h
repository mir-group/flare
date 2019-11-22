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
