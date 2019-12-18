#include <vector>
#include <Eigen/Dense>

// Structure class.
class Structure{
    public:
        Eigen::MatrixXd cell, cell_transpose, cell_transpose_inverse,
                        cell_dot, cell_dot_inverse, positions,
                        wrapped_positions;
        std::vector<int> species;
        double max_cutoff;

        Structure(const Eigen::MatrixXd & cell,
                  const std::vector<int> & species,
                  const Eigen::MatrixXd & positions);

        Eigen::MatrixXd wrap_positions();

        double get_max_cutoff();
};

// Local environment class.
class LocalEnvironment{
    public:
        std::vector<int> environment_indices, environment_species;
        int central_index, central_species;
        std::vector<double> rs, xs, ys, zs;
        double cutoff;
        int sweep;

        LocalEnvironment(Structure & structure, int atom,
                         double cutoff);

        void compute_environment(Structure & structure,
                                 std::vector<int> & environment_indices,
                                 std::vector<int> & environment_species,
                                 std::vector<double> & rs,
                                 std::vector<double> & xs,
                                 std::vector<double> & ys,
                                 std::vector<double> & zs,
                                 int sweep_val);
};

// Spherical harmonics.
void get_Y(std::vector<double> & Y, std::vector<double> & Yx,
           std::vector<double> & Yy, std::vector<double> & Yz, double x,
           double y, double z, int l);

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
