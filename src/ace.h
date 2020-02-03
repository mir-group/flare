#include <vector>
#include <Eigen/Dense>

// Structure class.
class Structure{
    public:
        Eigen::MatrixXd cell, cell_transpose, cell_transpose_inverse,
                        cell_dot, cell_dot_inverse, positions,
                        wrapped_positions;
        std::vector<int> species;
        double max_cutoff, volume;

        Structure();

        Structure(const Eigen::MatrixXd & cell,
                  const std::vector<int> & species,
                  const Eigen::MatrixXd & positions);

        Eigen::MatrixXd wrap_positions();

        double get_max_cutoff();
};

// Local environment class.
class LocalEnvironment{
    public:
        std::vector<int> environment_indices, environment_species,
            neighbor_list;
        int central_index, central_species, noa, sweep;
        std::vector<double> rs, xs, ys, zs;
        double cutoff, structure_volume;

        LocalEnvironment();

        LocalEnvironment(const Structure & structure, int atom,
                         double cutoff);

        static void compute_environment(const Structure & structure,
                                 int noa, int atom,
                                 double cutoff, int sweep_val,
                                 std::vector<int> & environment_indices,
                                 std::vector<int> & environment_species,
                                 std::vector<int> & neighbor_list,
                                 std::vector<double> & rs,
                                 std::vector<double> & xs,
                                 std::vector<double> & ys,
                                 std::vector<double> & zs);
};

// Descriptor calculator.
class DescriptorCalculator{
    private:
        void (*radial_pointer)(double *, double *, double, int,
                               std::vector<double>);
        void (*cutoff_pointer)(double *, double, double, std::vector<double>);
    public:
        Eigen::VectorXd single_bond_vals, descriptor_vals;
        Eigen::MatrixXd single_bond_force_dervs, single_bond_stress_dervs,
            descriptor_force_dervs, descriptor_stress_dervs;
        std::string radial_basis, cutoff_function;
        std::vector<double> radial_hyps, cutoff_hyps;
        std::vector<int> descriptor_settings;

    DescriptorCalculator();

    DescriptorCalculator(
        const std::string & radial_basis, const std::string & cutoff_function,
        const std::vector<double> & radial_hyps,
        const std::vector<double> & cutoff_hyps,
        const std::vector<int> & descriptor_settings);

    void compute_B1(const LocalEnvironment & env);

    void compute_B2(const LocalEnvironment & env);

};

// Local environment descriptor.
class LocalEnvironmentDescriptor : public LocalEnvironment{
    public:
        DescriptorCalculator descriptor_calculator;
        Eigen::VectorXd descriptor_vals;
        Eigen::MatrixXd descriptor_force_dervs;
        Eigen::MatrixXd descriptor_stress_dervs;

        double descriptor_norm;
        Eigen::MatrixXd force_dot, stress_dot;

        LocalEnvironmentDescriptor();

        LocalEnvironmentDescriptor(const Structure & structure, int atom,
            double cutoff, DescriptorCalculator & descriptor_calculator);
        
        void compute_descriptor();
};

// Structure descriptor. Computes descriptors of atomic environments in a
// structure.
class StructureDescriptor : public Structure{
    public:
        DescriptorCalculator descriptor_calculator;
        std::vector<Eigen::VectorXd> descriptor_vals;
        std::vector<Eigen::MatrixXd> descriptor_force_dervs;
        std::vector<Eigen::MatrixXd> descriptor_stress_dervs;
        double cutoff;

        StructureDescriptor();

        StructureDescriptor(const Eigen::MatrixXd & cell,
                            const std::vector<int> & species,
                            const Eigen::MatrixXd & positions,
                            DescriptorCalculator & descriptor_calculator,
                            double cutoff);

        void compute_descriptors();
};

// Structure dataset. Stores energy, force, and stress labels.
class StructureDataset : public StructureDescriptor{
    public:
        std::vector<double> energy;
        std::vector<double> force_components;
        std::vector<double> stress_components;

        StructureDataset();

        StructureDataset(const Eigen::MatrixXd & cell,
                         const std::vector<int> & species,
                         const Eigen::MatrixXd & positions,
                         DescriptorCalculator & descriptor_calculator,
                         double cutoff,
                         std::vector<double> energy = std::vector<double>{},
                         std::vector<double> force_components =
                            std::vector<double>{},
                         std::vector<double> stress_components =
                            std::vector<double>{});

};

// Spherical harmonics.
void get_Y(std::vector<double> & Y, std::vector<double> & Yx,
           std::vector<double> & Yy, std::vector<double> & Yz, const double x,
           const double y, const double z, const int l);

// Radial basis sets.
void equispaced_gaussians(double * basis_vals, double * basis_derivs,
                          double r, int N, std::vector<double> radial_hyps);

void chebyshev(double * basis_vals, double * basis_derivs,
               double r, int N, std::vector<double> radial_hyps);

// Radial cutoff functions.
void quadratic_cutoff(double * rcut_vals, double r, double rcut,
                      std::vector<double> cutoff_hyps);

void cos_cutoff(double * rcut_vals, double r, double rcut,
                std::vector<double> cutoff_hyps);

void hard_cutoff(double * rcut_vals, double r, double rcut,
                 std::vector<double> cutoff_hyps);

void calculate_radial(
    double * comb_vals, double * comb_x, double * comb_y, double * comb_z,
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>),
    void (*cutoff_function)(double *, double, double, std::vector<double>),
    double x, double y, double z, double r, double rcut, int N,
    std::vector<double> radial_hyps, std::vector<double> cutoff_hyps);

// Single bond basis functions.
void single_bond_update_env(
    Eigen::VectorXd & single_bond_vals,
    Eigen::MatrixXd & force_dervs, Eigen::MatrixXd & stress_dervs,
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>),
    void (*cutoff_function)(double *, double, double, std::vector<double>),
    double x, double y, double z, double r,  int s,
    int environment_index, int central_index,
    double rcut, int N, int lmax,
    const std::vector<double> & radial_hyps,
    const std::vector<double> & cutoff_hyps);

void single_bond_sum_env(
    Eigen::VectorXd & single_bond_vals,
    Eigen::MatrixXd & force_dervs, Eigen::MatrixXd & stress_dervs,
    void (*basis_function)(double *, double *, double, int,
                           std::vector<double>),
    void (*cutoff_function)(double *, double, double, std::vector<double>),
    const LocalEnvironment & env, double rcut, int N, int lmax,
    const std::vector<double> & radial_hyps,
    const std::vector<double> & cutoff_hyps);

// Rotationally invariant descriptors.
void B1_descriptor(
Eigen::VectorXd & B1_vals,
Eigen::MatrixXd & B1_force_dervs,
Eigen::MatrixXd & B1_stress_dervs,
const Eigen::VectorXd & single_bond_vals,
const Eigen::MatrixXd & single_bond_force_dervs,
const Eigen::MatrixXd & single_bond_stress_dervs,
const LocalEnvironment & env, int nos, int N, int lmax);

void B2_descriptor(
Eigen::VectorXd & B2_vals,
Eigen::MatrixXd & B2_force_dervs,
Eigen::MatrixXd & B2_stress_dervs,
const Eigen::VectorXd & single_bond_vals,
const Eigen::MatrixXd & single_bond_force_dervs,
const Eigen::MatrixXd & single_bond_stress_dervs,
const LocalEnvironment & env, int nos, int N, int lmax);
