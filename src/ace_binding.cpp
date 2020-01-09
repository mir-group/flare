#include "ace.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

// Define spherical harmonics class.
class SphericalHarmonics{
    public:
        double x, y, z;
        int lmax;
        std::vector<double> Y, Yx, Yy, Yz;

        SphericalHarmonics(double x, double y, double z, int lmax);
};

SphericalHarmonics :: SphericalHarmonics(double x, double y, double z,
                                         int lmax){
    int number_of_harmonics = (lmax + 1) * (lmax + 1);

    this->x = x;
    this->y = y;
    this->z = z;
    this->lmax = lmax;

    // Initialize spherical harmonic vectors.
    Y = std::vector<double>(number_of_harmonics, 0);
    Yx = std::vector<double>(number_of_harmonics, 0);
    Yy = std::vector<double>(number_of_harmonics, 0);
    Yz = std::vector<double>(number_of_harmonics, 0);

    get_Y(Y, Yx, Yy, Yz, x, y, z, lmax);
};

PYBIND11_MODULE(ace, m){
    // Bind the spherical harmonics class.
    py::class_<SphericalHarmonics>(m, "SphericalHarmonics")
        .def(py::init<double, double, double, int>())
        // Make attributes accessible.
        .def_readwrite("x", &SphericalHarmonics::x)
        .def_readwrite("y", &SphericalHarmonics::y)
        .def_readwrite("z", &SphericalHarmonics::z)
        .def_readwrite("lmax", &SphericalHarmonics::lmax)
        .def_readwrite("Y", &SphericalHarmonics::Y)
        .def_readwrite("Yx", &SphericalHarmonics::Yx)
        .def_readwrite("Yy", &SphericalHarmonics::Yy)
        .def_readwrite("Yz", &SphericalHarmonics::Yz);

    // Bind the structure class.
    py::class_<Structure>(m, "Structure")
        .def(py::init<const Eigen::MatrixXd &,
                      const std::vector<int> &,
                      const Eigen::MatrixXd &>())
        .def_readwrite("cell", &Structure::cell)
        .def_readwrite("species", &Structure::species)          
        .def_readwrite("positions", &Structure::positions)
        .def_readwrite("cell_transpose", &Structure::cell_transpose)
        .def_readwrite("wrapped_positions", &Structure::wrapped_positions)
        .def("wrap_positions", &Structure::wrap_positions);
    
    py::class_<LocalEnvironment>(m, "LocalEnvironment")
        .def(py::init<const Structure &, int, double>())
        .def_readwrite("sweep", &LocalEnvironment::sweep)
        .def_readwrite("central_index", &LocalEnvironment::central_index)
        .def_readwrite("noa", &LocalEnvironment::noa)
        .def_readwrite("environment_indices",
            &LocalEnvironment::environment_indices)
        .def_readwrite("environment_species",
            &LocalEnvironment::environment_species)
        .def_readwrite("rs", &LocalEnvironment::rs)
        .def_readwrite("xs", &LocalEnvironment::xs)
        .def_readwrite("ys", &LocalEnvironment::ys)
        .def_readwrite("zs", &LocalEnvironment::zs)
        .def_readwrite("cutoff", &LocalEnvironment::cutoff);
    
    py::class_<DescriptorCalculator>(m, "DescriptorCalculator")
        .def(py::init<const std::string &, const std::string &,
             const std::vector<double> &, const std::vector<double> &,
             const std::vector<int> &>())
        .def_readwrite("radial_basis", &DescriptorCalculator::radial_basis)
        .def_readwrite("cutoff_function", 
            &DescriptorCalculator::cutoff_function)
        .def_readwrite("single_bond_vals",
            &DescriptorCalculator::single_bond_vals)
        .def_readwrite("single_bond_force_dervs",
            &DescriptorCalculator::single_bond_force_dervs)
        .def_readwrite("single_bond_stress_dervs",
            &DescriptorCalculator::single_bond_stress_dervs)
        .def_readwrite("descriptor_vals",
            &DescriptorCalculator::descriptor_vals)
        .def_readwrite("descriptor_force_dervs",
            &DescriptorCalculator::descriptor_force_dervs)
        .def_readwrite("descriptor_stress_dervs",
            &DescriptorCalculator::descriptor_stress_dervs)
        .def("compute_B1", &DescriptorCalculator::compute_B1)
        .def("compute_B2", &DescriptorCalculator::compute_B2);
}
