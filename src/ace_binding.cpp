#include "y_grad.h"
#include "structure.h"
#include "local_environment.h"
#include "descriptor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <vector>
#include <Eigen/Dense>

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
        .def_readwrite("volume", &Structure::volume)
        .def("wrap_positions", &Structure::wrap_positions);

    py::class_<LocalEnvironment>(m, "LocalEnvironment")
        .def(py::init<const Structure &, int, double>())
        .def_readwrite("sweep", &LocalEnvironment::sweep)
        .def_readwrite("structure_volume", &LocalEnvironment::structure_volume)
        .def_readwrite("central_index", &LocalEnvironment::central_index)
        .def_readwrite("central_species", &LocalEnvironment::central_species)
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

    py::class_<B2_Calculator>(m, "B2_Calculator")
        .def(py::init<const std::string &, const std::string &,
             const std::vector<double> &, const std::vector<double> &,
             const std::vector<int> &>())
        .def_readwrite("radial_basis", &B2_Calculator::radial_basis)
        .def_readwrite("cutoff_function", &B2_Calculator::cutoff_function)
        .def_readwrite("descriptor_vals", &B2_Calculator::descriptor_vals)
        .def_readwrite("descriptor_force_dervs",
            &B2_Calculator::descriptor_force_dervs)
        .def_readwrite("descriptor_stress_dervs",
            &B2_Calculator::descriptor_stress_dervs)
        .def("compute", &B2_Calculator::compute);
}
