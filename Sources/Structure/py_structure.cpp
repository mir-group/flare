#include "py_structure.h"
#include "structure.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <vector>
#include <Eigen/Dense>

namespace py = pybind11;

void AddStructureModule(py::module m) {
    py::class_<Structure>(m, "Structure")
        .def(py::init<const Eigen::MatrixXd &,
                      const std::vector<int> &,
                      const Eigen::MatrixXd &>())
        .def_property("cell", &Structure::get_cell, &Structure::set_cell)
        .def_property("positions", &Structure::get_positions,
                      &Structure::set_positions)
        .def_property_readonly("wrapped_positions",
            &Structure::get_wrapped_positions)
        .def_readonly("species", &Structure::species)
        .def_readonly("volume", &Structure::volume)
        .def_readonly("noa", &Structure::noa)
        .def_readonly("max_cutoff", &Structure::max_cutoff);
}
