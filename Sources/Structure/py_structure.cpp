#include "py_structure.h"
#include "structure.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <vector>
#include <Eigen/Dense>

namespace py = pybind11;

void AddStructureModule(py::module m) {
    // Structure module
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
}
