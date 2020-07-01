#include "py_structure.h"
#include "structure.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

namespace py = pybind11;

void AddStructureModule(py::module m) {
    py::class_<Structure>(m, "Structure")
        // Constructor with integer species.
        .def(py::init<const Eigen::MatrixXd &,
                      const std::vector<int> &,
                      const Eigen::MatrixXd &,
                      const std::unordered_map<int, double> &,
                      const Eigen::MatrixXd &,
                      const std::vector<std::string> &>(),
             py::arg("cell"), py::arg("species"), py::arg("positions"),
             py::arg("mass_dict") = std::unordered_map<int, double> {},
             py::arg("prev_positions") = Eigen::MatrixXd::Zero(0, 3),
             py::arg("species_labels") = std::vector<std::string> {})

        // Constructor with string species.
        .def(py::init<const Eigen::MatrixXd &,
                      const std::vector<std::string> &,
                      const Eigen::MatrixXd &,
                      const std::unordered_map<int, double> &,
                      const Eigen::MatrixXd &,
                      const std::vector<std::string> &>(),
             py::arg("cell"), py::arg("species"), py::arg("positions"),
             py::arg("mass_dict") = std::unordered_map<int, double> {},
             py::arg("prev_positions") = Eigen::MatrixXd::Zero(0, 3),
             py::arg("species_labels") = std::vector<std::string> {})

        .def_property("cell", &Structure::get_cell, &Structure::set_cell)
        .def_property("positions", &Structure::get_positions,
                      &Structure::set_positions)
        .def_property_readonly("wrapped_positions",
            &Structure::get_wrapped_positions)
        .def_readonly("species", &Structure::species)
        .def_readonly("volume", &Structure::volume)
        .def_readonly("nat", &Structure::nat)
        .def_readonly("max_cutoff", &Structure::max_cutoff)
        .def_readwrite("prev_positions", &Structure::prev_positions)
        .def_readwrite("species_labels", &Structure::species_labels)
        .def_readwrite("mass_dict", &Structure::mass_dict)
        .def_readwrite("forces", &Structure::forces)
        .def_readwrite("stds", &Structure::stds);
}
