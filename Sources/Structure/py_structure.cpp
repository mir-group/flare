#include "py_structure.h"
#include "structure.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

void AddStructureModule(py::module m) {
  py::class_<Structure>(m, "Structure")
      // Constructor with integer species.
      .def(
          py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                   const Eigen::MatrixXd &,
                   const std::unordered_map<int, double> &,
                   const Eigen::MatrixXd &, const std::vector<std::string> &>(),
          py::arg("cell"), py::arg("species"), py::arg("positions"),
          py::arg("mass_dict") = std::unordered_map<int, double>{},
          py::arg("prev_positions") = Eigen::MatrixXd::Zero(0, 3),
          py::arg("species_labels") = std::vector<std::string>{})

      // Constructor with string species.
      .def(
          py::init<const Eigen::MatrixXd &, const std::vector<std::string> &,
                   const Eigen::MatrixXd &,
                   const std::unordered_map<std::string, double> &,
                   const Eigen::MatrixXd &, const std::vector<std::string> &>(),
          py::arg("cell"), py::arg("species"), py::arg("positions"),
          py::arg("mass_dict") = std::unordered_map<int, double>{},
          py::arg("prev_positions") = Eigen::MatrixXd::Zero(0, 3),
          py::arg("species_labels") = std::vector<std::string>{})

      .def_property("cell", &Structure::get_cell, &Structure::set_cell)
      .def_property("positions", &Structure::get_positions,
                    &Structure::set_positions)
      .def_property_readonly("wrapped_positions",
                             &Structure::get_wrapped_positions)
      .def_readonly("coded_species", &Structure::coded_species)
      .def_readonly("volume", &Structure::volume)
      .def_readonly("nat", &Structure::nat)
      .def_readonly("max_cutoff", &Structure::max_cutoff)
      .def_readwrite("prev_positions", &Structure::prev_positions)
      .def_readwrite("species_labels", &Structure::species_labels)
      .def_readwrite("mass_dict", &Structure::mass_dict)

      // Add computed properties. (Consider combining these in a
      // separate class.)
      .def_readwrite("forces", &Structure::forces)
      .def_readwrite("stds", &Structure::stds)
      .def_readwrite("local_energies", &Structure::local_energies)
      .def_readwrite("local_energy_stds", &Structure::local_energy_stds)
      .def_readwrite("partial_stresses", &Structure::partial_stresses)
      .def_readwrite("partial_stress_stds", &Structure::partial_stress_stds)
      .def_readwrite("stress", &Structure::stress)
      .def_readwrite("stress_stds", &Structure::stress_stds)
      .def_readwrite("potential_energy", &Structure::potential_energy)

      // Make the structure object picklable.
      // See
      // https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
      .def(py::pickle(
          [](const Structure &s) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                s.get_cell(), s.coded_species, s.get_positions(), s.mass_dict,
                s.species_labels, s.prev_positions, s.forces, s.stds);
          },
          [](py::tuple t) { // __setstate__
            if (t.size() != 8)
              throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            Structure p(t[0].cast<Eigen::MatrixXd>(),
                        t[1].cast<std::vector<int>>(),
                        t[2].cast<Eigen::MatrixXd>());

            /* Assign any additional state */
            p.forces = t[6].cast<Eigen::MatrixXd>();
            p.stds = t[7].cast<Eigen::MatrixXd>();

            return p;
          }));
}
