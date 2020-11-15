#include "kernel.h"
#include "structure.h"
#include "y_grad.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(_C_flare, m) {
  // Structure
  py::class_<Structure>(m, "Structure")
      .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                    const Eigen::MatrixXd &>())
      .def_readwrite("cell", &Structure::cell)
      .def_readwrite("species", &Structure::species)
      .def_readwrite("positions", &Structure::positions)
      .def_readwrite("cell_transpose", &Structure::cell_transpose)
      .def_readwrite("wrapped_positions", &Structure::wrapped_positions)
      .def_readwrite("volume", &Structure::volume)
      .def("wrap_positions", &Structure::wrap_positions);

//   // Sparse GP DTC
//   py::class_<SparseGP_DTC>(m, "SparseGP_DTC")
//       .def(py::init<>())
//       .def(py::init<std::vector<Kernel *>, double, double, double>())
//       .def("set_hyperparameters", &SparseGP_DTC::set_hyperparameters)
//       .def("predict", &SparseGP_DTC::predict)
//       .def("predict_on_structure", &SparseGP_DTC::predict_on_structure)
//       .def("add_sparse_environments", &SparseGP_DTC::add_sparse_environments)
//       .def("add_training_structure", &SparseGP_DTC::add_training_structure)
//       .def("update_matrices", &SparseGP_DTC::update_matrices)
//       .def("update_matrices_QR", &SparseGP_DTC::update_matrices_QR)
//       .def("compute_likelihood", &SparseGP_DTC::compute_likelihood)
//       .def("compute_likelihood_gradient",
//            &SparseGP_DTC::compute_likelihood_gradient)
//       .def("compute_beta", &SparseGP_DTC::compute_beta)
//       .def("write_beta", &SparseGP_DTC::write_beta)
//       .def_readwrite("Kuu_jitter", &SparseGP::Kuu_jitter)
//       .def_readonly("complexity_penalty", &SparseGP_DTC::complexity_penalty)
//       .def_readonly("data_fit", &SparseGP_DTC::data_fit)
//       .def_readonly("constant_term", &SparseGP_DTC::constant_term)
//       .def_readwrite("log_marginal_likelihood",
//                      &SparseGP_DTC::log_marginal_likelihood)
//       .def_readonly("is_positive", &SparseGP_DTC::is_positive)
//       .def_readwrite("likelihood_gradient", &SparseGP_DTC::likelihood_gradient)
//       .def_readonly("hyperparameters", &SparseGP_DTC::hyperparameters)
//       .def_readonly("training_structures", &SparseGP::training_structures)
//       .def_readonly("n_energy_labels", &SparseGP_DTC::n_energy_labels)
//       .def_readonly("n_force_labels", &SparseGP_DTC::n_force_labels)
//       .def_readonly("n_stress_labels", &SparseGP_DTC::n_stress_labels)
//       .def_readonly("sigma_f", &SparseGP_DTC::sigma_f)
//       .def_readonly("sigma_e", &SparseGP_DTC::sigma_e)
//       .def_readonly("sigma_s", &SparseGP_DTC::sigma_s)
//       .def_readonly("Kuu", &SparseGP::Kuu)
//       .def_readonly("Kuf_struc", &SparseGP::Kuf_struc);
}
