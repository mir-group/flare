#include "kernel.h"
#include "structure.h"
#include "y_grad.h"
#include "sparse_gp_dtc.h"
#include "b2.h"
#include "b3.h"
#include "two_body.h"
#include "three_body.h"
#include "three_body_wide.h"
#include "four_body.h"
#include "squared_exponential.h"
#include "normalized_dot_product.h"

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
      .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                    const Eigen::MatrixXd &, double,
                    std::vector<Descriptor *>>())
      .def_readwrite("cell", &Structure::cell)
      .def_readwrite("species", &Structure::species)
      .def_readwrite("positions", &Structure::positions)
      .def_readwrite("cell_transpose", &Structure::cell_transpose)
      .def_readwrite("wrapped_positions", &Structure::wrapped_positions)
      .def_readwrite("volume", &Structure::volume)
      .def_readwrite("energy", &Structure::energy)
      .def_readwrite("forces", &Structure::forces)
      .def_readwrite("stresses", &Structure::stresses)
      .def_readwrite("mean_efs", &Structure::mean_efs)
      .def_readwrite("variance_efs", &Structure::variance_efs)
      .def_readonly("descriptors", &Structure::descriptors)
      .def("wrap_positions", &Structure::wrap_positions);

  // Descriptor values
  py::class_<DescriptorValues>(m, "DescriptorValues")
      .def(py::init<>())
      .def_readwrite("n_descriptors", &DescriptorValues::n_descriptors)
      .def_readwrite("n_types", &DescriptorValues::n_types)
      .def_readwrite("n_atoms", &DescriptorValues::n_atoms)
      .def_readwrite("volume", &DescriptorValues::volume)
      .def_readwrite("descriptors", &DescriptorValues::descriptors)
      .def_readwrite("descriptor_force_dervs",
                     &DescriptorValues::descriptor_force_dervs)
      .def_readwrite("neighbor_coordinates",
                     &DescriptorValues::neighbor_coordinates)
      .def_readwrite("descriptor_norms", &DescriptorValues::descriptor_norms)
      .def_readwrite("descriptor_force_dots",
                     &DescriptorValues::descriptor_force_dots)
      .def_readwrite("cutoff_values", &DescriptorValues::cutoff_values)
      .def_readwrite("cutoff_dervs", &DescriptorValues::cutoff_dervs)
      .def_readwrite("neighbor_counts", &DescriptorValues::neighbor_counts)
      .def_readwrite("cumulative_neighbor_counts",
                     &DescriptorValues::cumulative_neighbor_counts)
      .def_readwrite("atom_indices", &DescriptorValues::atom_indices)
      .def_readwrite("neighbor_indices", &DescriptorValues::neighbor_indices)
      .def_readwrite("n_clusters_by_type",
                     &DescriptorValues::n_clusters_by_type)
      .def_readwrite("n_neighbors_by_type",
                     &DescriptorValues::n_neighbors_by_type);

  // Descriptor calculators
  py::class_<Descriptor>(m, "Descriptor")
      .def("compute_struc", &Descriptor::compute_struc);

  py::class_<TwoBody, Descriptor>(m, "TwoBody")
      .def(py::init<double, int, const std::string &,
                    const std::vector<double> &>());

  py::class_<ThreeBody, Descriptor>(m, "ThreeBody")
      .def(py::init<double, int, const std::string &,
                    const std::vector<double> &>());

  py::class_<ThreeBodyWide, Descriptor>(m, "ThreeBodyWide")
      .def(py::init<double, int, const std::string &,
                    const std::vector<double> &>());

  py::class_<FourBody, Descriptor>(m, "FourBody")
      .def(py::init<double, int, const std::string &,
                    const std::vector<double> &>());

  py::class_<B2, Descriptor>(m, "B2")
      .def(py::init<const std::string &, const std::string &,
                    const std::vector<double> &, const std::vector<double> &,
                    const std::vector<int> &>());

  py::class_<B3, Descriptor>(m, "B3")
      .def(py::init<const std::string &, const std::string &,
                    const std::vector<double> &, const std::vector<double> &,
                    const std::vector<int> &>());

  // Kernel functions
  py::class_<Kernel>(m, "Kernel");

  py::class_<NormalizedDotProduct, Kernel>(m, "NormalizedDotProduct")
      .def(py::init<double, double>());

  py::class_<SquaredExponential, Kernel>(m, "SquaredExponential")
      .def(py::init<double, double>());

  // Sparse GP DTC
  py::class_<SparseGP_DTC>(m, "SparseGP_DTC")
      .def(py::init<>())
      .def(py::init<std::vector<Kernel *>, double, double, double>())
      .def("set_hyperparameters", &SparseGP_DTC::set_hyperparameters)
      .def("predict_SOR", &SparseGP_DTC::predict_SOR)
      .def("predict_DTC", &SparseGP_DTC::predict_DTC)
      .def("add_all_environments", &SparseGP_DTC::add_all_environments)
      .def("add_random_environments", &SparseGP_DTC::add_random_environments)
      .def("add_uncertain_environments",
           &SparseGP_DTC::add_uncertain_environments)
      .def("add_training_structure", &SparseGP_DTC::add_training_structure)
      .def("update_matrices_QR", &SparseGP_DTC::update_matrices_QR)
      .def("compute_likelihood", &SparseGP_DTC::compute_likelihood)
      .def("compute_likelihood_gradient",
           &SparseGP_DTC::compute_likelihood_gradient)
    //   .def("compute_beta", &SparseGP_DTC::compute_beta)
    //   .def("write_beta", &SparseGP_DTC::write_beta)
      .def_readwrite("Kuu_jitter", &SparseGP_DTC::Kuu_jitter)
      .def_readonly("complexity_penalty", &SparseGP_DTC::complexity_penalty)
      .def_readonly("data_fit", &SparseGP_DTC::data_fit)
      .def_readonly("constant_term", &SparseGP_DTC::constant_term)
      .def_readwrite("log_marginal_likelihood",
                     &SparseGP_DTC::log_marginal_likelihood)
      .def_readwrite("likelihood_gradient", &SparseGP_DTC::likelihood_gradient)
      .def_readonly("hyperparameters", &SparseGP_DTC::hyperparameters)
      .def_readonly("training_structures", &SparseGP_DTC::training_structures)
      .def_readonly("n_energy_labels", &SparseGP_DTC::n_energy_labels)
      .def_readonly("n_force_labels", &SparseGP_DTC::n_force_labels)
      .def_readonly("n_stress_labels", &SparseGP_DTC::n_stress_labels)
      .def_readonly("force_noise", &SparseGP_DTC::force_noise)
      .def_readonly("energy_noise", &SparseGP_DTC::energy_noise)
      .def_readonly("stress_noise", &SparseGP_DTC::stress_noise)
      .def_readonly("Kuu", &SparseGP_DTC::Kuu)
      .def_readonly("Kuf", &SparseGP_DTC::Kuf)
      .def_readonly("alpha", &SparseGP_DTC::alpha)
      .def_readonly("Kuu_inverse", &SparseGP_DTC::Kuu_inverse);
}
