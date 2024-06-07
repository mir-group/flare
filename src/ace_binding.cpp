#include "kernel.h"
#include "structure.h"
#include "y_grad.h"
#include "sparse_gp.h"
#include "b2.h"
#include "b2_simple.h"
#include "b2_norm.h"
#include "b3.h"
#include "two_body.h"
#include "three_body.h"
#include "three_body_wide.h"
#include "four_body.h"
#include "squared_exponential.h"
#include "normalized_dot_product.h"
#include "dot_product.h"
#include "norm_dot_icm.h"

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
      .def_readwrite("noa", &Structure::noa)
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
      .def_readwrite("local_uncertainties", &Structure::local_uncertainties)
      .def_readwrite("descriptors", &Structure::descriptors)
      .def_readwrite("descriptor_calculators",
                    &Structure::descriptor_calculators)
      .def("compute_descriptors", &Structure::compute_descriptors)
      .def("wrap_positions", &Structure::wrap_positions)
      .def_static("to_json", &Structure::to_json)
      .def_static("from_json", &Structure::from_json);

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
      .def_readwrite("cumulative_type_count",
                     &DescriptorValues::cumulative_type_count)
      .def_readwrite("atom_indices", &DescriptorValues::atom_indices)
      .def_readwrite("neighbor_indices", &DescriptorValues::neighbor_indices)
      .def_readwrite("n_clusters_by_type",
                     &DescriptorValues::n_clusters_by_type)
      .def_readwrite("n_neighbors_by_type",
                     &DescriptorValues::n_neighbors_by_type);

  py::class_<ClusterDescriptor>(m, "ClusterDescriptor")
      .def_readonly("descriptors", &ClusterDescriptor::descriptors)
      .def_readonly("descriptor_norms", &ClusterDescriptor::descriptor_norms);

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
                    const std::vector<int> &>())
      .def(py::init<const std::string &, const std::string &,
                    const std::vector<double> &, const std::vector<double> &,
                    const std::vector<int> &,
                    const Eigen::MatrixXd &>())
      .def_readonly("radial_basis", &B2::radial_basis)
      .def_readonly("cutoff_function", &B2::cutoff_function)
      .def_readonly("radial_hyps", &B2::radial_hyps)
      .def_readonly("cutoff_hyps", &B2::cutoff_hyps)
      .def_readonly("cutoffs", &B2::cutoffs)
      .def_readonly("descriptor_settings", &B2::descriptor_settings);

  py::class_<B2_Simple, Descriptor>(m, "B2_Simple")
      .def(py::init<const std::string &, const std::string &,
                    const std::vector<double> &, const std::vector<double> &,
                    const std::vector<int> &>());

  py::class_<B2_Norm, Descriptor>(m, "B2_Norm")
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
      .def(py::init<double, double>())
      .def_readonly("sigma", &NormalizedDotProduct::sigma)
      .def_readwrite("power", &NormalizedDotProduct::power)
      .def_readonly("kernel_hyperparameters",
                    &NormalizedDotProduct::kernel_hyperparameters)
      .def("envs_envs", &NormalizedDotProduct::envs_envs)
      .def("envs_struc", &NormalizedDotProduct::envs_struc)
      .def("struc_struc", &NormalizedDotProduct::struc_struc);

  py::class_<DotProduct, Kernel>(m, "DotProduct")
      .def(py::init<double, double>())
      .def_readonly("sigma", &DotProduct::sigma)
      .def_readwrite("power", &DotProduct::power)
      .def_readonly("kernel_hyperparameters",
                    &DotProduct::kernel_hyperparameters)
      .def("envs_envs", &DotProduct::envs_envs)
      .def("envs_struc", &DotProduct::envs_struc)
      .def("struc_struc", &DotProduct::struc_struc);

  py::class_<NormalizedDotProduct_ICM, Kernel>(m, "NormalizedDotProduct_ICM")
      .def(py::init<double, double, Eigen::MatrixXd>());

  py::class_<SquaredExponential, Kernel>(m, "SquaredExponential")
      .def(py::init<double, double>());

  // Sparse GP DTC
  py::class_<SparseGP>(m, "SparseGP")
      .def(py::init<>())
      .def(py::init<std::vector<Kernel *>, double, double, double>())
      .def("set_hyperparameters", &SparseGP::set_hyperparameters)
      .def("predict_mean", &SparseGP::predict_mean)
      .def("predict_SOR", &SparseGP::predict_SOR)
      .def("predict_DTC", &SparseGP::predict_DTC)
      .def("predict_local_uncertainties",
           &SparseGP::predict_local_uncertainties)
      .def("add_all_environments", &SparseGP::add_all_environments)
      .def("add_specific_environments", &SparseGP::add_specific_environments)
      .def("add_random_environments", &SparseGP::add_random_environments)
      .def("add_uncertain_environments",
           &SparseGP::add_uncertain_environments)
      .def("add_training_structure", &SparseGP::add_training_structure,
                       py::arg("structure"),
                       py::arg("atom_indices") = - Eigen::VectorXi::Ones(1),
                       py::arg("rel_e_noise") = 1.0,
                       py::arg("rel_f_noise") = 1.0,
                       py::arg("rel_s_noise") = 1.0)
      .def("update_matrices_QR", &SparseGP::update_matrices_QR)
      .def("compute_likelihood", &SparseGP::compute_likelihood)
      .def("compute_likelihood_stable", &SparseGP::compute_likelihood_stable)
      .def("compute_likelihood_gradient",
           &SparseGP::compute_likelihood_gradient)
      .def("compute_likelihood_gradient_stable",
           &SparseGP::compute_likelihood_gradient_stable)
      .def("precompute_KnK", &SparseGP::precompute_KnK)
      .def("write_mapping_coefficients", &SparseGP::write_mapping_coefficients)
      .def_readonly("varmap_coeffs", &SparseGP::varmap_coeffs) // for debugging and unit test
      .def("compute_cluster_uncertainties", &SparseGP::compute_cluster_uncertainties) // for debugging and unit test
      .def("write_varmap_coefficients", &SparseGP::write_varmap_coefficients)
      .def("write_sparse_descriptors", &SparseGP::write_sparse_descriptors)
      .def("write_L_inverse", &SparseGP::write_L_inverse)
      .def_readwrite("Kuu_jitter", &SparseGP::Kuu_jitter)
      .def_readonly("complexity_penalty", &SparseGP::complexity_penalty)
      .def_readonly("data_fit", &SparseGP::data_fit)
      .def_readonly("constant_term", &SparseGP::constant_term)
      .def_readwrite("log_marginal_likelihood",
                     &SparseGP::log_marginal_likelihood)
      .def_readwrite("likelihood_gradient", &SparseGP::likelihood_gradient)
      .def_readonly("kernels", &SparseGP::kernels)
      .def_readonly("hyperparameters", &SparseGP::hyperparameters)
      .def_readonly("training_structures", &SparseGP::training_structures)
      .def_readonly("sparse_indices", &SparseGP::sparse_indices)
      .def_readonly("sparse_descriptors", &SparseGP::sparse_descriptors)
      .def_readonly("n_energy_labels", &SparseGP::n_energy_labels)
      .def_readonly("n_force_labels", &SparseGP::n_force_labels)
      .def_readonly("n_stress_labels", &SparseGP::n_stress_labels)
      .def_readonly("force_noise", &SparseGP::force_noise)
      .def_readonly("energy_noise", &SparseGP::energy_noise)
      .def_readonly("stress_noise", &SparseGP::stress_noise)
      .def_readonly("noise_vector", &SparseGP::noise_vector)
      .def_readonly("Kuu", &SparseGP::Kuu)
      .def_readonly("Kuu_kernels", &SparseGP::Kuu_kernels)
      .def_readonly("Kuf", &SparseGP::Kuf)
      .def_readonly("Kuf_kernels", &SparseGP::Kuf_kernels)
      .def_readwrite("Kuf_e_noise_Kfu", &SparseGP::Kuf_e_noise_Kfu)
      .def_readwrite("Kuf_f_noise_Kfu", &SparseGP::Kuf_f_noise_Kfu)
      .def_readwrite("Kuf_s_noise_Kfu", &SparseGP::Kuf_s_noise_Kfu)
      .def_readonly("alpha", &SparseGP::alpha)
      .def_readonly("Kuu_inverse", &SparseGP::Kuu_inverse)
      .def_readonly("Sigma", &SparseGP::Sigma)
      .def_readonly("n_sparse", &SparseGP::n_sparse)
      .def_readonly("n_labels", &SparseGP::n_labels)
      .def_readonly("y", &SparseGP::y)
      .def_static("to_json", &SparseGP::to_json)
      .def_static("from_json", &SparseGP::from_json);
}
