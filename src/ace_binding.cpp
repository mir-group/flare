#include "descriptor.h"
#include "dot_product_kernel.h"
#include "kernels.h"
#include "local_environment.h"
#include "sparse_gp.h"
#include "sparse_gp_dtc.h"
#include "structure.h"
#include "three_body_kernel.h"
#include "two_body_kernel.h"
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

  py::class_<StructureDescriptor, Structure>(m, "StructureDescriptor")
      .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                    const Eigen::MatrixXd &, double>())
      // n-body
      .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                    const Eigen::MatrixXd &, double, std::vector<double>>())
      // many-body
      .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                    const Eigen::MatrixXd &, double, std::vector<double>,
                    std::vector<DescriptorCalculator *>>())
      // n-body + many-body
      .def(py::init<const Eigen::MatrixXd &, const std::vector<int> &,
                    const Eigen::MatrixXd &, double, std::vector<double>,
                    std::vector<double>, std::vector<DescriptorCalculator *>>())
      .def_readwrite("local_environments",
                     &StructureDescriptor::local_environments)
      .def_readwrite("energy", &StructureDescriptor::energy)
      .def_readwrite("forces", &StructureDescriptor::forces)
      .def_readwrite("stresses", &StructureDescriptor::stresses)
      .def_readwrite("cutoff", &StructureDescriptor::cutoff)
      .def_readwrite("n_body_cutoffs", &StructureDescriptor::n_body_cutoffs)
      .def_readwrite("many_body_cutoffs",
                     &StructureDescriptor::many_body_cutoffs);

  // Local environment
  py::class_<LocalEnvironment>(m, "LocalEnvironment")
      .def(py::init<const Structure &, int, double>())
      // n-body
      .def(py::init<const Structure &, int, double, std::vector<double>>())
      // many-body
      .def(py::init<const Structure &, int, double, std::vector<double>,
                    std::vector<DescriptorCalculator *>>())
      // n-body + many-body
      .def(py::init<const Structure &, int, double, std::vector<double>,
                    std::vector<double>, std::vector<DescriptorCalculator *>>())
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
      .def_readwrite("many_body_cutoffs", &LocalEnvironment::many_body_cutoffs)
      .def_readwrite("many_body_indices", &LocalEnvironment::many_body_indices)
      .def_readwrite("cutoff", &LocalEnvironment::cutoff)
      .def_readwrite("descriptor_vals", &LocalEnvironment::descriptor_vals)
      .def_readwrite("descriptor_force_dervs",
                     &LocalEnvironment::descriptor_force_dervs)
      .def_readwrite("descriptor_stress_dervs",
                     &LocalEnvironment::descriptor_stress_dervs)
      .def_readwrite("descriptor_norm", &LocalEnvironment::descriptor_norm)
      .def("compute_descriptors", &LocalEnvironment::compute_descriptors)
      .def("compute_neighbor_descriptors",
           &LocalEnvironment::compute_neighbor_descriptors)
      .def_readwrite("force", &LocalEnvironment::force);

  m.def("compute_neighbor_descriptors", &compute_neighbor_descriptors);
  m.def("compute_descriptors", &compute_descriptors);

  // Descriptor calculators
  py::class_<DescriptorCalculator>(m, "DescriptorCalculator")
      .def("compute", &DescriptorCalculator::compute)
      .def_readwrite("radial_basis", &DescriptorCalculator::radial_basis)
      .def_readwrite("cutoff_function", &DescriptorCalculator::cutoff_function)
      .def_readwrite("descriptor_vals", &DescriptorCalculator::descriptor_vals)
      .def_readwrite("descriptor_force_dervs",
                     &DescriptorCalculator::descriptor_force_dervs)
      .def_readwrite("descriptor_stress_dervs",
                     &DescriptorCalculator::descriptor_stress_dervs)
      .def_readwrite("radial_hyps", &DescriptorCalculator::radial_hyps)
      .def_readwrite("cutoff_hyps", &DescriptorCalculator::cutoff_hyps)
      .def_readwrite("descriptor_settings",
                     &DescriptorCalculator::descriptor_settings);

  py::class_<B1_Calculator, DescriptorCalculator>(m, "B1_Calculator")
      .def(py::init<const std::string &, const std::string &,
                    const std::vector<double> &, const std::vector<double> &,
                    const std::vector<int> &, int>());

  py::class_<B2_Calculator, DescriptorCalculator>(m, "B2_Calculator")
      .def(py::init<const std::string &, const std::string &,
                    const std::vector<double> &, const std::vector<double> &,
                    const std::vector<int> &, int>());

  // Kernel functions
  py::class_<Kernel>(m, "Kernel")
      .def("env_env", &Kernel::env_env)
      .def("env_struc", &Kernel::env_struc);

  py::class_<TwoBodyKernel, Kernel>(m, "TwoBodyKernel")
      .def(py::init<double, double, std::string, std::vector<double>>());

  py::class_<ThreeBodyKernel, Kernel>(m, "ThreeBodyKernel")
      .def(py::init<double, double, std::string, std::vector<double>>());

  py::class_<DotProductKernel, Kernel>(m, "DotProductKernel")
      .def(py::init<double, double, int>());

  // Sparse GP
  py::class_<SparseGP>(m, "SparseGP")
      .def(py::init<std::vector<Kernel *>, double, double, double>())
      .def_readwrite("Kuu", &SparseGP::Kuu)
      .def_readwrite("kernels", &SparseGP::kernels)
      .def_readwrite("alpha", &SparseGP::alpha)
      .def_readwrite("Kuu_jitter", &SparseGP::Kuu_jitter)
      .def_readwrite("hyperparameters", &SparseGP::hyperparameters)
      .def("add_sparse_environment", &SparseGP::add_sparse_environment)
      .def("add_sparse_environments", &SparseGP::add_sparse_environments)
      .def("add_training_structure", &SparseGP::add_training_structure)
      .def("add_training_environment", &SparseGP::add_training_environment)
      .def("add_training_environments", &SparseGP::add_training_environments)
      .def("update_alpha", &SparseGP::update_alpha)
      .def("update_alpha_CG", &SparseGP::update_alpha_CG)
      .def("update_alpha_LLT", &SparseGP::update_alpha_LLT)
      .def("update_alpha_LDLT", &SparseGP::update_alpha_LDLT)
      .def("predict", &SparseGP::predict)
      .def("predict_force", &SparseGP::predict_force)
      .def("compute_beta", &SparseGP::compute_beta)
      .def("write_beta", &SparseGP::write_beta)
      .def("clear_environment_lists", &SparseGP::clear_environment_lists)
      .def_readwrite("sparse_environments", &SparseGP::sparse_environments)
      .def_readwrite("training_structures", &SparseGP::training_structures)
      .def_readwrite("training_environments", &SparseGP::training_environments)
      .def_readwrite("Kuf_env", &SparseGP::Kuf_env);
    
  // Sparse GP DTC
  py::class_<SparseGP_DTC>(m, "SparseGP_DTC")
    .def(py::init<std::vector<Kernel *>, double, double, double>())
    .def("set_hyperparameters", &SparseGP_DTC::set_hyperparameters)
    .def("predict_DTC", &SparseGP_DTC::predict_DTC)
    .def("add_sparse_environments", &SparseGP_DTC::add_sparse_environments)
    .def("add_training_structure", &SparseGP_DTC::add_training_structure)
    .def("update_matrices", &SparseGP_DTC::update_matrices)
    .def("compute_likelihood", &SparseGP_DTC::compute_likelihood)
    .def("compute_likelihood_gradient",
        &SparseGP_DTC::compute_likelihood_gradient)
    .def_readonly("log_marginal_likelihood",
        &SparseGP_DTC::log_marginal_likelihood)
    .def_readonly("likelihood_gradient",
        &SparseGP_DTC::likelihood_gradient);
}
