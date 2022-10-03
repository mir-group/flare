#ifndef PARALLEL_SGP_H
#define PARALLEL_SGP_H

#include "descriptor.h"
#include "kernel.h"
#include "structure.h"
#include "sparse_gp.h"
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"
#include "utils.h"

class ParallelSGP : public SparseGP {
public:
  // Training and sparse points.
  std::vector<std::vector<ClusterDescriptor>> local_sparse_descriptors;
  std::vector<std::vector<int>> local_label_indices;
  int local_label_size;

  // Parallel parameters
  int u_size, u_size_per_proc; 
  std::vector<int> u_size_single_kernel;
  int f_size, f_size_single_kernel, f_size_per_proc;
  int nmin_struc, nmax_struc, nmin_envs, nmax_envs;
  std::vector<std::vector<Eigen::VectorXi>> n_struc_clusters_by_type;
  int global_n_labels = 0;
  int global_n_energy_labels = 0;
  int global_n_force_labels = 0;
  int global_n_stress_labels = 0;
  std::vector<int> local_training_structure_indices;
  bool finalize_MPI = true;

  utils::Timer timer;

  Eigen::MatrixXd Kuf_local;

  // Constructors.
  ParallelSGP();

  /**
   Basic Parallel Sparse GP constructor. This class inherits from SparseGP class and accept
   the same input parameters.

   @param kernels A list of Kernel objects, e.g. NormalizedInnerProduct, SquaredExponential.
        Note the number of kernels should be equal to the number of descriptor calculators.
   @param energy_noise Noise hyperparameter for total energy.
   @param force_noise Noise hyperparameter for atomic forces.
   @param stress_noise Noise hyperparameter for total stress.
   */
  ParallelSGP(std::vector<Kernel *> kernels, double energy_noise,
           double force_noise, double stress_noise);

  // Destructor
  virtual ~ParallelSGP();
  
  void add_global_noise(int n_energy, int n_force, int n_stress); 
  Eigen::VectorXd global_noise_vector, local_noise_vector, local_e_noise_one, local_f_noise_one, local_s_noise_one;
  Eigen::MatrixXd local_labels;
  void add_training_structure(const Structure &structure);
  
  Eigen::VectorXi sparse_indices_by_type(int n_types, std::vector<int> species, const std::vector<int> atoms);    
  void add_specific_environments(const Structure&, std::vector<std::vector<int>>, bool update);
  void add_local_specific_environments(const Structure &structure, const std::vector<int> atoms);
  void add_global_specific_environments(const Structure &structure, const std::vector<int> atoms);
  void predict_local_uncertainties(Structure &structure);

  std::vector<Structure> predict_on_structures(std::vector<Structure> struc_list,
        double cutoff, std::vector<Descriptor *> descriptor_calculators);

  /**
   Method for constructing SGP model from training dataset.  

   @param training_strucs A list of Structure objects
   @param cutoff The cutoff for SGP
   @param descriptor_calculators A list of Descriptor objects, e.g. B2, B3, ...
   @param trianing_sparse_indices A list of indices of sparse environments in each training structure
   @param n_types An integer to specify number of types. For B2 descriptor, n_type is equal to the
        number of species
   */
  void build(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types, bool update = false);

  /**
   Method for loading training data distributedly. Each process loads a portion of the whole training
   data set, and load the whole sparse set.

   @param training_strucs A list of Structure objects
   @param cutoff The cutoff for SGP
   @param descriptor_calculators A list of Descriptor objects, e.g. B2, B3, ...
   @param trianing_sparse_indices A list of indices of sparse environments in each training structure
   @param n_types An integer to specify number of types. For B2 descriptor, n_type is equal to the
        number of species
   */
  void load_local_training_data(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types, bool update = false);

  void gather_sparse_descriptors(std::vector<std::vector<int>> n_clusters_by_type,
        const std::vector<Structure> &training_strucs);

  /**
   Method for computing kernel matrices and vectors
 
   @param training_strucs A list of Structure objects
   */
  void compute_kernel_matrices(const std::vector<Structure> &training_strucs);

  void set_hyperparameters(Eigen::VectorXd hyps);
  void stack_Kuf();
  void update_matrices_QR();

  Eigen::MatrixXd varmap_coeffs; // for debugging. TODO: remove this line 

  double compute_likelihood_gradient_stable(bool precomputed_KnK);
  Eigen::VectorXd y_K_alpha;
  void compute_likelihood_stable();
  void compute_KnK(bool precomputed_KnK);
  void precompute_KnK();
  Eigen::VectorXd compute_like_grad_of_kernel_hyps();
  Eigen::VectorXd compute_like_grad_of_noise(bool precomputed_KnK);

};
#endif
