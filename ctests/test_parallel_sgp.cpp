#include "parallel_sgp.h"
#include "sparse_gp.h"
#include "test_structure.h"
#include "omp.h"
#include "mpi.h"
#include <thread>
#include <chrono>
#include <numeric> // Iota
#include <blacs.h>

class ParSGPTest : public StructureTest {
public:
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;
  int n_atoms_1 = 10;
  int n_atoms_2 = 17;
  int n_atoms = 10;      
  int n_types = n_species;
  std::vector<Kernel *> kernels;
  SparseGP sparse_gp;
  std::vector<Structure> training_strucs;
  std::vector<std::vector<std::vector<int>>> sparse_indices;

  ParSGPTest() {
    blacs::initialize();

    std::vector<Descriptor *> dc;
    B2 ps(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
          descriptor_settings);
    dc.push_back(&ps);
    B2 ps1(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
          descriptor_settings);
    dc.push_back(&ps1);

    kernels.push_back(&kernel_norm);
    kernels.push_back(&kernel_3_norm);
    sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

    // Generate random labels
    Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
    Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
    Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);

    // Broadcast data such that different procs won't generate different random numbers
    MPI_Bcast(energy.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(forces.data(), n_atoms *  3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(stresses.data(), 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    test_struc = Structure(cell, species, positions, cutoff, dc);
    test_struc.energy = energy;
    test_struc.forces = forces;
    test_struc.stresses = stresses;

    // Make positions.
    Eigen::MatrixXd cell_1, cell_2;
    std::vector<int> species_1, species_2;
    Eigen::MatrixXd positions_1, positions_2;
    Eigen::VectorXd labels_1, labels_2;

    cell_1 = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    cell_2 = Eigen::MatrixXd::Identity(3, 3) * cell_size;

    positions_1 = Eigen::MatrixXd::Random(n_atoms_1, 3) * cell_size / 2;
    positions_2 = Eigen::MatrixXd::Random(n_atoms_2, 3) * cell_size / 2;
    MPI_Bcast(positions_1.data(), n_atoms_1 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(positions_2.data(), n_atoms_2 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    labels_1 = Eigen::VectorXd::Random(1 + n_atoms_1 * 3 + 6);
    labels_2 = Eigen::VectorXd::Random(1 + n_atoms_2 * 3 + 6);
    MPI_Bcast(labels_1.data(), 1 + n_atoms_1 * 3 + 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(labels_2.data(), 1 + n_atoms_2 * 3 + 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Make random species.
    for (int i = 0; i < n_atoms_1; i++) {
      species_1.push_back(rand() % n_species);
    }
    for (int i = 0; i < n_atoms_2; i++) {
      species_2.push_back(rand() % n_species);
    }
    MPI_Bcast(species_1.data(), n_atoms_1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(species_2.data(), n_atoms_2, MPI_INT, 0, MPI_COMM_WORLD);

    // Build kernel matrices for paralle sgp
    //std::vector<std::vector<std::vector<int>>> sparse_indices = {{{0, 1}, {2}}}; 
    //std::vector<std::vector<int>> comm_sparse_ind = {{0, 1, 5, 7}, {2, 3, 4}}; 
    std::vector<std::vector<int>> comm_sparse_ind = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 4, 5, 6, 7, 8, 9}};
    sparse_indices = {comm_sparse_ind, comm_sparse_ind};

    std::cout << "Start building" << std::endl;
    Structure struc_1 = Structure(cell_1, species_1, positions_1);
    struc_1.energy = labels_1.segment(0, 1);
    struc_1.forces = labels_1.segment(1, n_atoms_1 * 3);
    struc_1.stresses = labels_1.segment(1 + n_atoms_1 * 3, 6);
    std::cout << "Done struc_1" << std::endl;

    Structure struc_2 = Structure(cell_2, species_2, positions_2);
    struc_2.energy = labels_2.segment(0, 1);
    struc_2.forces = labels_2.segment(1, n_atoms_2 * 3);
    struc_2.stresses = labels_2.segment(1 + n_atoms_2 * 3, 6);
    std::cout << "Done struc_2" << std::endl;

    training_strucs = {struc_1, struc_2};

    if (blacs::mpirank == 0) {
      // Build sparse_gp (non parallel)
      Structure train_struc_1 = Structure(cell_1, species_1, positions_1, cutoff, dc);
      train_struc_1.energy = labels_1.segment(0, 1);
      train_struc_1.forces = labels_1.segment(1, n_atoms_1 * 3);
      train_struc_1.stresses = labels_1.segment(1 + n_atoms_1 * 3, 6);
    
      Structure train_struc_2 = Structure(cell_2, species_2, positions_2, cutoff, dc);
      train_struc_2.energy = labels_2.segment(0, 1);
      train_struc_2.forces = labels_2.segment(1, n_atoms_2 * 3);
      train_struc_2.stresses = labels_2.segment(1 + n_atoms_2 * 3, 6);
    
      sparse_gp.add_training_structure(train_struc_1);
      sparse_gp.add_specific_environments(train_struc_1, {sparse_indices[0][0], sparse_indices[1][0]});
      sparse_gp.add_training_structure(train_struc_2);
      sparse_gp.add_specific_environments(train_struc_2, {sparse_indices[0][1], sparse_indices[1][1]});
      sparse_gp.update_matrices_QR();
      std::cout << "Done QR for sparse_gp" << std::endl;
    
      sparse_gp.write_mapping_coefficients("beta.txt", "Me", 0);
      sparse_gp.write_varmap_coefficients("beta_var.txt", "Me", 0);
    } 
  }
};

TEST_F(ParSGPTest, BuildParSGP){
  std::vector<Descriptor *> dc;
  B2 ps(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps);
  B2 ps1(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps1);

  ParallelSGP parallel_sgp(kernels, sigma_e, sigma_f, sigma_s);
  parallel_sgp.build(training_strucs, cutoff, dc, sparse_indices, n_types);

  // Compute likelihood and gradient
  parallel_sgp.precompute_KnK();
  double likelihood_parallel = parallel_sgp.compute_likelihood_gradient_stable(true);
  Eigen::VectorXd like_grad_parallel = parallel_sgp.likelihood_gradient;

  // Build a serial SparsGP and compare with ParallelSGP
  if (blacs::mpirank == 0) {
    parallel_sgp.write_mapping_coefficients("par_beta.txt", "Me", 0);
    parallel_sgp.write_varmap_coefficients("par_beta_var.txt", "Me", 0);
  
    // Check the kernel matrices are consistent
    int n_clusters = 0;
    for (int i = 0; i < parallel_sgp.n_kernels; i++) {
      n_clusters += parallel_sgp.sparse_descriptors[i].n_clusters;
    }
    EXPECT_EQ(n_clusters, sparse_gp.Sigma.rows());
    EXPECT_EQ(parallel_sgp.sparse_descriptors[0].n_clusters, sparse_gp.sparse_descriptors[0].n_clusters);
    for (int t = 0; t < parallel_sgp.sparse_descriptors[0].n_types; t++) {
      for (int r = 0; r < parallel_sgp.sparse_descriptors[0].descriptors[t].rows(); r++) {
        for (int c = 0; c < parallel_sgp.sparse_descriptors[0].descriptors[t].cols(); c++) {
          double par_desc = parallel_sgp.sparse_descriptors[0].descriptors[t](r, c);
          double sgp_desc = sparse_gp.sparse_descriptors[0].descriptors[t](r, c);
          EXPECT_NEAR(par_desc, sgp_desc, 1e-6);
        }
      }
    }
 
    for (int r = 0; r < parallel_sgp.Kuu.rows(); r++) {
      for (int c = 0; c < parallel_sgp.Kuu.cols(); c++) {
        // Sometimes the accuracy is between 1e-6 ~ 1e-5        
        EXPECT_NEAR(parallel_sgp.Kuu(r, c), sparse_gp.Kuu(r, c), 1e-6);
      }
    }
    std::cout << "Kuu matches" << std::endl;
 
    for (int r = 0; r < parallel_sgp.Kuu_inverse.rows(); r++) {
      for (int c = 0; c < parallel_sgp.Kuu_inverse.rows(); c++) {
        // Sometimes the accuracy is between 1e-6 ~ 1e-5        
        EXPECT_NEAR(parallel_sgp.Kuu_inverse(r, c), sparse_gp.Kuu_inverse(r, c), 1e-5);
      }
    }
    std::cout << "Kuu_inverse matches" << std::endl;
  
    for (int r = 0; r < parallel_sgp.alpha.size(); r++) {
      EXPECT_NEAR(parallel_sgp.alpha(r), sparse_gp.alpha(r), 1e-6);
    }
    std::cout << "alpha matches" << std::endl;

    // Compare predictions on testing structure are consistent
    parallel_sgp.predict_local_uncertainties(test_struc);
    Structure test_struc_copy(test_struc.cell, test_struc.species, test_struc.positions, cutoff, dc);
    sparse_gp.predict_local_uncertainties(test_struc_copy);
  
    for (int r = 0; r < test_struc.mean_efs.size(); r++) {
      EXPECT_NEAR(test_struc.mean_efs(r), test_struc_copy.mean_efs(r), 1e-5);
    }
    std::cout << "mean_efs matches" << std::endl;
  
    for (int i = 0; i < test_struc.local_uncertainties.size(); i++) {
      for (int r = 0; r < test_struc.local_uncertainties[i].size(); r++) {
        EXPECT_NEAR(test_struc.local_uncertainties[i](r), test_struc_copy.local_uncertainties[i](r), 1e-5);
      }
    }

    // Test likelihood & gradient
    double likelihood_serial = sparse_gp.compute_likelihood_gradient_stable(false);
    std::cout << "likelihood: " << likelihood_serial << " " << likelihood_parallel << std::endl;
    EXPECT_NEAR(likelihood_serial, likelihood_parallel, 1e-6);

    Eigen::VectorXd like_grad_serial = sparse_gp.likelihood_gradient;
    EXPECT_EQ(like_grad_serial.size(), like_grad_parallel.size());
    for (int i = 0; i < like_grad_serial.size(); i++) {
      EXPECT_NEAR(like_grad_serial(i), like_grad_parallel(i), 1e-6);
      std::cout << "like grad " << like_grad_serial(i) << " " << like_grad_parallel(i) << std::endl;
    }

  }
  parallel_sgp.finalize_MPI = false;
}

TEST_F(ParSGPTest, UpdateTrainSet){
  std::vector<Descriptor *> dc;
  B2 ps(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps);
  B2 ps1(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps1);

  ParallelSGP parallel_sgp_1(kernels, sigma_e, sigma_f, sigma_s);
  bool update = false;
  parallel_sgp_1.build(training_strucs, cutoff, dc, sparse_indices, n_types, update);

  // Make positions.
  int n_atoms_3 = 11;
  Eigen::MatrixXd cell_3, positions_3;
  std::vector<int> species_3;
  Eigen::VectorXd labels_3;

  cell_3 = Eigen::MatrixXd::Identity(3, 3) * cell_size;

  positions_3 = Eigen::MatrixXd::Random(n_atoms_3, 3) * cell_size / 2;
  MPI_Bcast(positions_3.data(), n_atoms_3 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  labels_3 = Eigen::VectorXd::Random(1 + n_atoms_3 * 3 + 6);
  MPI_Bcast(labels_3.data(), 1 + n_atoms_3 * 3 + 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Make random species.
  for (int i = 0; i < n_atoms_3; i++) {
    species_3.push_back(rand() % n_species);
  }
  MPI_Bcast(species_3.data(), n_atoms_3, MPI_INT, 0, MPI_COMM_WORLD);

  // Build kernel matrices for paralle sgp
  //std::vector<std::vector<int>> comm_sparse_ind = {{0, 1, 5, 7}, {2, 3, 4}}; 
  std::vector<std::vector<int>> comm_sparse_ind = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 4, 5, 6, 7, 8, 9}};
  //sparse_indices = {comm_sparse_ind, comm_sparse_ind};

  std::cout << "Start building" << std::endl;
  Structure struc_3 = Structure(cell_3, species_3, positions_3);
  struc_3.energy = labels_3.segment(0, 1);
  struc_3.forces = labels_3.segment(1, n_atoms_3 * 3);
  struc_3.stresses = labels_3.segment(1 + n_atoms_3 * 3, 6);
  std::cout << "Done struc_3" << std::endl;

  training_strucs.push_back(struc_3);
  for (int i = 0; i < dc.size(); i++)
    sparse_indices[i].push_back(comm_sparse_ind[i]);

  update = true;
  parallel_sgp_1.build(training_strucs, cutoff, dc, sparse_indices, n_types, update);
  parallel_sgp_1.finalize_MPI = false;

  parallel_sgp_1.precompute_KnK();
  double likelihood_1 = parallel_sgp_1.compute_likelihood_gradient_stable(true);
  Eigen::VectorXd like_grad_1 = parallel_sgp_1.likelihood_gradient;

  ParallelSGP parallel_sgp_2(kernels, sigma_e, sigma_f, sigma_s);
  parallel_sgp_2.build(training_strucs, cutoff, dc, sparse_indices, n_types, false);
  parallel_sgp_2.finalize_MPI = false;

  parallel_sgp_2.precompute_KnK();
  double likelihood_2 = parallel_sgp_2.compute_likelihood_gradient_stable(true);
  Eigen::VectorXd like_grad_2 = parallel_sgp_2.likelihood_gradient;

  if (blacs::mpirank == 0) {
    // Compare predictions on testing structure are consistent
    Structure test_struc_1(test_struc.cell, test_struc.species, test_struc.positions, cutoff, dc);
    parallel_sgp_1.predict_local_uncertainties(test_struc_1);
    Structure test_struc_2(test_struc.cell, test_struc.species, test_struc.positions, cutoff, dc);
    parallel_sgp_2.predict_local_uncertainties(test_struc_2);
  
    for (int r = 0; r < test_struc_1.mean_efs.size(); r++) {
      EXPECT_NEAR(test_struc_1.mean_efs(r), test_struc_2.mean_efs(r), 1e-6);
    }
    std::cout << "parallel_sgp_1 and parallel_sgp_2 mean_efs matches" << std::endl;

    for (int i = 0; i < test_struc_1.local_uncertainties.size(); i++) {
      for (int r = 0; r < test_struc_1.local_uncertainties[i].size(); r++) {
        EXPECT_NEAR(test_struc_1.local_uncertainties[i](r), test_struc_2.local_uncertainties[i](r), 1e-6);
      }
    }

    EXPECT_NEAR(likelihood_1, likelihood_2, 1e-6);  
    EXPECT_EQ(like_grad_1.size(), like_grad_2.size());
    for (int i = 0; i < like_grad_1.size(); i++) {
      EXPECT_NEAR(like_grad_1(i), like_grad_2(i), 1e-6);
    }
  }

}

TEST_F(ParSGPTest, ParPredict){
  std::vector<Descriptor *> dc;
  B2 ps(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps);
  B2 ps1(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps1);

  ParallelSGP parallel_sgp(kernels, sigma_e, sigma_f, sigma_s);
  parallel_sgp.build(training_strucs, cutoff, dc, sparse_indices, n_types);

  std::cout << "create testing structures" << std::endl;
  //std::vector<std::shared_ptr<Structure>> test_strucs_par;
  std::vector<Structure> test_strucs_par;
  std::vector<Structure> test_strucs_ser;
  for (int t = 0; t < 10; t++) {
    Eigen::MatrixXd cell_1, positions_1;
    std::vector<int> species_1;
    Eigen::VectorXd labels_1;

    cell_1 = Eigen::MatrixXd::Identity(3, 3) * cell_size;

    positions_1 = Eigen::MatrixXd::Random(n_atoms_1, 3) * cell_size / 2;
    MPI_Bcast(positions_1.data(), n_atoms_1 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    labels_1 = Eigen::VectorXd::Random(1 + n_atoms_1 * 3 + 6);
    MPI_Bcast(labels_1.data(), 1 + n_atoms_1 * 3 + 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Make random species.
    for (int i = 0; i < n_atoms_1; i++) {
      species_1.push_back(rand() % n_species);
    }
    MPI_Bcast(species_1.data(), n_atoms_1, MPI_INT, 0, MPI_COMM_WORLD);

    blacs::barrier();

    //auto test_struc_1 = std::make_shared<Structure>(cell_1, species_1, positions_1);
    Structure test_struc_1(cell_1, species_1, positions_1);
    test_strucs_par.push_back(test_struc_1);    

    // Predict with serial SGP
    Structure test_struc_2(cell_1, species_1, positions_1, cutoff, dc);
    test_strucs_ser.push_back(test_struc_2);

    blacs::barrier();
  }

  test_strucs_par = parallel_sgp.predict_on_structures(test_strucs_par, cutoff, dc);
  if (blacs::mpirank == 0) {
    for (int t = 0; t < test_strucs_par.size(); t++) {
      sparse_gp.predict_local_uncertainties(test_strucs_ser[t]);
      for (int r = 0; r < test_strucs_ser[t].mean_efs.size(); r++) {
        EXPECT_NEAR(test_strucs_par[t].mean_efs(r), test_strucs_ser[t].mean_efs(r), 1e-6);
        std::cout << test_strucs_par[t].mean_efs(r) << " " << test_strucs_ser[t].mean_efs(r) << std::endl;
      }
      for (int a = 0; a < test_strucs_ser[t].noa; a++) {
        for (int i = 0; i < dc.size(); i++) {
          EXPECT_NEAR(test_strucs_par[t].local_uncertainties[i](a), test_strucs_ser[t].local_uncertainties[i](a), 1e-6);
        }
      }
    }
  }
}
