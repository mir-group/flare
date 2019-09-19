"""Sparse Gaussian process regression."""
import numpy as np
import flare.cutoffs as cf


class SparseGP:
    """Sparse Gaussian process regression model."""

    def __init__(self, environment_kernel, structure_kernel,
                 kernel_hyps, noise_hyps, cutoffs,
                 cutoff_func=cf.quadratic_cutoff):
        self.environment_kernel = environment_kernel
        self.structure_kernel = structure_kernel
        self.kernel_hyps = kernel_hyps
        self.noise_hyps = noise_hyps
        self.cutoffs = cutoffs
        self.cutoff_func = cutoff_func

        self.sparse_environments = []
        self.training_structures = []

        self.k_mm = np.zeros((0, 0))
        self.k_nm = np.zeros((0, 0))
        self.noise_matrix = np.zeros((0, 0))
        self.training_labels = np.zeros(0)
        self.alpha = np.zeros(0)

    def add_sparse_point(self, atomic_env):
        """Adds a sparse point to the GP model. Adds a row and column to the \
sparse covariance matrix K_mm and a column to the dataset covariance matrix \
K_nm.

        :param atomic_env: Atomic environment of the sparse point.
        :type atomic_env: env.AtomicEnvironment
        """
        # update list of sparse environments
        self.sparse_environments.append(atomic_env)

        # add a row and column to k_mm
        prev_mm_size = self.k_mm.shape[0]
        k_mm_updated = np.zeros((prev_mm_size+1, prev_mm_size+1))
        k_mm_updated[:prev_mm_size, :prev_mm_size] = self.k_mm

        for count, sparse_env in enumerate(self.sparse_environments):
            energy_kern = self.environment_kernel(atomic_env, sparse_env,
                                                  self.kernel_hyps,
                                                  self.cutoffs,
                                                  self.cutoff_func)
            k_mm_updated[prev_mm_size, count] = energy_kern
            k_mm_updated[count, prev_mm_size] = energy_kern

        self.k_mm = k_mm_updated

        # add a column to k_nm
        prev_nm_size = self.k_nm.shape
        k_nm_updated = np.zeros((prev_nm_size[0], prev_nm_size[1] + 1))

        index = 0
        for train_struc in self.training_structures:
            struc_kern = self.structure_kernel(atomic_env, train_struc,
                                               self.kernel_hyps,
                                               self.cutoffs,
                                               self.cutoff_func)
            kern_len = len(struc_kern)
            k_nm_updated[index:index+kern_len, prev_nm_size[1]+1] = \
                struc_kern
            index += kern_len

        self.k_nm = k_nm_updated

    def add_structure(self, structure):
        """Adds a training structure to the GP database.

        :param structure: Structure of atoms added to the database.
        :type structure: struc.Structure
        """

        # update list of training structures
        self.training_structures.append(structure)

        # update training labels
        self.training_labels = np.append(self.training_labels,
                                         structure.labels)

        # update noise matrix
        self.update_noise_matrix(structure)

        # update k_nm
        # number of rows added equals the number of labels on the structure
        prev_nm_size = self.k_nm.shape
        label_size = len(structure.labels)
        k_nm_updated = \
            np.zeros((prev_nm_size[0] + label_size, prev_nm_size[1]))
        k_nm_updated[:prev_nm_size[0], :prev_nm_size[1]] = self.k_nm

        for count, sparse_env in enumerate(self.sparse_environments):
            struc_kern = self.structure_kernel(sparse_env, structure,
                                               self.kernel_hyps, self.cutoffs,
                                               self.cutoff_func)
            k_nm_updated[prev_nm_size[0]:, count] = struc_kern

        self.k_nm = k_nm_updated

    def update_noise_matrix(self, structure):
        noise_flattened = np.diag(self.noise_matrix)

        if structure.energy is not None:
            noise_flattened = np.append(noise_flattened, self.noise_hyps[0])

        if structure.forces is not None:
            force_noise = np.array([self.noise_hyps[1]] * 3 * structure.nat)
            noise_flattened = np.append(noise_flattened, force_noise)

        self.noise_matrix = np.diag(noise_flattened)

    def set_alpha(self):
        """Computes alpha using the current covariance and noise matrices."""

        mat1 = np.matmul(np.transpose(self.k_nm), self.noise_matrix)
        mat2 = np.matmul(mat1, self.k_nm)
        mat3 = np.linalg.inv(self.k_mm + mat2)
        mat4 = np.matmul(mat1, self.training_labels)

        alpha = np.matmul(mat3, mat4)

        self.alpha = alpha

    def predict_on_structure(self, structure):
        """Predict energy and forces of a structure with the current sparse \
GP model."""

        # set structure labels to true to compute kernels
        structure.energy = True
        structure.forces = True

        kernel_array = np.zeros((len(self.sparse_environments),
                                1 + 3 * structure.nat))
        for count, sparse_env in enumerate(self.sparse_environments):
            kernel_array[count] = \
                self.structure_kernel(sparse_env, structure, self.kernel_hyps,
                                      self.cutoffs, self.cutoff_func)

        prediction = np.matmul(self.alpha, kernel_array)

        return prediction
