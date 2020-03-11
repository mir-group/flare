import numpy as np
import ace
import pickle

# create two and three body kernels
kernel_2b = ace.TwoBodyKernel(0.02, 0.60, "quadratic", [])
kernel_3b = ace.ThreeBodyKernel(0.013, 0.2, "quadratic", [])

# create many body kernel
sigma = 0.1
power = 4
kernel_mb = ace.DotProductKernel(sigma, power)

# create descriptor calculator
radial_basis = "chebyshev"
cutoff_function = "quadratic"
cutoff = 4.
nested_cutoffs = [4., 4., 4.]
radial_hyps = [0, 4.]
cutoff_hyps = []
descriptor_settings = [5, 1, 1]
b2_calculator = \
    ace.B2_Calculator(radial_basis, cutoff_function, radial_hyps,
                      cutoff_hyps, descriptor_settings)

# create sparse gp
sigma_e = 0.1
sigma_f = 0.1
sigma_s = 0.1
sparse_gp = ace.SparseGP([kernel_3b, kernel_mb],
                         sigma_e, sigma_f, sigma_s)

# define species (based on poscar)
# C, F, H, O, S
species = [0] * 16 + [1] * 36 + [2] * 164 + [3] * 96 + [4] * 4

# prepare data
file_base = '/Users/jonpvandermause/Research/Bosch/10-4-19/Sulfonate_AIMD/'\
            'extract_run'
file_tail = '.pkl'
file_nos = [8, 1, 7, 2, 6, 3, 5]
snapshots = [100, 200, 300, 400, 500]
test_snapshot = 300

# create test structure
test_no = 4
data_file = '{}{}{}'.format(file_base, test_no, file_tail)
open_file = open(data_file, 'rb')
vasp_data = pickle.load(open_file)
cell = vasp_data['lattice'][test_snapshot]
positions = vasp_data['atoms'][test_snapshot]
test_forces = vasp_data['forces'][test_snapshot]
test_struc = \
    ace.StructureDescriptor(cell, species, positions, cutoff,
                            nested_cutoffs, b2_calculator)
test_struc.forces = test_forces.reshape(-1)


# train the model
for snapshot in snapshots:
    for file_no in file_nos:
        data_file = '{}{}{}'.format(file_base, file_no, file_tail)
        open_file = open(data_file, 'rb')
        vasp_data = pickle.load(open_file)
        cell = vasp_data['lattice'][snapshot]
        positions = vasp_data['atoms'][snapshot]
        forces = vasp_data['forces'][snapshot]
        train_struc = \
            ace.StructureDescriptor(cell, species, positions, cutoff,
                                    nested_cutoffs, b2_calculator)
        train_struc.forces = forces.reshape(-1)

        # choose sparse points randomly (equal number for each species)
        sparse_inds = np.array([], dtype=int)
        sparse_inds = \
            np.append(sparse_inds,
                      np.random.choice(np.arange(0, 16, 1),
                                       size=4, replace=False))
        sparse_inds = \
            np.append(sparse_inds,
                      np.random.choice(np.arange(16, 52, 1),
                                       size=4, replace=False))
        sparse_inds = \
            np.append(sparse_inds,
                      np.random.choice(np.arange(52, 216, 1),
                                       size=4, replace=False))
        sparse_inds = \
            np.append(sparse_inds,
                      np.random.choice(np.arange(216, 312, 1),
                                       size=4, replace=False))
        sparse_inds = \
            np.append(sparse_inds,
                      np.random.choice(np.arange(312, 316, 1),
                                       size=4, replace=False))

        # add training structure and sparse environments
        print('adding training structure...')
        sparse_gp.add_training_structure(train_struc)
        print('adding sparse environments...')
        for ind in sparse_inds:
            print(ind)
            sparse_gp.add_sparse_environment(train_struc.local_environments[ind])

        # compute test error
        print('updating alpha...')
        sparse_gp.update_alpha()

        print('predicting on test structure...')
        force_pred = sparse_gp.predict(test_struc)
        mae = np.mean(np.abs(test_forces.reshape(-1) - force_pred[1:-6]))
        print(mae)

# # save maes and stds
# maes = np.array(maes)
# stds = np.array(stds)
# np.save('maes', maes)
# np.save('stds', stds)

# # pickle the gp model
# gp_name = 'Sulfonate_3p5.gp'
# gp_file = open(gp_name, 'wb')
# pickle.dump(gp_model, gp_file)
