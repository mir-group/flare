import numpy as np
import sys
import time
sys.path.append('../build')
from _C_flare import SparseGP_DTC, DotProductKernel, B2_Calculator, \
    StructureDescriptor


# Load training data.
NiTi_data = np.load('NiTi_AIMD.npz')
positions = NiTi_data['positions']
forces = NiTi_data['forces']
cells = NiTi_data['cells']
stresses = NiTi_data['stresses']
energies = NiTi_data['energies']
species = [0, 1, 1, 0] * 8  # Ti, Ni, Ni, Ti
single_atom_energy = -6281.24833043 / 2  # from EV_Energies directory
n_atoms = 32

# Create many-body kernel.
sigma = 2.0
power = 2
kernel = DotProductKernel(sigma, power, 0)

# Set up descriptor.
cutoff = 5.0
sigma = 2.0
power = 2
kernel = DotProductKernel(sigma, power, 0)
cutoff_function = "quadratic"
many_body_cutoffs = [cutoff]
radial_basis = "chebyshev"
radial_hyps = [0., cutoff]
cutoff_hyps = []
descriptor_settings = [2, 8, 3]
descriptor_calculator = \
    B2_Calculator(radial_basis, cutoff_function, radial_hyps, cutoff_hyps,
                  descriptor_settings, 0)

# Create sparse GP.
sigma_e = 0.02
sigma_f = 0.07
sigma_s = 0.001
sparse_gp = SparseGP_DTC([kernel], sigma_e, sigma_f, sigma_s)

# Create test structure.
test_frame = 3000
test_cell = cells[test_frame]
test_positions = positions[test_frame]
test_energy = energies[test_frame]
test_forces = forces[test_frame]
test_stresses = stresses[test_frame]
test_structure = StructureDescriptor(
    test_cell, species, test_positions, cutoff,
    many_body_cutoffs, [descriptor_calculator]
)

# Add training structures.
training_frames = [1500, 1750, 2000]
for train_count, frame_no in enumerate(training_frames):
    cell = cells[frame_no]
    pos = positions[frame_no]
    frcs = forces[frame_no]

    en = energies[frame_no]
    stress = stresses[frame_no]
    stress_tens = np.array([stress[0, 0], stress[0, 1], stress[0, 2],
                            stress[1, 1], stress[1, 2], stress[2, 2]])

    training_structure = StructureDescriptor(
        cell, species, pos, cutoff, many_body_cutoffs,
        [descriptor_calculator])

    training_structure.forces = frcs.reshape(-1)
    training_structure.stresses = stress_tens
    training_structure.energy = np.array([en]) - (n_atoms * single_atom_energy)

    sparse_gp.add_sparse_environments(training_structure.local_environments)
    sparse_gp.add_training_structure(training_structure)

sparse_gp.update_matrices_QR()

# Time mean prediction.
reps = 5
store_times = np.zeros(reps)
for n in range(reps):
    time1 = time.time()
    efs = sparse_gp.predict(test_structure)
    time2 = time.time()
    store_times[n] = time2 - time1

print('Mean prediction time: %.3f ms' % (np.mean(store_times) * 1e3))
print('Std: %.3f ms' % (np.std(store_times) * 1e3))

force_mae = \
    np.mean(np.abs(efs[1:-6].reshape(-1, 3) - test_forces))
print('Force MAE: %.3f eV/A' % force_mae)

# Time mean and variance prediction.
reps = 5
store_times = np.zeros(reps)
for n in range(reps):
    time1 = time.time()
    sparse_gp.predict_on_structure(test_structure)
    time2 = time.time()
    store_times[n] = time2 - time1

print('Number of sparse environments: %i' % sparse_gp.Kuu.shape[0])
print('Mean and variance prediction time: %.3f ms' % (np.mean(store_times) * 1e3))
print('Std: %.3f ms' % (np.std(store_times) * 1e3))

force_mae = \
    np.mean(np.abs(test_structure.mean_efs[1:-6].reshape(-1, 3) - test_forces))
print('Force MAE: %.3f eV/A' % force_mae)

# 8/29/20 (c741e3e3464b6f0dc3a4337e4e2c7b25574ed822)
# Laptop: 178(8) ms (M+V)
# Tempo: 269(2) ms (M+V)

# 8/29/20 (52e9ff7bad8ed5b19468407eb71a748ba74469e9)
# Laptop: 56(2) ms (M), 126(5) ms (M+V)
# Tempo: 11(6) ms (M), 52(3) ms (M+V)
