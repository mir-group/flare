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

# Make test structure.
test_frame = 3000
test_cell = cells[test_frame]
test_positions = positions[test_frame]


# Time descriptor calculation.
iterations = 100
store_times = np.zeros(iterations)

for n in range(iterations):
    time1 = time.time()
    test_structure = StructureDescriptor(
        test_cell, species, test_positions, cutoff,
        many_body_cutoffs, [descriptor_calculator]
    )
    time2 = time.time()
    store_times[n] = time2 - time1

print('mean: %.4f ms' % (np.mean(store_times) * 1e3))
print('std: %.4f ms' % (np.std(store_times) * 1e3))

# 8/29/20 (c741e3e3464b6f0dc3a4337e4e2c7b25574ed822)
# Laptop: 18.9(6) ms
# Tempo: 14.4(9) ms
