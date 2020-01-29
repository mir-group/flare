import numpy as np
import ace
import torch
import nnp

# -----------------------------------------------------------------------------
#                            load training data
# -----------------------------------------------------------------------------

data_file = '/n/home03/jonpvandermause/Schnet_Data/aspirin_dft.npz'
data = np.load(data_file)
no_strucs = len(data['E'])
training_size = 1000
training_pts = np.random.choice(no_strucs, training_size, replace=False)

species = data['z']
species_code = {'6': 0, '8': 1, '1' : 2}

coded_species = []
for spec in species:
    coded_species.append(species_code[str(spec)])

forces = data['F']
positions = data['R']
energies = data['E'] - np.mean(data['E'])
cell = np.eye(3) * 100
noa = len(species)

# -----------------------------------------------------------------------------
#                           set descriptor settings
# -----------------------------------------------------------------------------

cutoff = 7.
radial_basis = "chebyshev"
cutoff_function = "cosine"
radial_hyps = [0, cutoff]
cutoff_hyps = []
nos = 3
N = 12
no_radial = nos * N
lmax = 10
descriptor_settings = [nos, N, lmax]
descriptor_calculator = \
    ace.DescriptorCalculator(radial_basis, cutoff_function, radial_hyps,
                             cutoff_hyps, descriptor_settings)
descriptor_method = "compute_B2"

# -----------------------------------------------------------------------------
#                             set up nn model
# -----------------------------------------------------------------------------

layers = [100, 100, 100]
input_size = int(no_radial * (no_radial + 1) * (lmax + 1) / 2)
activation = torch.tanh

optimizer=torch.optim.SGD
optimizer_kwargs={"lr" : 1e-6, "momentum" : 0.9}

nn_model = nnp.NNP(nos, layers, input_size, activation, descriptor_calculator,
                   descriptor_method, cutoff, optimizer=optimizer,
                   optimizer_kwargs=optimizer_kwargs)

# -----------------------------------------------------------------------------
#                           train the network
# -----------------------------------------------------------------------------

# open file for training loss
epochs = 10000
progress_file = 'progress.txt'
f = open(progress_file, 'w')

train_ens = []
for n in training_pts:
    train_ens.append(energies[n])
print(str(np.std(train_ens)) + '\n')

skip = 1000
running_loss = 0
counter = 0
for n in range(epochs):
    f.write('epoch %i:\n' % n)
    for training_pt in training_pts:
        # set training structure
        positions_curr = positions[training_pt]
        structure_curr = ace.Structure(cell, coded_species, positions_curr)

        # set target
        target = torch.from_numpy(forces[training_pt].reshape(-1)).double()

        # update the weights
        running_loss += nn_model.update_weights(structure_curr, target)
        counter += 1

        if counter % skip == 0:
            f.write('%.6f\n' % (np.sqrt(running_loss / skip)))
            running_loss = 0
            f.flush()

f.close()
