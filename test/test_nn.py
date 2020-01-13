import ace
import nnp
import numpy as np
import torch
import copy

species = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]

coded_species = []
for spec in species:
    if spec == 6:
        coded_species.append(0)
    if spec == 8:
        coded_species.append(1)
    if spec == 1:
        coded_species.append(2)

cell = np.eye(3) * 100
positions = np.array([[ 2.12837244, -1.02214338, -0.28104315],
                      [ 0.50960419,  0.68424805, -1.75534119],
                      [ 2.56394908, -0.45691439, -1.40895434],
                      [ 1.83916061,  0.48815384, -2.13254568],
                      [-3.48298414,  1.13759712, -0.36659624],
                      [ 0.90236653, -0.55489857,  0.32401515],
                      [ 0.10444023,  0.11761382, -0.57907443],
                      [ 1.20998356, -2.11006509,  2.16021163],
                      [-1.49580309,  2.31093071, -0.1033624 ],
                      [-0.55755898, -0.76447274,  2.31374533],
                      [ 0.53088979, -1.19079794,  1.57532265],
                      [-2.01399916,  1.26567364, -0.34566542],
                      [-1.24917847,  0.11058142, -0.35152255],
                      [-0.26895874, -1.24669592,  3.20771389],
                      [ 2.65635959, -1.73157442,  0.26207606],
                      [-0.27368027,  1.27614842, -2.37945504],
                      [ 3.57300086, -0.82058207, -1.6675233 ],
                      [ 2.19445749,  0.74663571, -3.1649488 ],
                      [-3.6649996 ,  1.36982664,  0.68209262],
                      [-3.84709602,  1.91009051, -0.95156392],
                      [-3.87331649,  0.10910619, -0.56318171]])

noa = len(positions)
test_struc = ace.Structure(cell, coded_species, positions)

cutoff = 7
test_env = ace.LocalEnvironment(test_struc, 0, cutoff)

radial_basis = "chebyshev"
cutoff_function = "cosine"
radial_hyps = [0, cutoff]
cutoff_hyps = []
descriptor_settings = [3, 1, 0]
descriptor = \
    ace.DescriptorCalculator(radial_basis, cutoff_function, radial_hyps,
                             cutoff_hyps, descriptor_settings)

descriptor.compute_B2(test_env)

# Construct species NNP.
input_size = 6
layers = [10]
activation = torch.tanh
descriptor_method = "compute_B2"
spec_test = nnp.SpeciesNet(layers, input_size, activation)

# Test forward pass.
input_tensor = torch.tensor(descriptor.descriptor_vals, requires_grad=True)
out = spec_test.forward(input_tensor.double())
out.backward()

# Construct NNP object.
nnp_test = nnp.NNP(3, layers, input_size, activation, descriptor,
                   descriptor_method, cutoff)
test_E = nnp_test.predict_E(test_struc)
test_EF = nnp_test.predict_EF(test_struc)
test_EFS = nnp_test.predict_EFS(test_struc)
test_E.backward()
test_EF[1].backward()

# Try local EF function.
test1 = nnp_test.predict_local_EF(test_env)
test2 = nnp_test.predict_local_EFS(test_env)
test1[1].backward()
test2[1].backward()

# ------------------------
#  Test NN force
# ------------------------

cell = np.eye(3) * 10
species = [0, 0, 1, 2]
positions = np.array([[0, 0, 0],
                      [1, 0.2, -0.1],
                      [0.4, -0.3, 0.2],
                      [0.1, 0.1, -0.3]])
delt = 1e-2
atom = 1
comp = 1
pos_delt = copy.copy(positions)
pos_delt[atom, comp] += delt

pos_delt_2 = copy.copy(positions)
pos_delt_2[atom, comp] -= delt

test_struc = ace.Structure(cell, species, positions)
struc_delt = ace.Structure(cell, species, pos_delt)
struc_delt_2 = ace.Structure(cell, species, pos_delt_2)

test_F = nnp_test.predict_F(test_struc)
test_E = nnp_test.predict_E(test_struc)
E_delt = nnp_test.predict_E(struc_delt)
E_delt_2 = nnp_test.predict_E(struc_delt_2)

force_an = test_F[atom * 3 + comp].item()
force_delt = -((E_delt - E_delt_2) / (2 * delt))[0].item()

print(force_an)
print(force_delt)
print(force_delt - force_an)

# Check parameter gradients.
param_list = list(nnp_test.parameters())
nnp_test.zero_grad()
# print(param_list[0].grad)
# test_F = nnp_test.predict_F(test_struc)
random_target = torch.randn(len(positions) * 3).double()
nnp_test.update_weights(test_struc, random_target)
# print(param_list)

# print('\nspec0 parameters\n')
# print(list(nnp_test.spec0.parameters())[0])

# # Test autograd function.
# x = torch.tensor([1., 2., 3.], requires_grad=True)
# y = (x * x * x).sum()
# z = torch.autograd.grad(y, x, create_graph=True)[0]
# z.sum().backward()
# print(x.grad)

x = torch.zeros(2)
y = torch.tensor([1., 2.], requires_grad=True)
z = torch.tensor([4., 5.], requires_grad=True)
x += y * z
x += y * z * y * z
s = x.sum()
s.backward()
print(z.grad)

# print(nnp_test.spec0.forward(torch.randn(10, 6).double()))

# # Test NN gradients
# print(list(nnp_test.parameters())[0])
# print(list(nnp_test.parameters())[0].grad)

# x = torch.tensor([5.], requires_grad=True)
# y = x * x * x
# y2 = x * x
# z = torch.autograd.grad(y, x, create_graph=True)[0]
# z2 = torch.autograd.grad(y2, x, create_graph=True)[0]
# (z + z2).backward()
# print(x.grad)
