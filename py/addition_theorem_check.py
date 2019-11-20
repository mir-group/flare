import numpy as np
import torch
import spherical_harmonics
import time

torch.set_printoptions(precision=12)

# define test vectors
vec1_np = np.array([0.2, 0.5, 0.3])
r1_np = np.linalg.norm(vec1_np)

vec1 = torch.tensor([0.2, 0.5, 0.3])
r1 = spherical_harmonics.compute_distance(vec1)

vec2_np = np.array([-0.412, 0.719, 1.12])
r2_np = np.linalg.norm(vec2_np)

vec2 = torch.tensor([-0.412, 0.719, 1.12])
r2 = spherical_harmonics.compute_distance(vec2)

# define rotation operator
theta = 47
r_x = torch.tensor([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

# create rotated vectors
rot1 = torch.matmul(r_x, vec1)
rot2 = torch.matmul(r_x, vec2)

# compute spherical harmonics
l = 6

time0 = time.time()
for n in range(1000):
    harm1 = spherical_harmonics.get_Y_np(vec1_np, r1_np, l)
time1 = time.time()
print((time1-time0) / 1000)

time0 = time.time()
for n in range(1000):
    harm1 = spherical_harmonics.get_Y(vec1, r1, l)
time1 = time.time()
print((time1-time0) / 1000)

harm2 = spherical_harmonics.get_Y(vec2, r2, l)
harm3 = spherical_harmonics.get_Y(rot1, r1, l)
harm4 = spherical_harmonics.get_Y(rot2, r2, l)

# check that the addition theorem is satisfied
print(torch.sum(harm1 * harm2))
print(torch.sum(harm3 * harm4))
