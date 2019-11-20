import numpy as np
from spherical_harmonics import get_Y_grad, get_Y_np
import time

l = 4
vec = np.array([1, 2, 3])
r = np.linalg.norm(vec)
vec_x = np.array([1 + 1e-6, 2, 3])

# time y grad
time0 = time.time()
harm1, grad1 = get_Y_grad(vec, l)
time1 = time.time()
print(time1 - time0)

rep = 1000
time0 = time.time()
for n in range(rep):
    harm1, grad1 = get_Y_grad(vec, l)
time1 = time.time()
print(time1 - time0)

# time y
time0 = time.time()
for n in range(rep):
    harm2 = get_Y_np(vec, r, l)
time1 = time.time()
print(time1 - time0)

# check gradient

print(harm1)
print(harm2)
