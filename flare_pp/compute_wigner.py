from sympy.physics.wigner import wigner_3j
import numpy as np

lmax = 1
count = 0
nonzero_count = 0

# Count nonzero L triples.
lcount = 0
for l1 in range(lmax + 1):
    for l2 in range(lmax + 1):
        for l3 in range(lmax + 1):
            if (np.abs(l1 - l2) <= l3) and (l3 <= l1 + l2):
                lcount += 1

                print("lvals:")
                print(l1)
                print(l2)
                print(l3)

print(lcount)

for l1 in range(lmax + 1):
    ind_1 = ((lmax + 1)**4) * (l1 * l1)
    for l2 in range(lmax + 1):
        ind_2 = ind_1 + ((lmax + 1)**2) * (l2 * l2) * (2 * l1 + 1)
        for l3 in range(lmax + 1):
            ind_3 = ind_2 + (l3 * l3) * (2 * l2 + 1) * (2 * l1 + 1)
            for m1 in range(2 * l1 + 1):
                m1_val = m1 - l1
                ind_4 = ind_3 + m1 * (2 * l3 + 1) * (2 * l2 + 1)
                for m2 in range(2 * l2 + 1):
                    m2_val = m2 - l2
                    ind_5 = ind_4 + m2 * (2 * l3 + 1)
                    for m3 in range(2 * l3 + 1):
                        m3_val = m3 - l3
                        wig_val = wigner_3j(l1, l2, l3, m1_val, m2_val, m3_val)
                        if (wig_val != 0):
                            print('wigner3j_coeffs(%i) = %s;' %
                                  (count, wig_val))
                            nonzero_count += 1

                        # print(ind_5 + m3)
                        # print(count)
                        count += 1

# print(nonzero_count)
# print(count)
