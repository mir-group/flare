import numpy as np

a = 1.0
b = 3.5
order = 81
elem1 = 'C'
elem2 = 'C'

label_2 = '8C-g81-t24-s0'
label_3 = '8C-g81-t24-s72'
coefs_2 = np.load(label_2 + '.npy')
coefs_3 = np.load(label_3 + '.npy')

f = open('C8.txt', 'w')
header_comment = '''# #2bodyarray #3bodyarray
# elem1 elem2 a b order
'''
f.write(header_comment)
header = '''
{twobodyarray} {threebodyarray}
{elem1} {elem2} {a} {b} {order}
'''.format(twobodyarray=1, threebodyarray=1, elem1=elem1, elem2=elem2, a=a, b=b, order=order)
f.write(header)

for c, coef in enumerate(coefs_2):
    f.write('{:.10e} '.format(coef))
    if c % 5 == 4:
        f.write('\n')

f.write('\n')

a = [1.0, 1.0, 0.0]
b = [3.5, 3.5, np.pi]
order = [81, 81, 81]
elem1 = 'C'
elem2 = 'C'
elem3 = 'C'

header_3 = '''
{elem1} {elem2} {elem3} {a1} {a2} {a3} {b1} {b2} {b3:.10e} {order1} {order2} {order3}
'''.format(elem1=elem1, elem2=elem2, elem3=elem3,
        a1=a[0], a2=a[1], a3=a[2], 
        b1=b[0], b2=b[1], b3=b[2], 
        order1=order[0], order2=order[1], order3=order[2])
f.write(header_3)

n = 0
for i in range(coefs_3.shape[0]):
    for j in range(coefs_3.shape[1]):
        for k in range(coefs_3.shape[2]):
            coef = coefs_3[i, j, k]
            f.write('{:.10e} '.format(coef))
            if n % 5 == 4:
                f.write('\n')
            n += 1

f.close()



