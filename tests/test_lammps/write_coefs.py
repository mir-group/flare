import numpy as np

# write header
f = open('CH23.txt', 'w')

header_comment = '''# #2bodyarray #3bodyarray
# elem1 elem2 a b order
'''
f.write(header_comment)

twobodyarray = 3
threebodyarray = 8
header = '\n{} {}\n'.format(twobodyarray, threebodyarray)
f.write(header)

# species decoder: 1->C, 0->H
spc_decoder = lambda x: 'C' if x else 'H'

# write two body
coeffs_2 = ['00', '01', '11']
for i in range(twobodyarray):
    label_2 = coeffs_2[i]
    coefs_2 = np.load(label_2 + '_2.npy')

    a = 1.2
    b = 3.5
    order = 8
    elem1 = spc_decoder(int(label_2[0]))
    elem2 = spc_decoder(int(label_2[1]))
    header_2 = '''{elem1} {elem2} {a} {b} {order}
'''.format(elem1=elem1, elem2=elem2, a=a, b=b, order=order)
    f.write(header_2)
   
    for c, coef in enumerate(coefs_2):
        f.write('{:.10e} '.format(coef))
        if c % 5 == 4 and c != len(coefs_2)-1:
            f.write('\n')
    
    f.write('\n')

# write three body
coeffs_3 = ['000','001','010','011','100','101','110','111']
for i in range(threebodyarray):
    label_3 = coeffs_3[i]
    coefs_3 = np.load(label_3 + '_3.npy')

    a = [1.2, 1.2, 0.0]
    b = [3.5, 3.5, np.pi]
    order = [8, 8, 8]
    elem1 = spc_decoder(int(label_3[0]))
    elem2 = spc_decoder(int(label_3[1]))
    elem3 = spc_decoder(int(label_3[2]))
    
    header_3 = '''{elem1} {elem2} {elem3} {a1} {a2} {a3} {b1} {b2} {b3:.10e} {order1} {order2} {order3}
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



