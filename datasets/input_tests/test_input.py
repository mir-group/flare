import numpy as np
import sys
sys.path.append('../../modules')
import qe_input

# make test input
input_file_name = './test.in'
output_file_name = 'test.out'
calculation = 'md'
pw_loc = '/Users/jonpvandermause/AP275/qe-6.0/bin/pw.x'
scf_inputs = dict(pseudo_dir='test/pseudo',
                  outdir='.',
                  nat=2,
                  ntyp=2,
                  ecutwfc=18.0,
                  cell=np.eye(3),
                  species=['C', 'Si'],
                  positions=[np.array([0, 0, 0]),
                             np.array([0.5, 0.5, 0.5])],
                  kvec=np.array([4, 4, 4]),
                  ion_names=['C', 'Si'],
                  ion_masses=[2.0, 3.0],
                  ion_pseudo=['C.pz-rrkjus.UPF', 'Si.pz-rrkjus.UPF'])

md_inputs = dict(dt=20,
                 nstep=1000,
                 ion_temperature='rescaling',
                 tempw=1000)

test_md = qe_input.QEInput(input_file_name, output_file_name, pw_loc, 
                           calculation, scf_inputs, md_inputs)
