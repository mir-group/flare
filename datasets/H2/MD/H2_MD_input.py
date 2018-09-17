import numpy as np
import sys
import os
sys.path.append('../../../modules')
import qe_input

input_file_name = './H2_md.in'
output_file_name = './H2_md.out'
pw_loc = '/Users/jonpvandermause/AP275/qe-6.0/bin/pw.x'
calculation = 'md'

scf_inputs = dict(pseudo_dir='/Users/jonpvandermause/AP275/qe-6.0/pseudo',
                  outdir='./output',
                  nat=2,
                  ntyp=1,
                  ecutwfc=18.0,
                  cell=10 * np.eye(3),
                  species=['H', 'H'],
                  positions=[np.array([0, 0, 0]),
                             np.array([0, 0, 0.7])],
                  kvec=np.array([1, 1, 1]),
                  ion_names=['H'],
                  ion_masses=[1.008],
                  ion_pseudo=['H.pbe-kjpaw.UPF'])

md_inputs = dict(dt=5,
                 nstep=10,
                 ion_temperature='not_controlled',
                 tempw=300)

H2_md = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                         calculation, scf_inputs, md_inputs)

H2_md.run_espresso()

# remove output directory
if os.path.isdir('output'):
    os.system('rm -r output')

