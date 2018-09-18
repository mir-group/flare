import numpy as np
import sys
import os
sys.path.append('../../../modules')
import qe_input

input_file_name = './H2_relax.in'
output_file_name = './H2_relax.out'
pw_loc = os.environ.get('PWSCF_COMMAND')
calculation = 'relax'

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

H2_relax = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                            calculation, scf_inputs)

H2_relax.run_espresso()

# remove output directory
if os.path.isdir('output'):
    os.system('rm -r output')
