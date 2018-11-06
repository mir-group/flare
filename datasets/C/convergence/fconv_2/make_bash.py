import numpy as np
import sys
import os
sys.path.append('../../../../modules')
import qe_input


bash_name = 'convergence.sh'

bash_inputs = dict(n=32, N=1, t=7, e='test.err', p='kozinsky', o='test.out',
                   mem_per_cpu=1000,
                   mail_user='jonathan_vandermause@g.harvard.edu',
                   command='python force_convergence.py')

conv_bash = qe_input.BashInput(bash_name, bash_inputs)
conv_bash.write_bash_text()

if os.path.isdir('__pycache__'):
    os.system('rm -r __pycache__')