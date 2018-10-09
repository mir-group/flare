import numpy as np
import sys
import os
import qe_input


bash_name = 'C_VC.sh'

bash_inputs = dict(n=32, N=1, t=1, e='test.err', p='kozinsky', o='test.out',
                   mem_per_cpu=1000,
                   mail_user='jonathan_vandermause@g.harvard.edu',
                   command='python C_VC_Input.py')

conv_bash = qe_input.BashInput(bash_name, bash_inputs)
conv_bash.write_bash_text()
