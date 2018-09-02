#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""" Compute Two-body kernel between chemical environments
    (for test purposes only)

Jon V
"""

import numpy as np

def two_body(x1, x2, d1, d2, sig, ls):
    d = sig*sig / (ls*ls*ls*ls)
    e = ls*ls
    f = 1/(2*ls*ls)
    kern = 0
    
    # record central atom types
    c1 = x1['central_atom']
    c2 = x2['central_atom']
    
    x1_len = len(x1['dists'])
    x2_len = len(x2['dists'])
    
    for m in range(x1_len):
        e1 = x1['types'][m]
        r1 = x1['dists'][m]
        coord1 = x1[d1][m]
        for n in range(x2_len):
            e2 = x2['types'][n]
            r2 = x2['dists'][n]
            coord2 = x2[d2][n]
            
            # check that atom types match
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rr = (r1-r2)*(r1-r2)
                kern += d*np.exp(-f*rr)*coord1*coord2*(e-rr)
                
    return kern

