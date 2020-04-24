import pytest
import numpy as np
from flare.struc import Structure
from flare.mask_helper import ParameterMasking


def test_generate():

    pm = ParameterMasking()
    pm.define_group('spec', 'Cu', ['Cu'])
    pm.define_group('spec', 'O', ['O'])
    pm.define_group('spec', 'C', ['C'])
    pm.define_group('spec', 'H', ['H'])
    pm.define_group('bond', 'CuCu', ['Cu', 'Cu'])
    pm.define_group('bond', '**', ['C', 'H'])
    pm.define_group('triplet', 'Cu', ['Cu', 'Cu', 'Cu'])
    pm.define_group('triplet', '*', ['Cu', 'Cu', 'C'])
    pm.define_group('mb', '1.5', ['C', 'H'])
    pm.define_group('mb', '1.5', ['C', 'O'])
    pm.define_group('mb', '1.5', ['O', 'H'])
    pm.define_group('mb', '2', ['O', 'Cu'])
    pm.define_group('mb', '2', ['H', 'Cu'])
    pm.define_group('mb', '2.8', ['Cu', 'Cu'])
    pm.define_parameters('bond', 'CuCu', 1, 0.5)
    pm.define_parameters('bond', '**', 1, 0.5)
    pm.define_parameters('triplet', 'Cu', 1, 0.5)
    pm.define_parameters('triplet', '*', 1, 0.5)
    pm.define_parameters('triplet', '*', 1, 0.5)
    pm.define_parameters('mb', '1.5', 1, 0.5, 1.5)
    pm.define_parameters('mb', '2', 1, 0.5, 2)
    pm.define_parameters('mb', '2.8', 1, 0.5, 2.8)
    hm = pm.generate_dict()
    print(hm)

def test_generate():

    pm = ParameterMasking()
    pm.define_group('spec', 'Cu', ['Cu'])
    pm.define_group('spec', '*', ['*'])
    pm.define_group('bond', 'CuCu', ['Cu', 'Cu'])
    pm.define_group('bond', '**', ['*', '*'])
    pm.define_group('triplet', 'Cu', ['Cu', 'Cu', 'Cu'])
    pm.define_group('triplet', '*', ['*', '*', '*'])
    pm.define_parameters('bond', 'CuCu', 1, 0.5)
    pm.define_parameters('bond', '**', 1, 0.5)
    pm.define_parameters('triplet', 'Cu', 1, 0.5)
    pm.define_parameters('triplet', '*', 1, 0.5)
    hm = pm.generate_dict()
    print(hm)
