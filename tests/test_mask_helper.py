import pytest
import numpy as np
from flare.struc import Structure
from flare.mask_helper import HyperParameterMasking


def test_generate_by_line():

    pm = HyperParameterMasking()
    pm.define_group('specie', 'He', ['He'])
    pm.define_group('specie', 'O', ['O'])
    pm.define_group('specie', 'C', ['C'])
    pm.define_group('specie', 'H', ['H'])
    pm.define_group('bond', '**', ['C', 'H'])
    pm.define_group('bond', 'HeHe', ['He', 'He'])
    pm.define_group('triplet', '***', ['He', 'He', 'C'])
    pm.define_group('triplet', 'HeHeHe', ['He', 'He', 'He'])
    pm.define_group('mb', '1.5', ['C', 'H'])
    pm.define_group('mb', '1.5', ['C', 'O'])
    pm.define_group('mb', '1.5', ['O', 'H'])
    pm.define_group('mb', '2', ['O', 'He'])
    pm.define_group('mb', '2', ['H', 'He'])
    pm.define_group('mb', '2.8', ['He', 'He'])
    pm.set_parameters('**', [1, 0.5])
    pm.set_parameters('HeHe', [1, 0.5])
    pm.set_parameters('***', [1, 0.5])
    pm.set_parameters('HeHeHe', [1, 0.5])
    pm.set_parameters('1.5', [1, 0.5, 1.5])
    pm.set_parameters('2', [1, 0.5, 2])
    pm.set_parameters('2.8', [1, 0.5, 2.8])
    pm.set_parameters('cutoff2b', 5)
    pm.set_parameters('cutoff3b', 4)
    pm.set_parameters('cutoffmb', 3)
    hm = pm.generate_dict()
    print(hm)
    HyperParameterMasking.check_instantiation(hm)

def test_generate_by_line2():

    pm = HyperParameterMasking()
    pm.define_group('specie', 'He', ['He'])
    pm.define_group('specie', 'rest', ['C', 'H', 'O'])
    pm.define_group('bond', '**', ['*', '*'])
    pm.define_group('bond', 'HeHe', ['He', 'He'])
    pm.define_group('triplet', '***', ['*', '*', '*'])
    pm.define_group('triplet', 'Heall', ['He', 'He', 'He'])
    pm.set_parameters('**', [1, 0.5])
    pm.set_parameters('HeHe', [1, 0.5])
    pm.set_parameters('Heall', [1, 0.5])
    pm.set_parameters('***', [1, 0.5])
    pm.set_parameters('cutoff2b', 5)
    pm.set_parameters('cutoff3b', 4)
    hm = pm.generate_dict()
    print(hm)
    HyperParameterMasking.check_instantiation(hm)

def test_generate_by_list():

    pm = HyperParameterMasking()
    pm.list_groups('specie', ['He', 'C', 'H', 'O'])
    pm.list_groups('bond', [['*', '*'], ['He','He']])
    pm.list_groups('triplet', [['*', '*', '*'], ['He','He', 'He']])
    pm.list_parameters({'bond0':[1, 0.5], 'bond1':[2, 0.2],
                        'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                        'cutoff2b':2, 'cutoff3b':1})
    hm = pm.generate_dict()
    print(hm)
    HyperParameterMasking.check_instantiation(hm)

def test_initialization():
    pm = HyperParameterMasking(species=['He', 'C', 'H', 'O'],
                          bonds=[['*', '*'], ['He','He']],
                          triplets=[['*', '*', '*'], ['He','He', 'He']],
                          parameters={'bond0':[1, 0.5], 'bond1':[2, 0.2],
                                'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                                'cutoff2b':2, 'cutoff3b':1})
    hm = pm.hyps_mask
    print(hm)
    HyperParameterMasking.check_instantiation(hm)

def test_opt():
    pm = HyperParameterMasking(species=['He', 'C', 'H', 'O'],
                          bonds=[['*', '*'], ['He','He']],
                          triplets=[['*', '*', '*'], ['He','He', 'He']],
                          parameters={'bond0':[1, 0.5, 1], 'bond1':[2, 0.2, 2],
                                'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                                'cutoff2b':2, 'cutoff3b':1},
                          constraints={'bond0':[False, True]})
    hm = pm.hyps_mask
    print(hm)
    HyperParameterMasking.check_instantiation(hm)
