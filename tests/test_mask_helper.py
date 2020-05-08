import pytest
import numpy as np
from flare.struc import Structure
from flare.mask_helper import HyperParameterMasking
from .test_gp import dumpcompare


def test_generate_by_line():

    pm = HyperParameterMasking()
    pm.define_group('specie', 'O', ['O'])
    pm.define_group('specie', 'C', ['C'])
    pm.define_group('specie', 'H', ['H'])
    pm.define_group('bond', '**', ['C', 'H'])
    pm.define_group('bond', 'OO', ['O', 'O'])
    pm.define_group('triplet', '***', ['O', 'O', 'C'])
    pm.define_group('triplet', 'OOO', ['O', 'O', 'O'])
    pm.define_group('mb', '1.5', ['C', 'H'])
    pm.define_group('mb', '1.5', ['C', 'O'])
    pm.define_group('mb', '1.5', ['O', 'H'])
    pm.define_group('mb', '2', ['O', 'O'])
    pm.define_group('mb', '2', ['H', 'O'])
    pm.define_group('mb', '2.8', ['O', 'O'])
    pm.set_parameters('**', [1, 0.5])
    pm.set_parameters('OO', [1, 0.5])
    pm.set_parameters('***', [1, 0.5])
    pm.set_parameters('OOO', [1, 0.5])
    pm.set_parameters('1.5', [1, 0.5, 1.5])
    pm.set_parameters('2', [1, 0.5, 2])
    pm.set_parameters('2.8', [1, 0.5, 2.8])
    pm.set_parameters('cutoff2b', 5)
    pm.set_parameters('cutoff3b', 4)
    pm.set_parameters('cutoffmb', 3)
    hm = pm.generate_dict()
    print(hm)
    HyperParameterMasking.check_instantiation(hm)
    HyperParameterMasking.check_matching(hm, hm['hyps'], hm['cutoffs'])

def test_generate_by_line2():

    pm = HyperParameterMasking()
    pm.define_group('specie', 'O', ['O'])
    pm.define_group('specie', 'rest', ['C', 'H'])
    pm.define_group('bond', '**', ['*', '*'])
    pm.define_group('bond', 'OO', ['O', 'O'])
    pm.define_group('triplet', '***', ['*', '*', '*'])
    pm.define_group('triplet', 'Oall', ['O', 'O', 'O'])
    pm.set_parameters('**', [1, 0.5])
    pm.set_parameters('OO', [1, 0.5])
    pm.set_parameters('Oall', [1, 0.5])
    pm.set_parameters('***', [1, 0.5])
    pm.set_parameters('cutoff2b', 5)
    pm.set_parameters('cutoff3b', 4)
    hm = pm.generate_dict()
    print(hm)
    HyperParameterMasking.check_instantiation(hm)
    HyperParameterMasking.check_matching(hm, hm['hyps'], hm['cutoffs'])

def test_generate_by_list():

    pm = HyperParameterMasking()
    pm.list_groups('specie', ['O', 'C', 'H'])
    pm.list_groups('bond', [['*', '*'], ['O','O']])
    pm.list_groups('triplet', [['*', '*', '*'], ['O','O', 'O']])
    pm.list_parameters({'bond0':[1, 0.5], 'bond1':[2, 0.2],
                        'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                        'cutoff2b':2, 'cutoff3b':1})
    hm = pm.generate_dict()
    print(hm)
    HyperParameterMasking.check_instantiation(hm)
    HyperParameterMasking.check_matching(hm, hm['hyps'], hm['cutoffs'])

def test_initialization():
    pm = HyperParameterMasking(species=['O', 'C', 'H'],
                          bonds=[['*', '*'], ['O','O']],
                          triplets=[['*', '*', '*'], ['O','O', 'O']],
                          parameters={'bond0':[1, 0.5], 'bond1':[2, 0.2],
                                'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                                'cutoff2b':2, 'cutoff3b':1})
    hm = pm.hyps_mask
    print(hm)
    HyperParameterMasking.check_instantiation(hm)
    HyperParameterMasking.check_matching(hm, hm['hyps'], hm['cutoffs'])

def test_opt():
    pm = HyperParameterMasking(species=['O', 'C', 'H'],
                          bonds=[['*', '*'], ['O','O']],
                          triplets=[['*', '*', '*'], ['O','O', 'O']],
                          parameters={'bond0':[1, 0.5, 1], 'bond1':[2, 0.2, 2],
                                'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                                'cutoff2b':2, 'cutoff3b':1},
                          constraints={'bond0':[False, True]})
    hm = pm.hyps_mask
    print(hm)
    HyperParameterMasking.check_instantiation(hm)
    HyperParameterMasking.check_matching(hm, hm['hyps'], hm['cutoffs'])

def test_randomization():
    pm = HyperParameterMasking(species=['O', 'C', 'H'],
                          bonds=True, triplets=True,
                          mb=False, random=True,
                          parameters={'cutoff2b': 7,
                              'cutoff3b': 4.5,
                              'cutoffmb': 3},
                          verbose=True)
    hm = pm.hyps_mask
    print(hm)
    HyperParameterMasking.check_instantiation(hm)
    HyperParameterMasking.check_matching(hm, hm['hyps'], hm['cutoffs'])
    name = pm.find_group('specie', 'O')
    print("find group name for O", name)
    name = pm.find_group('bond', ['O', 'C'])
    print("find group name for O-C", name)

def test_from_dict():
    pm = HyperParameterMasking(species=['O', 'C', 'H'],
                          bonds=True, triplets=True,
                          mb=False, random=True,
                          parameters={'cutoff2b': 7,
                              'cutoff3b': 4.5,
                              'cutoffmb': 3},
                          verbose=True)
    hm = pm.hyps_mask
    HyperParameterMasking.check_instantiation(hm)
    HyperParameterMasking.check_matching(hm, hm['hyps'], hm['cutoffs'])
    print(hm['hyps'])
    print("obtain test hm", hm)

    pm1 = HyperParameterMasking.from_dict(hm, verbose=True)
    print("from_dict")
    hm1 = pm1.generate_dict()
    print(hm['hyps'])
    print(hm1['hyps'][:33], hm1['hyps'][33:])

    dumpcompare(hm, hm1)
