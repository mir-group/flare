import pytest
import numpy as np
from flare.struc import Structure
from flare.utils.parameter_helper import ParameterHelper
from flare.parameters import Parameters
from .test_gp import dumpcompare

def test_initialization():
    '''
    simplest senario
    '''
    pm = ParameterHelper(kernels=['bond', 'triplet'],
                         parameters={'bond':[1, 0.5],
                                     'triplet':[1, 0.5],
                                     'cutoff_bond':2,
                                     'cutoff_triplet':1,
                                     'noise':0.05},
                         verbose="DEBUG")
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)

@pytest.mark.parametrize('ones', [True, False])
def test_initialization(ones):
    '''
    simplest senario
    '''
    pm = ParameterHelper(kernels=['bond', 'triplet'],
                         parameters={'cutoff_bond':2,
                                     'cutoff_triplet':1,
                                     'noise':0.05},
                         ones=ones,
                         random=not ones,
                         verbose="DEBUG")
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)

def test_initialization2():
    pm = ParameterHelper(species=['O', 'C', 'H'],
                         kernels={'bond':[['*', '*'], ['O','O']],
                                  'triplet':[['*', '*', '*'], ['O','O', 'O']]},
                          parameters={'bond0':[1, 0.5], 'bond1':[2, 0.2],
                                'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                                'cutoff_bond':2, 'cutoff_triplet':1},
                          verbose="DEBUG")
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)

def test_generate_by_line():

    pm = ParameterHelper(verbose="DEBUG")
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
    pm.set_parameters('cutoff_bond', 5)
    pm.set_parameters('cutoff_triplet', 4)
    pm.set_parameters('cutoff_mb', 3)
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)

def test_generate_by_line2():

    pm = ParameterHelper(verbose="DEBUG")
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
    pm.set_parameters('cutoff_bond', 5)
    pm.set_parameters('cutoff_triplet', 4)
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)

def test_generate_by_list():

    pm = ParameterHelper(verbose="DEBUG")
    pm.list_groups('specie', ['O', 'C', 'H'])
    pm.list_groups('bond', [['*', '*'], ['O','O']])
    pm.list_groups('triplet', [['*', '*', '*'], ['O','O', 'O']])
    pm.list_parameters({'bond0':[1, 0.5], 'bond1':[2, 0.2],
                        'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                        'cutoff_bond':2, 'cutoff_triplet':1})
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)


def test_opt():
    pm = ParameterHelper(species=['O', 'C', 'H'],
                          kernels={'bond':[['*', '*'], ['O','O']],
                                   'triplet':[['*', '*', '*'], ['O','O', 'O']]},
                          parameters={'bond0':[1, 0.5, 1], 'bond1':[2, 0.2, 2],
                                'triplet0':[1, 0.5], 'triplet1':[2, 0.2],
                                'cutoff_bond':2, 'cutoff_triplet':1},
                          constraints={'bond0':[False, True]},
                          verbose="DEBUG")
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)

def test_randomization():
    pm = ParameterHelper(species=['O', 'C', 'H'],
                          kernels=['bond', 'triplet'],
                          allseparate=True,
                          random=True,
                          parameters={'cutoff_bond': 7,
                              'cutoff_triplet': 4.5,
                              'cutoff_mb': 3},
                          verbose="debug")
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)
    name = pm.find_group('specie', 'O')
    name = pm.find_group('bond', ['O', 'C'])

def test_from_dict():
    pm = ParameterHelper(species=['O', 'C', 'H'],
                         kernels=['bond', 'triplet'],
                         allseparate=True,
                         random=True,
                         parameters={'cutoff_bond': 7,
                             'cutoff_triplet': 4.5,
                             'cutoff_mb': 3},
                         verbose="debug")
    hm = pm.as_dict()
    Parameters.check_instantiation(hm)

    pm1 = ParameterHelper.from_dict(hm, verbose="debug", init_spec=['O', 'C', 'H'])
    print("from_dict")
    hm1 = pm1.as_dict()
    print(hm['hyps'])
    print(hm1['hyps'][:33], hm1['hyps'][33:])

    Parameters.compare_dict(hm, hm1)
