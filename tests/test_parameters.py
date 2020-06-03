import pytest
import numpy as np
from flare.struc import Structure
from flare.utils.parameter_helper import ParameterHelper
from flare.parameters import Parameters


def test_initialization():
    '''
    simplest senario
    '''
    pm = ParameterHelper(kernels=['twobody', 'threebody'],
                         parameters={'twobody': [1, 0.5],
                                     'threebody': [1, 0.5],
                                     'cutoff_twobody': 2,
                                     'cutoff_threebody': 1,
                                     'noise': 0.05},
                         verbose="DEBUG")
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)


@pytest.mark.parametrize('ones', [True, False])
def test_initialization2(ones):
    '''
    simplest senario
    '''
    pm = ParameterHelper(kernels=['twobody', 'threebody'],
                         parameters={'cutoff_twobody': 2,
                                     'cutoff_threebody': 1,
                                     'noise': 0.05},
                         ones=ones,
                         random=not ones,
                         verbose="DEBUG")
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)


def test_initialization3():
    pm = ParameterHelper(species=['O', 'C', 'H'],
                         kernels={'twobody': [['*', '*'], ['O', 'O']],
                                  'threebody': [['*', '*', '*'], ['O', 'O', 'O']]},
                         parameters={'twobody0': [1, 0.5], 'twobody1': [2, 0.2],
                                     'threebody0': [1, 0.5], 'threebody1': [2, 0.2],
                                     'cutoff_twobody': 2, 'cutoff_threebody': 1},
                         verbose="DEBUG")
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)


def test_generate_by_line():

    pm = ParameterHelper(verbose="DEBUG")
    pm.define_group('specie', 'O', ['O'])
    pm.define_group('specie', 'C', ['C'])
    pm.define_group('specie', 'H', ['H'])
    pm.define_group('twobody', '**', ['C', 'H'])
    pm.define_group('twobody', 'OO', ['O', 'O'])
    pm.define_group('threebody', '***', ['O', 'O', 'C'])
    pm.define_group('threebody', 'OOO', ['O', 'O', 'O'])
    pm.define_group('manybody', '1.5', ['C', 'H'])
    pm.define_group('manybody', '1.5', ['C', 'O'])
    pm.define_group('manybody', '1.5', ['O', 'H'])
    pm.define_group('manybody', '2', ['O', 'O'])
    pm.define_group('manybody', '2', ['H', 'O'])
    pm.define_group('manybody', '2.8', ['O', 'O'])
    pm.set_parameters('**', [1, 0.5])
    pm.set_parameters('OO', [1, 0.5])
    pm.set_parameters('***', [1, 0.5])
    pm.set_parameters('OOO', [1, 0.5])
    pm.set_parameters('1.5', [1, 0.5, 1.5])
    pm.set_parameters('2', [1, 0.5, 2])
    pm.set_parameters('2.8', [1, 0.5, 2.8])
    pm.set_parameters('cutoff_twobody', 5)
    pm.set_parameters('cutoff_threebody', 4)
    pm.set_parameters('cutoff_manybody', 3)
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)


def test_generate_by_line2():

    pm = ParameterHelper(verbose="DEBUG")
    pm.define_group('specie', 'O', ['O'])
    pm.define_group('specie', 'rest', ['C', 'H'])
    pm.define_group('twobody', '**', ['*', '*'])
    pm.define_group('twobody', 'OO', ['O', 'O'])
    pm.define_group('threebody', '***', ['*', '*', '*'])
    pm.define_group('threebody', 'Oall', ['O', 'O', 'O'])
    pm.set_parameters('**', [1, 0.5])
    pm.set_parameters('OO', [1, 0.5])
    pm.set_parameters('Oall', [1, 0.5])
    pm.set_parameters('***', [1, 0.5])
    pm.set_parameters('cutoff_twobody', 5)
    pm.set_parameters('cutoff_threebody', 4)
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)


def test_generate_by_list():

    pm = ParameterHelper(verbose="DEBUG")
    pm.list_groups('specie', ['O', 'C', 'H'])
    pm.list_groups('twobody', [['*', '*'], ['O', 'O']])
    pm.list_groups('threebody', [['*', '*', '*'], ['O', 'O', 'O']])
    pm.list_parameters({'twobody0': [1, 0.5], 'twobody1': [2, 0.2],
                        'threebody0': [1, 0.5], 'threebody1': [2, 0.2],
                        'cutoff_twobody': 2, 'cutoff_threebody': 1})
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)


def test_opt():
    pm = ParameterHelper(species=['O', 'C', 'H'],
                         kernels={'twobody': [['*', '*'], ['O', 'O']],
                                  'threebody': [['*', '*', '*'], ['O', 'O', 'O']]},
                         parameters={'twobody0': [1, 0.5, 1], 'twobody1': [2, 0.2, 2],
                                     'threebody0': [1, 0.5], 'threebody1': [2, 0.2],
                                     'cutoff_twobody': 2, 'cutoff_threebody': 1},
                         constraints={'twobody0': [False, True]},
                         verbose="DEBUG")
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)


def test_randomization():
    pm = ParameterHelper(species=['O', 'C', 'H'],
                         kernels=['twobody', 'threebody'],
                         allseparate=True,
                         random=True,
                         parameters={'cutoff_twobody': 7,
                                     'cutoff_threebody': 4.5,
                                     'cutoff_manybody': 3},
                         verbose="debug")
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)
    name = pm.find_group('specie', 'O')
    name = pm.find_group('twobody', ['O', 'C'])


def test_from_dict():
    pm = ParameterHelper(species=['O', 'C', 'H'],
                         kernels=['twobody', 'threebody'],
                         allseparate=True,
                         random=True,
                         parameters={'cutoff_twobody': 7,
                                     'cutoff_threebody': 4.5,
                                     'cutoff_manybody': 3},
                         verbose="debug")
    hm = pm.as_dict()
    Parameters.check_instantiation(
        hm['hyps'], hm['cutoffs'], hm['kernels'], hm)

    pm1 = ParameterHelper.from_dict(
        hm, verbose="debug", init_spec=['O', 'C', 'H'])
    print("from_dict")
    hm1 = pm1.as_dict()
    print(hm['hyps'])
    print(hm1['hyps'][:33], hm1['hyps'][33:])

    Parameters.compare_dict(hm, hm1)
