import pytest
import numpy as np

from pytest import raises

from flare.utils.parameter_helper import ParameterHelper
from flare.utils.parameters import Parameters


def test_initialization():
    """
    simplest senario
    """
    pm = ParameterHelper(
        kernels=["twobody", "threebody"],
        parameters={
            "twobody": [1, 0.5],
            "threebody": [1, 0.5],
            "cutoff_twobody": 2,
            "cutoff_threebody": 1,
            "noise": 0.05,
        },
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


@pytest.mark.parametrize("ones", [True, False])
def test_initialization2(ones):
    """
    check ones, random
    """
    pm = ParameterHelper(
        kernels=["twobody", "threebody"],
        parameters={"cutoff_twobody": 2, "cutoff_threebody": 1, "noise": 0.05},
        ones=ones,
        random=not ones,
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


@pytest.mark.parametrize("ones", [True, False])
def test_initialization_allsep(ones):
    """
    check ones, random
    """
    specie_list = ["C", "H", "O"]
    pm = ParameterHelper(
        species=specie_list,
        kernels=["twobody", "threebody"],
        parameters={"cutoff_twobody": 2, "cutoff_threebody": 1, "noise": 0.05},
        allseparate=True,
        ones=ones,
        random=not ones,
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)
    name_list = []
    for i in range(3):
        name = pm.find_group("specie", specie_list[i])
        assert name not in name_list
        name_list += [name]
    name_list = []
    for i in range(3):
        for j in range(i, 3):
            name = pm.find_group("twobody", [specie_list[i], specie_list[j]])
            assert name not in name_list
            name_list += [name]

    with raises(RuntimeError):
        pm = ParameterHelper(
            species=[],
            kernels=["twobody", "threebody"],
            parameters={"cutoff_twobody": 2, "cutoff_threebody": 1, "noise": 0.05},
            allseparate=True,
            ones=ones,
            random=not ones,
        )


def test_initialization3():
    """check group definition"""
    pm = ParameterHelper(
        species=["O", "C", "H"],
        kernels={
            "twobody": [["*", "*"], ["O", "O"]],
            "threebody": [["*", "*", "*"], ["O", "O", "O"]],
        },
        parameters={
            "twobody0": [1, 0.5],
            "twobody1": [2, 0.2],
            "threebody0": [1, 0.5],
            "threebody1": [2, 0.2],
            "cutoff_twobody": 2,
            "cutoff_threebody": 1,
        },
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


def test_initialization4():
    """check cut3b"""
    pm = ParameterHelper(
        species=["O", "C", "H"],
        kernels={
            "twobody": [["*", "*"], ["O", "O"]],
            "threebody": [["*", "*", "*"], ["O", "O", "O"]],
        },
        cutoff_groups={"cut3b": [["*", "*"], ["O", "O"]]},
        parameters={
            "twobody0": [1, 0.5],
            "twobody1": [2, 0.2],
            "threebody0": [1, 0.5],
            "threebody1": [2, 0.2],
            "cut3b0": 5,
            "cutoff_twobody": 2,
        },
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


def test_initialization5():
    """check universal"""
    pm = ParameterHelper(
        species=["O", "C", "H"],
        kernels={
            "twobody": [["*", "*"], ["O", "O"]],
            "threebody": [["*", "*", "*"], ["O", "O", "O"]],
        },
        parameters={
            "sigma": 1,
            "lengthscale": 0.5,
            "cutoff_threebody": 3,
            "cutoff_twobody": 2,
        },
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)

    pm = ParameterHelper(
        kernels=["twobody", "threebody"],
        parameters={
            "sigma": 1.0,
            "lengthscale": 0.5,
            "cutoff_twobody": 2,
            "cutoff_threebody": 1,
            "noise": 0.05,
        },
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


def test_generate_by_line():

    pm = ParameterHelper(verbose="DEBUG")
    pm.define_group("specie", "O", ["O"])
    pm.define_group("specie", "C", ["C"])
    pm.define_group("specie", "H", ["H"])
    pm.define_group("twobody", "**", ["C", "H"])
    pm.define_group("twobody", "OO", ["O", "O"], atomic_str=True)
    pm.define_group("threebody", "***", ["O", "O", "C"])
    pm.define_group("threebody", "OOO", ["O", "O", "O"])
    pm.define_group("manybody", "1.5", ["C", "H"])
    pm.define_group("manybody", "1.5", ["C", "O"])
    pm.define_group("manybody", "1.5", ["O", "H"])
    pm.define_group("manybody", "2", ["O", "O"])
    pm.define_group("manybody", "2", ["H", "O"])
    pm.define_group("manybody", "2.8", ["O", "O"])
    pm.set_parameters("**", [1, 0.5])
    pm.set_parameters("OO", [1, 0.5])
    pm.set_parameters("***", [1, 0.5])
    pm.set_parameters("OOO", [1, 0.5])
    pm.set_parameters("1.5", [1, 0.5, 1.5])
    pm.set_parameters("2", [1, 0.5, 2])
    pm.set_parameters("2.8", [1, 0.5, 2.8])
    pm.set_constraints("2", [True, False])
    pm.set_constraints("2.8", False)
    pm.set_parameters("cutoff_twobody", 5)
    pm.set_parameters("cutoff_threebody", 4)
    pm.set_parameters("cutoff_manybody", 3)
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


def test_generate_by_line2():

    pm = ParameterHelper(verbose="DEBUG")
    pm.define_group("specie", "O", ["O"])
    pm.define_group("specie", "rest", ["C", "H"])
    pm.define_group("twobody", "**", ["*", "*"])
    pm.define_group("twobody", "OO", ["O", "O"])
    pm.define_group("threebody", "***", ["*", "*", "*"])
    pm.define_group("threebody", "Oall", ["O", "O", "O"])
    pm.set_parameters("**", [1, 0.5])
    pm.set_parameters("OO", [1, 0.5])
    pm.set_parameters("Oall", [1, 0.5])
    pm.set_parameters("***", [1, 0.5])
    pm.set_parameters("cutoff_twobody", 5)
    pm.set_parameters("cutoff_threebody", 4)
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


def test_generate_by_list():

    pm = ParameterHelper(verbose="DEBUG")
    pm.list_groups("specie", ["O", ["C", "N"], "H"])
    pm.list_groups("twobody", [["*", "*"], ["O", "O"]])
    pm.list_groups("threebody", [["*", "*", "*"], ["O", "O", "O"]])
    pm.list_parameters(
        {
            "twobody0": [1, 0.5],
            "twobody1": [2, 0.2],
            "threebody0": [1, 0.5],
            "threebody1": [2, 0.2],
            "cutoff_twobody": 2,
            "cutoff_threebody": 1,
        }
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


def test_generate_by_list2():

    pm = ParameterHelper(verbose="DEBUG")
    pm.list_groups("specie", {"s1": "O", "s2": ["C", "N"], "s3": "H"})
    pm.list_groups("twobody", {"t0": ["*", "*"], "t1": [["s1", "s1"], ["s1", "s3"]]})
    pm.list_groups("threebody", [["*", "*", "*"], ["s1", "s1", "s1"]])
    pm.list_parameters(
        {
            "t0": [1, 0.5],
            "t1": [2, 0.2],
            "threebody0": [1, 0.5],
            "threebody1": [2, 0.2],
            "cutoff_twobody": 2,
            "cutoff_threebody": 1,
        }
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


def test_generate_by_list_error():

    pm = ParameterHelper(verbose="DEBUG")
    pm.list_groups("specie", ["O", ["C", "N"], "H"])
    with raises(RuntimeError):
        pm.list_groups("specie", ["O", "C", "H"])

    pm = ParameterHelper(verbose="DEBUG")
    with raises(RuntimeError):
        pm.list_groups("specie", "O")

    pm = ParameterHelper(verbose="DEBUG")
    with raises(RuntimeError):
        pm.list_groups("specie", "O")


def test_opt():
    pm = ParameterHelper(
        species=["O", "C", "H"],
        kernels={
            "twobody": [["*", "*"], ["O", "O"]],
            "threebody": [["*", "*", "*"], ["O", "O", "O"]],
        },
        parameters={
            "twobody0": [1, 0.5, 1],
            "twobody1": [2, 0.2, 2],
            "threebody0": [1, 0.5],
            "threebody1": [2, 0.2],
            "cutoff_twobody": 2,
            "cutoff_threebody": 1,
        },
        constraints={"twobody0": [False, True]},
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)


def test_from_dict():
    pm = ParameterHelper(
        species=["O", "C", "H"],
        kernels=["twobody", "threebody"],
        allseparate=True,
        random=True,
        parameters={"cutoff_twobody": 7, "cutoff_threebody": 4.5, "cutoff_manybody": 3},
        verbose="debug",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)

    pm1 = ParameterHelper.from_dict(hm, verbose="debug", init_spec=["O", "C", "H"])
    hm1 = pm1.as_dict()
    Parameters.compare_dict(hm, hm1)


def test_constraints1():
    """
    simplest senario
    """
    pm = ParameterHelper(
        species=["O", "C", "H"],
        kernels={
            "twobody": [["*", "*"], ["O", "O"]],
            "threebody": [["*", "*", "*"], ["O", "O", "O"]],
        },
        parameters={
            "twobody0": [1, 0.5],
            "twobody1": [2, 0.2],
            "threebody0": [1, 0.5],
            "threebody1": [2, 0.2],
            "cutoff_twobody": 2,
            "cutoff_threebody": 1,
        },
        constraints={
            "twobody0": [True, False],
            "threebody0": [False, True],
            "noise": False,
        },
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)
    assert hm["train_noise"] is False
    hyps = hm["hyps"]
    assert len(hyps) == 6
    assert hyps[0] == 1
    assert hyps[1] == 2
    assert hyps[2] == 0.2
    assert hyps[3] == 2
    assert hyps[4] == 0.5
    assert hyps[5] == 0.2


def test_constraints2():
    """
    simplest senario
    """
    pm = ParameterHelper(
        kernels=["twobody", "threebody"],
        parameters={
            "twobody": [1, 0.5],
            "threebody": [1, 0.5],
            "cutoff_twobody": 2,
            "cutoff_threebody": 1,
            "noise": 0.05,
        },
        constraints={"twobody": [True, False]},
        verbose="DEBUG",
    )
    hm = pm.as_dict()
    Parameters.check_instantiation(hm["hyps"], hm["cutoffs"], hm["kernels"], hm)
    hyps = hm["hyps"]
    assert hyps[0] == 1
    assert hyps[1] == 1


def test_check_one_conflict():
    """
    simplest senario
    """
    with raises(RuntimeError):
        pm = ParameterHelper(
            kernels=["twobody", "threebody"],
            parameters={"cutoff_twobody": 2, "cutoff_threebody": 1, "noise": 0.05},
            ones=True,
            random=True,
            verbose="DEBUG",
        )

    with raises(RuntimeError):
        pm = ParameterHelper(
            kernels=["twobody", "threebody"],
            parameters={
                "sigma": 0.5,
                "lengthscale": 1.0,
                "cutoff_twobody": 2,
                "cutoff_threebody": 1,
                "noise": 0.05,
            },
            ones=True,
            random=False,
            verbose="DEBUG",
        )
