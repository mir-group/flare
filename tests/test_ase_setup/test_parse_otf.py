import os
import sys
import numpy as np
from flare.otf_parser import OtfAnalysis
from flare.kernels import two_plus_three_body, two_plus_three_body_grad
from flare.mc_simple import two_plus_three_body_mc, two_plus_three_body_mc_grad
from flare.env import AtomicEnvironment
from flare.predict import predict_on_structure


def test_parse_header():

    header_dict = OtfAnalysis('test_files/VelocityVerlet.log').header

    assert header_dict['frames'] == 0
    assert header_dict['atoms'] == 4
    assert header_dict['species_set'] == {'Ag', 'I'}
    assert header_dict['dt'] == .001
    assert header_dict['kernel'] == 'two_plus_three_body_mc'
    assert header_dict['n_hyps'] == 5
    assert header_dict['algo'] == 'BFGS'
    assert np.equal(header_dict['cell'],
                    np.array([[7.71,  0.   , 0.   ],
                              [0.  ,  3.855, 0.   ],
                              [0.  ,  0.   , 3.855]])).all()


def test_gp_parser():
    """
    Test the capability of otf parser to read GP/DFT info
    :return:
    """

    parsed = OtfAnalysis('test_files/VelocityVerlet.log')
    assert (parsed.gp_species_list == [['Ag', 'I']*2])

    gp_positions = parsed.gp_position_list
    assert len(gp_positions) == 1

    pos1 = 1.819218
    pos2 = -0.141231
    assert(pos1 == gp_positions[0][-1][1])
    assert(pos2 == gp_positions[-1][0][2])

    force1 = -0.424080
    force2 = 0.498037 
    assert(force1 == parsed.gp_force_list[0][-1][1])
    assert(force2 == parsed.gp_force_list[-1][0][2])


def test_md_parser():
    """
    Test the capability of otf parser to read MD info
    :return:
    """

    parsed = OtfAnalysis('test_files/VelocityVerlet.log')

    pos1 = -0.172516 
    assert(pos1 == parsed.position_list[0][0][2])
    assert(len(parsed.position_list[0]) == 4)

def test_output_md_structures():

    parsed = OtfAnalysis('test_files/VelocityVerlet.log')

    positions = parsed.position_list
    forces = parsed.force_list

    structures = parsed.output_md_structures()

    assert np.isclose(structures[-1].positions, positions[-1]).all()
    assert np.isclose(structures[-1].forces, forces[-1]).all()


def test_replicate_gp():
    """
    Based on gp_test_al.out, ensures that given hyperparameters and DFT calls
    a GP model can be reproduced and correctly re-predict forces and
    uncertainties
    :return:
    """

    parsed = OtfAnalysis('test_files/VelocityVerlet.log')

    positions = parsed.position_list
    forces = parsed.force_list

    gp_model = parsed.make_gp(kernel=two_plus_three_body_mc,
                              kernel_grad=two_plus_three_body_mc_grad)

    structures = parsed.output_md_structures()

    assert np.isclose(structures[-1].positions, positions[-1]).all()
    assert np.isclose(structures[-1].forces, forces[-1]).all()

    final_structure = structures[-1]

    pred_for, pred_stds = predict_on_structure(final_structure, gp_model)

    assert np.isclose(final_structure.forces, pred_for, rtol=1e-3).all()
    assert np.isclose(final_structure.stds, pred_stds, rtol=1e-3).all()

    set_of_structures = structures[-3:-1]
    for structure in set_of_structures:
        pred_for, pred_stds = predict_on_structure(structure, gp_model)
        assert np.isclose(structure.forces, pred_for, rtol=1e-3, atol=1e-6).all()
        assert np.isclose(structure.stds, pred_stds, rtol=1e-3, atol=1e-6).all()
