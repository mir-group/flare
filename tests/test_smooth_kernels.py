# import pytest
# import numpy as np
# import sys
# from numpy.random import rand
# sys.path.append('../otf_engine')
# import env
# import gp
# import struc
# import kernels
# from kernels import n_body_sc_grad, n_body_mc_grad
# import smooth_kernels
# from smooth_kernels import two_body_smooth, two_body_smooth_grad, \
#     two_body_smooth_en, three_body_quad, three_body_quad_en, \
#     three_body_quad_grad, two_body_quad_mc, two_body_quad_mc_grad, \
#     two_body_quad, two_body_quad_grad, two_plus_three_quad, \
#     two_plus_three_quad_grad

# -----------------------------------------------------------------------------
#                                fixtures
# -----------------------------------------------------------------------------


# # set up two test environments
# @pytest.fixture(scope='module')
# def env1():
#     cell = np.eye(3)
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

#     positions_1 = [np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])]
#     species_1 = ['A', 'A']
#     atom_1 = 0
#     test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
#     env1 = env.ChemicalEnvironment(test_structure_1, atom_1)

#     yield env1

#     del env1


# @pytest.fixture(scope='module')
# def test_structure_1():
#     cell = np.eye(3)
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

#     positions_1 = [np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])]
#     species_1 = ['A', 'A']
#     test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)

#     yield test_structure_1

#     del test_structure_1


# @pytest.fixture(scope='module')
# def env2():
#     cell = np.eye(3)
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

#     positions_2 = [np.array([0, 0, 0]), np.array([0.25, 0.3, 0.4])]
#     species_2 = ['A', 'A']
#     atom_2 = 0
#     test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
#     env2 = env.ChemicalEnvironment(test_structure_2, atom_2)

#     yield env2

#     del env2


# @pytest.fixture(scope='module')
# def test_structure_2():
#     cell = np.eye(3)
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

#     positions_2 = [np.array([0, 0, 0]), np.array([0.25, 0.3, 0.4])]
#     species_2 = ['A', 'A']
#     test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)

#     yield test_structure_2

#     del test_structure_2


# # set up two test environments
# @pytest.fixture(scope='module')
# def delt_env():
#     delta = 1e-8
#     cell = np.eye(3)
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

#     positions_2 = [np.array([delta, 0, 0]), np.array([0.25, 0.3, 0.4])]
#     species_2 = ['A', 'A']
#     atom_2 = 0
#     test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
#     delt_env = env.ChemicalEnvironment(test_structure_2, atom_2)

#     yield delt_env

#     del delt_env


# # -----------------------------------------------------------------------------
# #                          test kernel functions
# # -----------------------------------------------------------------------------


# def test_two_body_quad_mc():
#     cell = np.eye(3)
#     cutoff = 2

#     positions_1 = [np.array([0, 0, 0]),
#                    np.array([rand(), rand(), rand()]),
#                    np.array([rand(), rand(), rand()])]
#     species_1 = ['A', 'B', 'A']
#     atom_1 = 0
#     test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
#     env1 = env.ChemicalEnvironment(test_structure_1, atom_1)

#     positions_2 = [np.array([0, 0, 0]),
#                    np.array([rand(), rand(), rand()]),
#                    np.array([rand(), rand(), rand()])]
#     species_2 = ['A', 'B', 'A']
#     atom_2 = 0
#     test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
#     env2 = env.ChemicalEnvironment(test_structure_2, atom_2)

#     d1 = 3
#     d2 = 2
#     sig1 = 1
#     sig2 = 2
#     ls = 1
#     sign = 0.1
#     hyps = np.array([sig1, sig2, ls])

#     bodies = None

#     grad_test = two_body_quad_mc_grad(env1, env2, bodies, d1, d2, hyps, cutoff)

#     delta = 1e-8
#     tol = 1e-5
#     new_sig1 = sig1 + delta
#     new_sig2 = sig2 + delta
#     new_ls = ls + delta

#     sig1_derv_brute = (two_body_quad_mc(env1, env2, bodies, d1, d2,
#                                         np.array([new_sig1, sig2, ls]),
#                                         cutoff) -
#                        two_body_quad_mc(env1, env2, bodies, d1, d2,
#                                         hyps, cutoff)) / delta

#     sig2_derv_brute = (two_body_quad_mc(env1, env2, bodies, d1, d2,
#                                         np.array([sig1, new_sig2, ls]),
#                                         cutoff) -
#                        two_body_quad_mc(env1, env2, bodies, d1, d2,
#                                         hyps, cutoff)) / delta

#     l_derv_brute = (two_body_quad_mc(env1, env2, bodies, d1, d2,
#                                      np.array([sig1, sig2, new_ls]),
#                                      cutoff) -
#                     two_body_quad_mc(env1, env2, bodies, d1, d2,
#                                      hyps, cutoff)) / delta

#     assert(np.isclose(grad_test[1][0], sig1_derv_brute))
#     assert(np.isclose(grad_test[1][1], sig2_derv_brute))
#     assert(np.isclose(grad_test[1][2], l_derv_brute))


# def test_three_body_smooth_grad(env1, env2):
#     d1 = 3
#     d2 = 2
#     sig = 0.5
#     ls = 0.2
#     d = 0.1
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001
#     hyps = np.array([sig, ls, d])

#     bodies = None

#     _, kern_grad = three_body_quad_grad(env1, env2, bodies,
#                                         d1, d2, hyps, cutoff)

#     delta = 1e-8
#     tol = 1e-5
#     new_sig = sig + delta
#     new_ls = ls + delta

#     sig_derv_brute = (three_body_quad(env1, env2, bodies, d1, d2,
#                       np.array([new_sig, ls, d]), cutoff) -
#                       three_body_quad(env1, env2, bodies, d1, d2,
#                                       hyps, cutoff)) / delta

#     l_derv_brute = (three_body_quad(env1, env2, bodies, d1, d2,
#                                     np.array([sig, new_ls, d]), cutoff) -
#                     three_body_quad(env1, env2, bodies, d1, d2,
#                                     hyps, cutoff)) / delta

#     assert(np.isclose(kern_grad[0], sig_derv_brute, tol))
#     assert(np.isclose(kern_grad[1], l_derv_brute, tol))


# def test_two_body_smooth_grad(env1, env2):
#     d1 = 3
#     d2 = 2
#     sig = 0.5
#     ls = 0.2
#     d = 0.1
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001
#     hyps = np.array([sig, ls, d])

#     bodies = None

#     _, kern_grad = two_body_smooth_grad(env1, env2, bodies,
#                                         d1, d2, hyps, cutoff)

#     delta = 1e-8
#     tol = 1e-5
#     new_sig = sig + delta
#     new_ls = ls + delta
#     new_d = d + delta

#     sig_derv_brute = (two_body_smooth(env1, env2, bodies, d1, d2,
#                       np.array([new_sig, ls, d]), cutoff) -
#                       two_body_smooth(env1, env2, bodies, d1, d2,
#                                       hyps, cutoff)) / delta

#     l_derv_brute = (two_body_smooth(env1, env2, bodies, d1, d2,
#                                     np.array([sig, new_ls, d]), cutoff) -
#                     two_body_smooth(env1, env2, bodies, d1, d2,
#                                     hyps, cutoff)) / delta

#     d_derv_brute = (two_body_smooth(env1, env2, bodies, d1, d2,
#                                     np.array([sig, ls, new_d]), cutoff) -
#                     two_body_smooth(env1, env2, bodies, d1, d2,
#                                     hyps, cutoff)) / delta

#     assert(np.isclose(kern_grad[0], sig_derv_brute, tol))
#     assert(np.isclose(kern_grad[1], l_derv_brute, tol))
#     assert(np.isclose(kern_grad[2], d_derv_brute, tol))


# def test_three_body_smooth():
#     # create env 1
#     delt = 1e-5
#     cell = np.eye(3)
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5]))

#     positions_1 = [np.array([0, 0, 0]),
#                    np.array([0.1, 0.2, 0.3]),
#                    np.array([0.3, 0.2, 0.1])]
#     positions_2 = [np.array([delt, 0, 0]),
#                    np.array([0.1, 0.2, 0.3]),
#                    np.array([0.3, 0.2, 0.1])]
#     positions_3 = [np.array([-delt, 0, 0]),
#                    np.array([0.1, 0.2, 0.3]),
#                    np.array([0.3, 0.2, 0.1])]

#     species_1 = ['A', 'B', 'A']
#     atom_1 = 0
#     test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
#     test_structure_2 = struc.Structure(cell, species_1, positions_2, cutoff)
#     test_structure_3 = struc.Structure(cell, species_1, positions_3, cutoff)

#     env1_1 = env.ChemicalEnvironment(test_structure_1, atom_1)
#     env1_2 = env.ChemicalEnvironment(test_structure_2, atom_1)
#     env1_3 = env.ChemicalEnvironment(test_structure_3, atom_1)

#     # create env 2
#     positions_1 = [np.array([0, 0, 0]),
#                    np.array([0.25, 0.3, 0.4]),
#                    np.array([0.4, 0.3, 0.25])]
#     positions_2 = [np.array([0, delt, 0]),
#                    np.array([0.25, 0.3, 0.4]),
#                    np.array([0.4, 0.3, 0.25])]
#     positions_3 = [np.array([0, -delt, 0]),
#                    np.array([0.25, 0.3, 0.4]),
#                    np.array([0.4, 0.3, 0.25])]

#     species_2 = ['A', 'A', 'B']
#     atom_2 = 0
#     test_structure_1 = struc.Structure(cell, species_2, positions_1, cutoff)
#     test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
#     test_structure_3 = struc.Structure(cell, species_2, positions_3, cutoff)

#     env2_1 = env.ChemicalEnvironment(test_structure_1, atom_2)
#     env2_2 = env.ChemicalEnvironment(test_structure_2, atom_2)
#     env2_3 = env.ChemicalEnvironment(test_structure_3, atom_2)

#     sig = 2
#     ls = 0.5
#     d1 = 1
#     d2 = 2
#     bodies = None

#     hyps = np.array([sig, ls])

#     # check force kernel
#     calc1 = three_body_quad_en(env1_2, env2_2, bodies, hyps, cutoff)
#     calc2 = three_body_quad_en(env1_3, env2_3, bodies, hyps, cutoff)
#     calc3 = three_body_quad_en(env1_2, env2_3, bodies, hyps, cutoff)
#     calc4 = three_body_quad_en(env1_3, env2_2, bodies, hyps, cutoff)

#     kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delt**2)
#     kern_analytical = three_body_quad(env1_1, env2_1, bodies,
#                                       d1, d2, hyps, cutoff)

#     assert(np.isclose(kern_finite_diff, kern_analytical))


# def test_two_body_smooth(env1, env2):
#     # create env 1
#     delt = 1e-5
#     cell = np.eye(3)
#     cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5]))

#     positions_1 = [np.array([0, 0, 0]),
#                    np.array([0.1, 0.2, 0.3]),
#                    np.array([0.3, 0.2, 0.1])]
#     positions_2 = [np.array([delt, 0, 0]),
#                    np.array([0.1, 0.2, 0.3]),
#                    np.array([0.3, 0.2, 0.1])]
#     positions_3 = [np.array([-delt, 0, 0]),
#                    np.array([0.1, 0.2, 0.3]),
#                    np.array([0.3, 0.2, 0.1])]

#     species_1 = ['A', 'B', 'A']
#     atom_1 = 0
#     test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
#     test_structure_2 = struc.Structure(cell, species_1, positions_2, cutoff)
#     test_structure_3 = struc.Structure(cell, species_1, positions_3, cutoff)

#     env1_1 = env.ChemicalEnvironment(test_structure_1, atom_1)
#     env1_2 = env.ChemicalEnvironment(test_structure_2, atom_1)
#     env1_3 = env.ChemicalEnvironment(test_structure_3, atom_1)

#     # create env 2
#     positions_1 = [np.array([0, 0, 0]),
#                    np.array([0.25, 0.3, 0.4]),
#                    np.array([0.4, 0.3, 0.25])]
#     positions_2 = [np.array([0, delt, 0]),
#                    np.array([0.25, 0.3, 0.4]),
#                    np.array([0.4, 0.3, 0.25])]
#     positions_3 = [np.array([0, -delt, 0]),
#                    np.array([0.25, 0.3, 0.4]),
#                    np.array([0.4, 0.3, 0.25])]

#     species_2 = ['A', 'A', 'B']
#     atom_2 = 0
#     test_structure_1 = struc.Structure(cell, species_2, positions_1, cutoff)
#     test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
#     test_structure_3 = struc.Structure(cell, species_2, positions_3, cutoff)

#     env2_1 = env.ChemicalEnvironment(test_structure_1, atom_2)
#     env2_2 = env.ChemicalEnvironment(test_structure_2, atom_2)
#     env2_3 = env.ChemicalEnvironment(test_structure_3, atom_2)

#     sig = 1
#     ls = 1
#     d = 0.1
#     d1 = 1
#     d2 = 2
#     bodies = None

#     hyps = np.array([sig, ls, d])

#     # check force kernel
#     calc1 = two_body_smooth_en(env1_2, env2_2, hyps, cutoff)
#     calc2 = two_body_smooth_en(env1_3, env2_3, hyps, cutoff)
#     calc3 = two_body_smooth_en(env1_2, env2_3, hyps, cutoff)
#     calc4 = two_body_smooth_en(env1_3, env2_2, hyps, cutoff)

#     kern_finite_diff = (calc1 + calc2 - calc3 - calc4) / (4*delt**2)
#     kern_analytical = two_body_smooth(env1_1, env2_1, bodies,
#                                       d1, d2, hyps, cutoff)

#     tol = 1e-4
#     assert(np.isclose(kern_finite_diff, kern_analytical, tol))

#     def test_two_plus_three_quad():
#         # create env 1
#         cell = np.eye(3)
#         cutoff_2 = 0.9
#         cutoff_3 = 0.8

#         positions_1 = [np.array([0, 0, 0]),
#                        np.array([0.1, 0.2, 0.3]),
#                        np.array([0.3, 0.2, 0.1])]
#         species_1 = ['A', 'B', 'A']
#         atom_1 = 0
#         test_structure_1_2 = struc.Structure(cell, species_1,
#                                              positions_1, cutoff_2)
#         env1_2 = env.ChemicalEnvironment(test_structure_1_2, atom_1)
#         test_structure_1_3 = struc.Structure(cell, species_1,
#                                              positions_1, cutoff_3)
#         env1_3 = env.ChemicalEnvironment(test_structure_1_3, atom_1)

#         # create env 2
#         delt = 1e-5
#         cell = np.eye(3)
#         cutoff_2 = 0.9
#         cutoff_3 = 0.8

#         positions_1 = [np.array([0, 0, 0]),
#                        np.array([0.25, 0.3, 0.4]),
#                        np.array([0.4, 0.3, 0.25])]
#         species_1 = ['A', 'B', 'A']
#         atom_1 = 0
#         test_structure_2_2 = struc.Structure(cell, species_1,
#                                              positions_1, cutoff_2)
#         env2_2 = env.ChemicalEnvironment(test_structure_2_2, atom_1)
#         test_structure_2_3 = struc.Structure(cell, species_1,
#                                              positions_1, cutoff_3)
#         env2_3 = env.ChemicalEnvironment(test_structure_2_3, atom_1)

#         d1 = 1
#         d2 = 2
#         hyps = np.array([1, 1, 1, 1])
#         cutoffs = [cutoff_2, cutoff_3]

#         test1 = two_plus_three_quad_grad(env1_2, env2_2, None,
#                                          d1, d2, hyps, cutoffs)
#         test2 = two_body_quad_grad(env1_2, env2_2, None,
#                                    d1, d2, hyps, cutoffs[0])
#         test3 = three_body_quad_grad(env1_3, env2_3, None,
#                                      d1, d2, hyps, cutoffs[1])
#         test4 = two_plus_three_quad(env1_2, env2_2, None,
#                                     d1, d2, hyps, cutoffs)
#         test5 = two_body_quad(env1_2, env2_2, None,
#                               d1, d2, hyps, cutoffs[0])
#         test6 = three_body_quad(env1_3, env2_3, None,
#                                 d1, d2, hyps, cutoffs[1])

#         assert(np.isclose(test1[0], test2[0]+test3[0]))
#         assert(np.isclose(test1[1][0], test2[1][0]))
#         assert(np.isclose(test1[1][1], test2[1][1]))
#         assert(np.isclose(test1[1][2], test3[1][0]))
#         assert(np.isclose(test1[1][3], test3[1][1]))
#         assert(np.isclose(test4, test5+test6))
