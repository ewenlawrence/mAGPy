"""
Tests for agp python file
"""

from magpy.terms import Terms
from magpy.bsf_set import BSFSetBase, EvenSet, OddSet
from magpy.hamiltonian import Hamiltonian
from magpy.meta_graph import MetaGraph

import pytest
import numpy as np
import scipy.sparse as sp


def analytical_x_2_qubit(zz, x):
    return -0.5*(1)/((zz**2)+4*(x**2))


def analytical_zz_2_qubit(zz, x):
    return (-x)/(2*(zz**2) + 8*(x**2))


def test_meta_graph_bad_input():
    X_terms = Terms(term_type='X',
                    connections=np.array([1, 1]).astype(bool))
    ZZ_terms = Terms(term_type='ZZ',
                     connections=np.array([[0, 1], [0, 0]]).astype(bool))

    lambda_set = BSFSetBase(input_terms=[X_terms],
                            magnitudes=[1.0])

    j_set = BSFSetBase(input_terms=[ZZ_terms],
                       magnitudes=[1.0])

    # Good inputs
    good_hamiltonian = Hamiltonian(list_of_bsf=[lambda_set, j_set],
                                   variable_names=['lambda', 'J'])
    good_lambda_index = 0
    good_nickname = "test"

    # Initialise with bad inputs
    # Hamiltonian
    with pytest.raises(TypeError):
        MetaGraph(hamiltonian="bla",
                  lambda_index=good_lambda_index,
                  nickname=good_nickname)

    # Lambda Index
    with pytest.raises(TypeError):
        MetaGraph(hamiltonian=good_hamiltonian,
                  lambda_index="bla",
                  nickname=good_nickname)
    with pytest.raises(ValueError):
        MetaGraph(hamiltonian=good_hamiltonian,
                  lambda_index=3,
                  nickname=good_nickname)

    # Nickname
    with pytest.raises(TypeError):
        MetaGraph(hamiltonian=good_hamiltonian,
                  lambda_index=good_lambda_index,
                  nickname=11.1)

    # Create good meta graph
    good_meta_graph = MetaGraph(hamiltonian=good_hamiltonian,
                                lambda_index=good_lambda_index,
                                nickname=good_nickname)

    # Change attributes to bad
    # Hamiltonian
    with pytest.raises(ValueError):
        good_meta_graph.hamiltonian = good_hamiltonian

    # Lambda Index
    with pytest.raises(TypeError):
        good_meta_graph.lambda_index = "bla"
    with pytest.raises(ValueError):
        good_meta_graph.lambda_index = 3

    # Nickname
    with pytest.raises(TypeError):
        good_meta_graph.nickname = 11.1


def test_meta_graph():
    # Inputs
    X_terms = Terms(term_type='X',
                    connections=np.array([1, 1]).astype(bool))
    ZZ_terms = Terms(term_type='ZZ',
                     connections=np.array([[0, 1], [0, 0]]).astype(bool))

    lambda_set = BSFSetBase(input_terms=[X_terms],
                            magnitudes=[1.0])

    j_set = BSFSetBase(input_terms=[ZZ_terms],
                       magnitudes=[1.0])

    # Good inputs
    hamiltonian_A = Hamiltonian(list_of_bsf=[lambda_set, j_set],
                                variable_names=['lambda', 'J'])
    lambda_index_A = 0
    lambda_index_B = 1
    nickname_A = "2 qubit x"
    nickname_B = "2 qubit zz"

    max_odd_A = 10
    max_odd_B = 5
    lambda_values_A = [0.0, 0.5, 1.0]
    lambda_values_B = [-1.0, -2.0]
    constant_values_A = [-1.0]
    constant_values_B = [0.5]

    # Expected outputs
    exp_even_set_0_A = EvenSet(input_terms=[X_terms],
                               magnitudes=[1.0],
                               hamiltonian=hamiltonian_A)
    exp_even_set_0_B = EvenSet(input_terms=[ZZ_terms],
                               magnitudes=[1.0],
                               hamiltonian=hamiltonian_A)
    exp_agp_ops_A = [np.array([[[0b01], [0b11]],
                               [[0b10], [0b11]]],
                              dtype=np.uint8)]
    exp_agp_ops_B = [np.array([[[0b01], [0b11]],
                               [[0b10], [0b11]]],
                              dtype=np.uint8)]
    exp_alphas_A = []
    for lam_A in lambda_values_A:
        alpha = analytical_x_2_qubit(zz=constant_values_A[0], x=lam_A)
        exp_alphas_A.append([alpha, alpha])
    exp_alphas_A = np.array(exp_alphas_A)
    exp_alphas_B = []
    for lam_B in lambda_values_B:
        alpha = analytical_zz_2_qubit(zz=lam_B, x=constant_values_B[0])
        exp_alphas_B.append([alpha, alpha])
    exp_alphas_B = np.array(exp_alphas_B)
    
    # Create a meta graph
    meta_graph_A = MetaGraph(hamiltonian=hamiltonian_A,
                             lambda_index=lambda_index_A,
                             nickname=nickname_A)

    # Retrieve attributes
    assert meta_graph_A.hamiltonian == hamiltonian_A
    assert meta_graph_A.lambda_index == lambda_index_A
    assert meta_graph_A.nickname == nickname_A
    assert meta_graph_A._current_max_set == 0
    assert meta_graph_A._exact == False
    assert meta_graph_A._odd_sets == []
    computed_even_set_0_A = meta_graph_A._even_sets[0]
    assert computed_even_set_0_A.hamiltonian == exp_even_set_0_A.hamiltonian
    assert np.array_equal(computed_even_set_0_A.bsf_array,
                          exp_even_set_0_A.bsf_array)
    assert np.array_equal(computed_even_set_0_A.magnitudes,
                          exp_even_set_0_A.magnitudes)
    assert meta_graph_A._left_lam_maps == []
    assert meta_graph_A._right_lam_maps == []
    assert meta_graph_A._left_const_maps == []
    assert meta_graph_A._right_const_maps == []

    # Check parsing functions
    # max_odd
    with pytest.raises(TypeError):
        meta_graph_A._parse_max_odd("bla")
    with pytest.raises(ValueError):
        meta_graph_A._parse_max_odd(0)

    # lambda_values
    with pytest.raises(TypeError):
        meta_graph_A._parse_lambda_values("bla")
    with pytest.warns(UserWarning):
        meta_graph_A._parse_lambda_values([0, 1, 2])
    with pytest.raises(TypeError):
        meta_graph_A._parse_lambda_values(["bla", "bla"])
    with pytest.raises(ValueError):
        meta_graph_A._parse_lambda_values([[0.1], [0.2]])

    # constant_values
    with pytest.raises(TypeError):
        meta_graph_A._parse_constant_values("bla")
    with pytest.warns(UserWarning):
        meta_graph_A._parse_constant_values([0])
    with pytest.raises(TypeError):
        meta_graph_A._parse_constant_values(["bla", "bla"])
    with pytest.raises(ValueError):
        meta_graph_A._parse_constant_values([[0.1]])
    with pytest.raises(ValueError):
        meta_graph_A._parse_constant_values([1.0, -1.0])

    # Check dense matrix correct
    diag_test = [sp.csr_array([[1, 1],
                               [1, 1]]),
                 sp.csr_array([[2, 2, 2],
                               [2, 2, 2],
                               [2, 2, 2]])]
    off_diag_test = [sp.csr_array([[3, 3, 3],
                                   [3, 3, 3]])]
    exp_dense = [[1, 1, 3, 3, 3],
                 [1, 1, 3, 3, 3],
                 [3, 3, 2, 2, 2],
                 [3, 3, 2, 2, 2],
                 [3, 3, 2, 2, 2]]
    computed_dense = meta_graph_A._create_dense(diag=diag_test,
                                                off_diag=off_diag_test)
    assert np.array_equal(exp_dense, computed_dense)

    # Check single odd_even step
    meta_graph_A._compute_odd_even_step()
    computed_agp_ops_A = meta_graph_A._odd_sets[0].bsf_array
    assert np.array_equal(computed_agp_ops_A, exp_agp_ops_A)

    # Check numpy solution is correct
    agp_A = meta_graph_A.compute_AGP(lambda_values=lambda_values_A,
                                     constant_values=constant_values_A,
                                     max_odd=max_odd_A,
                                     solver='numpy')
    assert np.allclose(agp_A.coefficients, exp_alphas_A)
    assert agp_A.exact

    # Change to B
    meta_graph_A.lambda_index = lambda_index_B
    meta_graph_A.nickname = nickname_B

    # Retrieve attributes
    assert meta_graph_A.hamiltonian == hamiltonian_A
    assert meta_graph_A.lambda_index == lambda_index_B
    assert meta_graph_A.nickname == nickname_B
    assert meta_graph_A._current_max_set == 0
    assert meta_graph_A._exact == False
    assert meta_graph_A._odd_sets == []
    computed_even_set_0_B = meta_graph_A._even_sets[0]
    assert computed_even_set_0_B.hamiltonian == exp_even_set_0_B.hamiltonian
    assert np.array_equal(computed_even_set_0_B.bsf_array,
                          exp_even_set_0_B.bsf_array)
    assert np.array_equal(computed_even_set_0_B.magnitudes,
                          exp_even_set_0_B.magnitudes)
    assert meta_graph_A._left_lam_maps == []
    assert meta_graph_A._right_lam_maps == []
    assert meta_graph_A._left_const_maps == []
    assert meta_graph_A._right_const_maps == []

    # Check single odd_even step
    meta_graph_A._compute_odd_even_step()
    computed_agp_ops_B = meta_graph_A._odd_sets[0].bsf_array
    assert np.array_equal(computed_agp_ops_B, exp_agp_ops_B)

    # Check numpy solution is correct
    agp_B = meta_graph_A.compute_AGP(lambda_values=lambda_values_B,
                                     constant_values=constant_values_B,
                                     max_odd=max_odd_B,
                                     solver='numpy')
    print(agp_B.coefficients)
    print(exp_alphas_B)
    assert np.allclose(agp_B.coefficients, exp_alphas_B)
    assert agp_A.exact


def test_ring():
    # Test against analytical result for ring #TODO
    pass
