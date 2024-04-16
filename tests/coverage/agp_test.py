"""
Tests for agp python file
"""

import pytest

from magpy.terms import Terms
from magpy.bsf_set import BSFSetBase, OddSet, EvenSet
from magpy.hamiltonian import Hamiltonian
from magpy.agp import AGP

import scipy.sparse as sp
import numpy as np


def test_agp_bad_input():
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
    good_lambda_values = np.array([0.0, 1.0, 2.0])
    good_constant_values = [-1.0]

    base_even_set = EvenSet(input_terms=[X_terms],
                            magnitudes=[1.0],
                            hamiltonian=good_hamiltonian)

    good_operators = [OddSet(input_terms=[Terms(term_type="YZ",
                                                connections=np.array([[0, 1], [1, 0]]).astype(bool))],
                             magnitudes=[1.0],
                             hamiltonian=good_hamiltonian,
                             left_set=base_even_set,
                             left_map=[sp.csr_array([[0.0, 0.0],
                                                     [0.0, 0.0]]),
                                       sp.csr_array([[-2.0, 0.0],
                                                     [0.0, -2.0]])])]
    good_coefficients = np.array([0.5,1.0,0.1])
    good_exact = True
    good_nickname = "Two qubit"

    # Initialising with bad input
    # hamiltonian
    with pytest.raises(TypeError):
        AGP(hamiltonian="bla",
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)

    # lambda_index
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index="bla",
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.raises(ValueError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=3,
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)

    # lambda_values
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values="bla",
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.warns(UserWarning):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=[1],
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=["bla", "bla"],
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.raises(ValueError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=[[1.0], [2.0]],
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)

    # constant_values
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values="bla",
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.warns(UserWarning):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=[1],
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=["bla"],
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.raises(ValueError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=[[1.0], [2.0]],
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.raises(ValueError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=[1.0, 2.0],
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)

    # operators
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators="bla",
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators=["bla"],
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=good_nickname)

    # coefficients
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients="bla",
            exact=good_exact,
            nickname=good_nickname)
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=np.array(["bla"]),
            exact=good_exact,
            nickname=good_nickname)

    # exact
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact="bla",
            nickname=good_nickname)
        
    # nickname
    with pytest.raises(TypeError):
        AGP(hamiltonian=good_hamiltonian,
            lambda_index=good_lambda_index,
            lambda_values=good_lambda_values,
            constant_values=good_constant_values,
            operators=good_operators,
            coefficients=good_coefficients,
            exact=good_exact,
            nickname=1.0)

    # Create good
    good_agp = AGP(hamiltonian=good_hamiltonian,
                   lambda_index=good_lambda_index,
                   lambda_values=good_lambda_values,
                   constant_values=good_constant_values,
                   operators=good_operators,
                   coefficients=good_coefficients,
                   exact=good_exact,
                   nickname=good_nickname)
    
    # Changing input (all immutable except nickname)
    # hamiltonian
    with pytest.raises(ValueError):
        good_agp.hamiltonian = good_hamiltonian

    # lambda_index
    with pytest.raises(ValueError):
        good_agp.lambda_index = good_lambda_index

    # lambda_values
    with pytest.raises(ValueError):
        good_agp.lambda_values = good_lambda_values

    # constant_values
    with pytest.raises(ValueError):
        good_agp.constant_values = good_constant_values

    # operators
    with pytest.raises(ValueError):
        good_agp.operators = good_operators

    # coefficients
    with pytest.raises(ValueError):
        good_agp.coefficients = good_coefficients

    # exact
    with pytest.raises(ValueError):
        good_agp.exact = good_exact

    # nickname
    with pytest.raises(TypeError):
        good_agp.nickname = 1.0


def test_agp():
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
    lambda_values_A = np.array([0.0, 1.0, 2.0])
    constant_values_A = [-1.0]

    base_even_set = EvenSet(input_terms=[X_terms],
                            magnitudes=[1.0],
                            hamiltonian=hamiltonian_A)

    operators_A = [OddSet(input_terms=[Terms(term_type="YZ",
                                             connections=np.array([[0, 1], [1, 0]]).astype(bool))],
                          magnitudes=[1.0],
                          hamiltonian=hamiltonian_A,
                          left_set=base_even_set,
                          left_map=[sp.csr_array([[0.0, 0.0],
                                                  [0.0, 0.0]]),
                                    sp.csr_array([[-2.0, 0.0],
                                                  [0.0, -2.0]])])]
    coefficients_A = np.array([0.5,1.0,0.1])
    exact_A = True

    agp_A = AGP(hamiltonian=hamiltonian_A,
                lambda_index=lambda_index_A,
                lambda_values=lambda_values_A,
                constant_values=constant_values_A,
                operators=operators_A,
                coefficients=coefficients_A,
                exact=exact_A)

    # retrieve attributes
    assert agp_A.hamiltonian == hamiltonian_A
    assert agp_A.hamiltonian == hamiltonian_A
    assert agp_A.lambda_index == lambda_index_A
    assert np.array_equal(agp_A.lambda_values, lambda_values_A)
    assert np.array_equal(agp_A.constant_values, constant_values_A)
    assert agp_A.operators == operators_A
    assert np.array_equal(agp_A.coefficients, coefficients_A)
    assert agp_A.exact == exact_A
    assert agp_A.nickname == "AGP object"
