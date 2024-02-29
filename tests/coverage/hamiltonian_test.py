"""
Tests for hamiltonian python file
"""

import pytest

import numpy as np

from magpy.terms import Terms
from magpy.bsf_set import BSFSetBase
from magpy.hamiltonian import Hamiltonian


def test_hamiltonian_bad_input():
    # Good inputs
    terms_XX = Terms(term_type='XX',
                     connections=np.array([[0, 1, 0],
                                           [1, 0, 1],
                                           [0, 1, 0]]).astype(bool))
    terms_YY = Terms(term_type='YY',
                     connections=np.array([[0, 1, 0],
                                           [1, 0, 1],
                                           [0, 1, 0]]).astype(bool))
    terms_ZZ = Terms(term_type='ZZ',
                     connections=np.array([[0, 1, 0],
                                           [1, 0, 1],
                                           [0, 1, 0]]).astype(bool))
    bsf_set_XX = BSFSetBase(input_terms=[terms_XX,
                                         terms_YY],
                            magnitudes=[1.0, 1.0])

    bsf_set_Z = BSFSetBase(input_terms=[terms_ZZ],
                           magnitudes=[1.0])

    good_list_of_bsf = [bsf_set_XX, bsf_set_Z]
    good_variable_names = ["XX", "Z"]
    good_nickname = "test"

    # Initialising with bad inputs
    # list_of_bf
    with pytest.raises(TypeError):  # non list inputs
        Hamiltonian(list_of_bsf="bla",
                    variable_names=good_variable_names,
                    nickname=good_nickname)
    with pytest.raises(TypeError):  # non BSFBaseSet elements
        Hamiltonian(list_of_bsf=["bla"],
                    variable_names=good_variable_names,
                    nickname=good_nickname)

    # variable_names
    with pytest.raises(TypeError):  # non list inputs
        Hamiltonian(list_of_bsf=good_list_of_bsf,
                    variable_names="bla",
                    nickname=good_nickname)
    with pytest.raises(TypeError):  # non str elements
        Hamiltonian(list_of_bsf=good_list_of_bsf,
                    variable_names=[False, False],
                    nickname=good_nickname)
    with pytest.raises(ValueError):  # wrong length
        Hamiltonian(list_of_bsf=good_list_of_bsf,
                    variable_names=["XX"],
                    nickname=good_nickname)

    # nickname
    with pytest.raises(TypeError):  # bad type
        Hamiltonian(list_of_bsf=good_list_of_bsf,
                    variable_names=good_variable_names,
                    nickname=False)

    # set good inputs
    good_hamiltonian = Hamiltonian(list_of_bsf=good_list_of_bsf,
                                   variable_names=good_variable_names,
                                   nickname=good_nickname)

    # Change to bad inputs
    # list_of_bsf
    with pytest.raises(TypeError):  # non list inputs
        good_hamiltonian.list_of_bsf = "bla"
    with pytest.raises(TypeError):  # non BSFBaseSet elements
        good_hamiltonian.list_of_bsf = ["bla"]
    with pytest.raises(ValueError):  # wrong length
        good_hamiltonian.list_of_bsf = [bsf_set_XX]

    # variable_names
    with pytest.raises(TypeError):  # non list inputs
        good_hamiltonian.variable_names = "bla"
    with pytest.raises(TypeError):  # non str elements
        good_hamiltonian.variable_names = [False, False]
    with pytest.raises(ValueError):  # wrong length
        good_hamiltonian.variable_names = ["XX"]

    # nickname
    with pytest.raises(TypeError):  # bad type
        good_hamiltonian.nickname = False


def test_hamiltonian():
    terms_XX = Terms(term_type='XX',
                     connections=np.array([[0, 1, 0],
                                           [1, 0, 1],
                                           [0, 1, 0]]).astype(bool))
    terms_YY = Terms(term_type='YY',
                     connections=np.array([[0, 1, 0],
                                           [1, 0, 1],
                                           [0, 1, 0]]).astype(bool))
    terms_ZZ = Terms(term_type='ZZ',
                     connections=np.array([[0, 1, 0],
                                           [1, 0, 1],
                                           [0, 1, 0]]).astype(bool))
    bsf_set_XX = BSFSetBase(input_terms=[terms_XX,
                                         terms_YY],
                            magnitudes=[1.0, 1.0])
    bsf_set_X = BSFSetBase(input_terms=[terms_XX],
                           magnitudes=[1.0])
    bsf_set_Y = BSFSetBase(input_terms=[terms_YY],
                           magnitudes=[1.0])
    bsf_set_Z = BSFSetBase(input_terms=[terms_ZZ],
                           magnitudes=[1.0])

    # Inputs A
    list_of_bsf_A = [bsf_set_XX, bsf_set_Z]
    variable_names_A = ["XX", "Z"]

    # Inputs B
    list_of_bsf_B = [bsf_set_X, bsf_set_Y]
    variable_names_B = ["X", "Y"]
    nickname_B = "XY model"

    # Create hamiltonian A
    hamiltonian_A = Hamiltonian(list_of_bsf=list_of_bsf_A,
                                variable_names=variable_names_A)

    # Retrieve inputs
    for bsf_A, exp_bsf_A in zip(hamiltonian_A.list_of_bsf, list_of_bsf_A):
        assert bsf_A == exp_bsf_A
    for var_A, exp_var_A in zip(hamiltonian_A.variable_names, variable_names_A):
        assert var_A == exp_var_A
    assert hamiltonian_A.nickname == "Hamiltonian object"

    # Change to inputs B
    hamiltonian_A.list_of_bsf = list_of_bsf_B
    hamiltonian_A.variable_names = variable_names_B
    hamiltonian_A.nickname = nickname_B

    # Retrieve inputs
    for bsf_A, exp_bsf_B in zip(hamiltonian_A.list_of_bsf, list_of_bsf_B):
        assert bsf_A == exp_bsf_B
    for var_A, exp_var_B in zip(hamiltonian_A.variable_names, variable_names_B):
        assert var_A == exp_var_B
    assert hamiltonian_A.nickname == nickname_B
