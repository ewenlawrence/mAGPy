"""
Tests for terms python file
"""

import pytest

import numpy as np

from magpy.terms import Terms


def test_terms_bad_input():
    # Good inputs
    good_term_type = 'Xy'
    good_connections = np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]]).astype(bool)
    good_nickname = "Test"

    # Initialising bad inputs
    # Term type
    with pytest.raises(TypeError):  # bad type
        Terms(term_type=["bla", "bla"],
              connections=good_connections,
              nickname=good_nickname)
    with pytest.raises(ValueError):  # not only 'X','Y','Z','I'
        Terms(term_type="bla",
              connections=good_connections,
              nickname=good_nickname)

    # Connections
    with pytest.raises(TypeError):  # bad type
        Terms(term_type=good_term_type,
              connections="bla",
              nickname=good_nickname)
    with pytest.raises(TypeError):  # bad dtype of array
        Terms(term_type=good_term_type,
              connections=["bla"],
              nickname=good_nickname)
    with pytest.raises(ValueError):  # wrong size of terms
        Terms(term_type=good_term_type,
              connections=np.array([0]).astype(bool),
              nickname=good_nickname)
    with pytest.raises(ValueError):  # axes not same size
        Terms(term_type=good_term_type,
              connections=np.array([[0, 0]]).astype(bool),
              nickname=good_nickname)
    with pytest.raises(ValueError):  # non zero double shared index
        Terms(term_type=good_term_type,
              connections=np.array([[1, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]]).astype(bool),
              nickname=good_nickname)
    with pytest.warns(UserWarning):  # non bool input, but can be cast to bool
        Terms(term_type=good_term_type,
              connections=[[0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 0]],
              nickname=good_nickname)

    # Nickname
    with pytest.raises(TypeError):  # bad type
        Terms(term_type=good_term_type,
              connections=good_connections,
              nickname=False)

    # Use good inputs
    good_terms = Terms(term_type=good_term_type,
                       connections=good_connections,
                       nickname=good_nickname)

    # Changing to bad inputs
    # Term type
    with pytest.raises(TypeError):  # bad type
        good_terms.term_type = ["bla", "bla"]
    with pytest.raises(ValueError):  # not only 'X','Y','Z','I'
        good_terms.term_type = "bla"
    with pytest.raises(ValueError):  # wrong size of terms
        good_terms.term_type = "XYZ"

    # Connections
    with pytest.raises(TypeError):  # bad type
        good_terms.connections = "bla"
    with pytest.raises(TypeError):  # bad dtype of array
        good_terms.connections = ["bla"]
    with pytest.raises(ValueError):  # wrong size of terms
        good_terms.connections = np.array([0]).astype(bool)
    with pytest.raises(ValueError):  # axes not same size
        good_terms.connections = np.array([[0, 0]]).astype(bool)
    with pytest.raises(ValueError):  # non zero double shared index
        good_terms.connections = np.array([[1, 0, 0],
                                           [0, 0, 0],
                                           [0, 0, 0]]).astype(bool)
    with pytest.warns(UserWarning):  # non bool input, but can be cast to bool
        good_terms.connections = [[0, 1, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]

    # Nickname
    with pytest.raises(TypeError):  # bad type
        good_terms.nickname = False


def test_terms():
    # Inputs A
    term_type_A = 'XY'
    connections_A = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]]).astype(bool)

    # Inputs B
    term_type_B = 'xz'
    connections_B = np.array([[0, 0, 1],
                              [0, 0, 0],
                              [1, 0, 0]]).astype(bool)
    nickname_B = "Test_B"

    # Inputs C
    term_type_C = 'X'
    connections_C = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1]).astype(bool)
    nickname_C = "Test_C"

    # Expected outputs
    exp_bsf_array_A = np.array([[[0b011], [0b010]],
                                [[0b011], [0b001]],
                                [[0b110], [0b100]],
                                [[0b110], [0b010]]],
                               dtype=np.uint8)
    exp_bsf_array_B = np.array([[[0b001], [0b100]],
                                [[0b100], [0b001]]],
                               dtype=np.uint8)
    exp_bsf_array_C = np.array([[[0b0000000001], [0b0000000000]],
                                [[0b1000000000], [0b0000000000]]],
                               dtype=np.uint16)

    # Create terms A
    terms_A = Terms(term_type=term_type_A,
                    connections=connections_A)

    # Retrieve attributes
    assert terms_A.term_type == term_type_A.upper()
    assert np.array_equal(terms_A.connections, connections_A)
    assert terms_A.nickname == "Terms object"
    assert terms_A._term_size == 2

    # Compute bsf array
    assert np.array_equal(terms_A.generate_bsf(), exp_bsf_array_A)

    # Change attributes such that terms_A is terms_B
    terms_A.term_type = term_type_B
    terms_A.connections = connections_B
    terms_A.nickname = nickname_B

    # Retrieve attributes
    assert terms_A.term_type == term_type_B.upper()
    assert np.array_equal(terms_A.connections, connections_B)
    assert terms_A.nickname == nickname_B
    assert terms_A._term_size == 2

    # Compute bsf array
    assert np.array_equal(terms_A.generate_bsf(), exp_bsf_array_B)

    # Create terms C
    terms_C = Terms(term_type=term_type_C,
                    connections=connections_C,
                    nickname=nickname_C)

    # Retrieve attributes
    assert terms_C.term_type == term_type_C.upper()
    assert np.array_equal(terms_C.connections, connections_C)
    assert terms_C.nickname == nickname_C
    assert terms_C._term_size == 1

    # Compute bsf array
    assert np.array_equal(terms_C.generate_bsf(), exp_bsf_array_C)
