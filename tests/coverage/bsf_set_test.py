"""
Tests for bsf_set python file
"""

import pytest

import numpy as np

from magpy.terms import Terms
from magpy.bsf_set import BSFSetBase


def test_bsfsetbase_bad_input():
    # Good inputs
    tmp_terms = Terms(term_type='YY',
                      connections=np.array([[0, 1, 0],
                                            [1, 0, 1],
                                            [0, 1, 0]]).astype(bool))
    good_bsf_array_input = tmp_terms.generate_bsf()
    good_input_array = [Terms(term_type='XX',
                              connections=np.array([[0, 1, 0],
                                                    [1, 0, 1],
                                                    [0, 1, 0]]).astype(bool)),
                        good_bsf_array_input,
                        Terms(term_type='ZZ',
                              connections=np.array([[0, 1, 0],
                                                    [1, 0, 1],
                                                    [0, 1, 0]]).astype(bool))]
    good_magnitudes = [1.0, 1.0, 1.0]
    good_nickname = "test"

    # Initialising bad inputs
    # Input array
    with pytest.raises(TypeError):  # non list type
        BSFSetBase(input_terms="bla",
                   magnitudes=good_magnitudes,
                   nickname=good_nickname)
    with pytest.raises(TypeError):  # wrong data type
        BSFSetBase(input_terms=["bla", "bla", "bla"],
                   magnitudes=good_magnitudes,
                   nickname=good_nickname)
    with pytest.raises(ValueError):  # ndarray contains non integer subclass
        BSFSetBase(input_terms=[np.array(["bla"], dtype=str),
                                np.array(["bla"], dtype=str),
                                np.array(["bla"], dtype=str)],
                   magnitudes=good_magnitudes,
                   nickname=good_nickname)
    with pytest.raises(ValueError):  # ndarray wrong ndim
        BSFSetBase(input_terms=[np.array([1], dtype=int),
                                np.array([1], dtype=int),
                                np.array([1], dtype=int)],
                   magnitudes=good_magnitudes,
                   nickname=good_nickname)
    with pytest.raises(ValueError):  # ndarray wrong shape
        BSFSetBase(input_terms=[np.array([[[1]]], dtype=int),
                                np.array([[[1]]], dtype=int),
                                np.array([[[1]]], dtype=int)],
                   magnitudes=good_magnitudes,
                   nickname=good_nickname)
    with pytest.raises(TypeError):  # different integer types
        BSFSetBase(input_terms=[np.array([[[1], [1]]], dtype=np.uint8),
                                np.array([[[1], [1]]], dtype=np.uint16),
                                np.array([[[1], [1]]], dtype=np.uint8)],
                   magnitudes=good_magnitudes,
                   nickname=good_nickname)
    with pytest.raises(TypeError):  # different integer number
        BSFSetBase(input_terms=[np.array([[[1], [1]]], dtype=np.uint8),
                                np.array([[[1, 1], [1, 1]]], dtype=np.uint8),
                                np.array([[[1], [1]]], dtype=np.uint8)],
                   magnitudes=good_magnitudes,
                   nickname=good_nickname)

    # Magnitudes
    with pytest.raises(TypeError):  # non arraylike input
        BSFSetBase(input_terms=good_input_array,
                   magnitudes="bla",
                   nickname=good_nickname)
    with pytest.raises(TypeError):  # dtype not subclass of floating
        BSFSetBase(input_terms=good_input_array,
                   magnitudes=["bla", "bla", "bla"],
                   nickname=good_nickname)
    with pytest.raises(ValueError):  # wrong dimension
        BSFSetBase(input_terms=good_input_array,
                   magnitudes=[[1.0], [1.0], [1.0]],
                   nickname=good_nickname)
    with pytest.raises(ValueError):  # wrong number of magnitudes
        BSFSetBase(input_terms=good_input_array,
                   magnitudes=[1.0, 1.0, 1.0, 1.0],
                   nickname=good_nickname)

    # Nickname
    with pytest.raises(TypeError):  # non str type
        BSFSetBase(input_terms=good_input_array,
                   magnitudes=good_magnitudes,
                   nickname=False)

    # Use good inputs
    good_bsf_set_base = BSFSetBase(input_terms=good_input_array,
                                   magnitudes=good_magnitudes,
                                   nickname=good_nickname)
    # Changing to bad inputs
    # Input array
    with pytest.raises(TypeError):  # non list type
        good_bsf_set_base.bsf_array = "bla"
    with pytest.raises(ValueError):  # wrong num of bsf inputs
        good_bsf_set_base.bsf_array = [good_bsf_array_input,
                                       good_bsf_array_input,
                                       good_bsf_array_input,
                                       good_bsf_array_input]
    with pytest.raises(TypeError):  # wrong data type
        good_bsf_set_base.bsf_array = ["bla", "bla", "bla"]
    with pytest.raises(ValueError):  # ndarray contains non integer subclass
        good_bsf_set_base.bsf_array = [np.array(["bla"], dtype=str),
                                       np.array(["bla"], dtype=str),
                                       np.array(["bla"], dtype=str)]
    with pytest.raises(ValueError):  # ndarray wrong ndim
        good_bsf_set_base.bsf_array = [np.array([1], dtype=int),
                                       np.array([1], dtype=int),
                                       np.array([1], dtype=int)]
    with pytest.raises(ValueError):  # ndarray wrong shape
        good_bsf_set_base.bsf_array = [np.array([[[1]]], dtype=int),
                                       np.array([[[1]]], dtype=int),
                                       np.array([[[1]]], dtype=int)]
    with pytest.raises(TypeError):  # different integer types
        good_bsf_set_base.bsf_array = [np.array([[[1], [1]]], dtype=np.uint8),
                                       np.array([[[1], [1]]], dtype=np.uint16),
                                       np.array([[[1], [1]]], dtype=np.uint8)]
    with pytest.raises(TypeError):  # different integer number
        good_bsf_set_base.bsf_array = [np.array([[[1], [1]]], dtype=np.uint8),
                                       np.array([[[1, 1], [1, 1]]],
                                                dtype=np.uint8),
                                       np.array([[[1], [1]]], dtype=np.uint8)]

    # Magnitudes
    with pytest.raises(TypeError):  # non arraylike input
        good_bsf_set_base.magnitudes = "bla"
    with pytest.raises(TypeError):  # dtype not subclass of floating
        good_bsf_set_base.magnitudes = ["bla", "bla", "bla"]
    with pytest.raises(ValueError):  # wrong dimension
        good_bsf_set_base.magnitudes = [[1.0], [1.0], [1.0]]
    with pytest.raises(ValueError):  # wrong number of magnitudes
        good_bsf_set_base.magnitudes = [1.0, 1.0, 1.0, 1.0]

    # Nickname
    with pytest.raises(TypeError):  # non str type
        good_bsf_set_base.nickname = False


def test_bsf_set_base():
    # Inputs A
    input_array_A = [Terms(term_type='XX',
                           connections=np.array([[0, 1, 0],
                                                 [1, 0, 1],
                                                 [0, 1, 0]]).astype(bool)),
                     Terms(term_type='YY',
                           connections=np.array([[0, 1, 0],
                                                 [1, 0, 1],
                                                 [0, 1, 0]]).astype(bool)),
                     Terms(term_type='ZZ',
                           connections=np.array([[0, 1, 0],
                                                 [1, 0, 1],
                                                 [0, 1, 0]]).astype(bool))]
    magnitudes_A = [1.0, 1.0, 1.0]

    # Inputs B
    input_array_B = [Terms(term_type='XX',
                           connections=np.array([[0, 0, 1],
                                                 [0, 0, 0],
                                                 [1, 0, 0]]).astype(bool)),
                     Terms(term_type='YZ',
                           connections=np.array([[0, 1, 0],
                                                 [1, 0, 1],
                                                 [0, 1, 0]]).astype(bool)),
                     Terms(term_type='ZZ',
                           connections=np.array([[0, 0, 1],
                                                 [0, 0, 0],
                                                 [1, 0, 0]]).astype(bool))]
    magnitudes_B = [1.0, 2.0, -1.0]
    nickname_B = "Test B"

    # Expected bsf array
    expected_bsf_A = [np.array([[[0b011], [0b000]],
                                [[0b011], [0b000]],
                                [[0b110], [0b000]],
                                [[0b110], [0b000]]],
                               dtype=np.uint8),
                      np.array([[[0b011], [0b011]],
                                [[0b011], [0b011]],
                                [[0b110], [0b110]],
                                [[0b110], [0b110]]],
                               dtype=np.uint8),
                      np.array([[[0b000], [0b011]],
                                [[0b000], [0b011]],
                                [[0b000], [0b110]],
                                [[0b000], [0b110]]],
                               dtype=np.uint8)]
    expected_bsf_B = [np.array([[[0b101], [0b000]],
                                [[0b101], [0b000]]],
                               dtype=np.uint8),
                      np.array([[[0b001], [0b011]],
                                [[0b010], [0b011]],
                                [[0b010], [0b110]],
                                [[0b100], [0b110]]],
                               dtype=np.uint8),
                      np.array([[[0b000], [0b101]],
                                [[0b000], [0b101]]],
                               dtype=np.uint8)]

    # Create bsf_set_base A
    bsf_set_base_A = BSFSetBase(input_terms=input_array_A,
                                magnitudes=magnitudes_A)

    # Retrieve attributes
    for bsf_A, exp_A in zip(bsf_set_base_A.bsf_array, expected_bsf_A):
        assert np.array_equal(bsf_A, exp_A)
    assert np.array_equal(bsf_set_base_A.magnitudes, magnitudes_A)
    assert bsf_set_base_A.nickname == "BSFSetBase object"
    assert bsf_set_base_A._int_num == 1
    assert bsf_set_base_A._int_type == np.uint8

    # Change attributes such that terms_A is terms_B
    bsf_set_base_A.bsf_array = input_array_B
    bsf_set_base_A.magnitudes = magnitudes_B
    bsf_set_base_A.nickname = nickname_B

    # Retrieve attributes
    for bsf_A, exp_B in zip(bsf_set_base_A.bsf_array, expected_bsf_B):
        assert np.array_equal(bsf_A, exp_B)
    assert np.array_equal(bsf_set_base_A.magnitudes, magnitudes_B)
    assert bsf_set_base_A.nickname == "Test B"
    assert bsf_set_base_A._int_num == 1
    assert bsf_set_base_A._int_type == np.uint8
