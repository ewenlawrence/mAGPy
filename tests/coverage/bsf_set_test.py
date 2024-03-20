"""
Tests for bsf_set python file
"""

import pytest

import numpy as np
import scipy.sparse as sp

from magpy.terms import Terms
from magpy.bsf_set import BSFSetBase, LinkedBSFSet, EvenSet, OddSet
from magpy.hamiltonian import Hamiltonian


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


def test_linked_bsf_set_bad_input():
    terms_X = Terms(term_type='X',
                    connections=np.array([1, 1]).astype(bool))
    terms_ZZ = Terms(term_type='ZZ',
                     connections=np.array([[0, 1],
                                          [0, 0]]).astype(bool))

    bsf_set_X = BSFSetBase(input_terms=[terms_X],
                           magnitudes=[1.0])
    bsf_set_Z = BSFSetBase(input_terms=[terms_ZZ],
                           magnitudes=[1.0])

    # Good inputs
    good_input_array = [Terms(term_type='XX',
                              connections=np.array([[0, 1, 0],
                                                    [1, 0, 1],
                                                    [0, 1, 0]]).astype(bool))]
    good_magnitudes = [1.0]
    good_hamiltonian = Hamiltonian(list_of_bsf=[bsf_set_X, bsf_set_Z],
                                   variable_names=["X", "ZZ"])
    good_left_set = LinkedBSFSet(input_terms=good_input_array,
                                 magnitudes=good_magnitudes,
                                 hamiltonian=good_hamiltonian)
    good_nickname = "test"

    other_hamiltonian = Hamiltonian(list_of_bsf=[bsf_set_X],
                                    variable_names=["X"])
    bad_linked_set = LinkedBSFSet(input_terms=good_input_array,
                                  magnitudes=good_magnitudes,
                                  hamiltonian=other_hamiltonian)

    # Initialise with bad inputs
    # Hamiltonian
    with pytest.raises(TypeError):  # not hamiltonian instance
        LinkedBSFSet(input_terms=good_input_array,
                     magnitudes=good_magnitudes,
                     hamiltonian="bla",
                     left_set=good_left_set,
                     nickname=good_nickname)

    # left_set
    with pytest.raises(TypeError):  # not LinkedBSFSet
        LinkedBSFSet(input_terms=good_input_array,
                     magnitudes=good_magnitudes,
                     hamiltonian=good_hamiltonian,
                     left_set="bla",
                     nickname=good_nickname)
    with pytest.raises(ValueError):  # hamiltonian not the same
        LinkedBSFSet(input_terms=good_input_array,
                     magnitudes=good_magnitudes,
                     hamiltonian=good_hamiltonian,
                     left_set=bad_linked_set,
                     nickname=good_nickname)

    # Create good LinkedBSFSet
    good_linked_bsf_set = LinkedBSFSet(input_terms=good_input_array,
                                       magnitudes=good_magnitudes,
                                       hamiltonian=good_hamiltonian,
                                       left_set=good_left_set,
                                       nickname=good_nickname)

    # right_set
    with pytest.raises(TypeError):  # not LinkedBSFSet
        good_linked_bsf_set.right_set = "bla"
    with pytest.raises(ValueError):  # hamiltonian not the same
        good_linked_bsf_set.right_set = bad_linked_set

    # Set good right_set
    good_right_set = LinkedBSFSet(input_terms=good_input_array,
                                  magnitudes=good_magnitudes,
                                  hamiltonian=good_hamiltonian,
                                  left_set=good_linked_bsf_set,
                                  nickname=good_nickname)
    good_linked_bsf_set.right_set = good_right_set

    # Check immutable
    with pytest.raises(ValueError):  # hamiltonian change
        good_linked_bsf_set.hamiltonian = good_hamiltonian

    with pytest.raises(ValueError):  # left_set change
        good_linked_bsf_set.left_set = good_left_set

    with pytest.raises(ValueError):  # right_set change
        good_linked_bsf_set.right_set = good_right_set


def test_linked_bsf_set():
    terms_X = Terms(term_type='X',
                    connections=np.array([1, 1]).astype(bool))
    terms_ZZ = Terms(term_type='ZZ',
                     connections=np.array([[0, 1],
                                          [0, 0]]).astype(bool))

    bsf_set_X = BSFSetBase(input_terms=[terms_X],
                           magnitudes=[2.0])
    bsf_set_Z = BSFSetBase(input_terms=[terms_ZZ],
                           magnitudes=[1.0])

    # Hamiltonian inputs
    list_of_bsf = [bsf_set_X, bsf_set_Z]
    variable_names = ["X", "ZZ"]
    nickname = "Two qubits"

    ham = Hamiltonian(list_of_bsf=list_of_bsf,
                      variable_names=variable_names,
                      nickname=nickname)

    # Inputs A
    terms_A = Terms(term_type='X',
                    connections=np.array([1, 1]).astype(bool))
    magnitudes_A = [1.0]

    # Input B
    terms_B = Terms(term_type='ZZ',
                    connections=np.array([[0, 1],
                                          [0, 0]]).astype(bool))
    magnitudes_B = [1.0]
    nickname_B = "test B"

    # Expected output A
    exp_bsf_array_A = [np.array([[[1], [0]], [[2], [0]]], dtype=np.int8)]
    exp_commuted_terms_A = [np.array([[[1], [3]], [[2], [3]]], dtype=np.int8)]
    exp_maps_A = [sp.csr_array([[0.0, 0.0],
                               [0.0, 0.0]]),
                  sp.csr_array([[1.0, 0.0],
                                [0.0, 1.0]])]

    # Expected outputs B
    exp_bsf_array_B = [np.array([[[0], [3]]], dtype=np.int8)]
    exp_commuted_terms_B = [np.array([[[1], [3]], [[2], [3]]], dtype=np.int8)]
    exp_maps_B = [sp.csr_array([[-2.0],
                               [-2.0]]),
                  sp.csr_array([[0.0],
                                [0.0]])]

    # Setup A
    linkedbsfset_A = LinkedBSFSet(input_terms=[terms_A],
                                  magnitudes=magnitudes_A,
                                  hamiltonian=ham)

    # Retrieve attributes
    for bsf_A, exp_A in zip(linkedbsfset_A.bsf_array, exp_bsf_array_A):
        assert np.array_equal(bsf_A, exp_A)
    assert np.array_equal(linkedbsfset_A.magnitudes, magnitudes_A)
    assert linkedbsfset_A.nickname == "LinkedBSFSet object"
    assert linkedbsfset_A._int_num == 1
    assert linkedbsfset_A._int_type == np.uint8

    # Test commute
    commuted_terms_A, maps_A = linkedbsfset_A.commute()
    assert np.array_equal(commuted_terms_A, exp_commuted_terms_A[0])
    for map_A, exp_map_A in zip(maps_A, exp_maps_A):
        assert np.array_equal(map_A.todense(), exp_map_A.todense())

    # Setup B
    linkedbsfset_B = LinkedBSFSet(input_terms=[terms_B],
                                  magnitudes=magnitudes_B,
                                  hamiltonian=ham,
                                  nickname=nickname_B)

    # Retrieve attributes
    for bsf_B, exp_B in zip(linkedbsfset_B.bsf_array, exp_bsf_array_B):
        assert np.array_equal(bsf_B, exp_B)
    assert np.array_equal(linkedbsfset_B.magnitudes, magnitudes_B)
    assert linkedbsfset_B.nickname == nickname_B
    assert linkedbsfset_B._int_num == 1
    assert linkedbsfset_B._int_type == np.uint8

    # Test commute
    commuted_terms_B, maps_B = linkedbsfset_B.commute()
    assert np.array_equal(commuted_terms_B, exp_commuted_terms_B[0])
    for map_B, exp_map_B in zip(maps_B, exp_maps_B):
        assert np.array_equal(map_B.todense(), exp_map_B.todense())


def test_linked_bsf_set_large():
    # Test with multiple ints representing bsf (N>64)
    large_X_connections = np.ones((66), dtype=bool)
    large_terms_X = Terms(term_type='X',
                          connections=large_X_connections)
    large_Z_connections = np.zeros((66, 66), dtype=bool)
    large_Z_connections[0, 1] = True
    large_Z_connections[-2, -1] = True
    large_terms_ZZ = Terms(term_type='ZZ',
                           connections=large_Z_connections)

    large_bsf_set_X = BSFSetBase(input_terms=[large_terms_X],
                                 magnitudes=[1.0])
    large_bsf_set_Z = BSFSetBase(input_terms=[large_terms_ZZ],
                                 magnitudes=[1.0])

    # Hamiltonian inputs
    large_list_of_bsf = [large_bsf_set_X, large_bsf_set_Z]
    large_variable_names = ["X", "ZZ"]
    large_nickname = "Large partially connected chain"

    large_ham = Hamiltonian(list_of_bsf=large_list_of_bsf,
                            variable_names=large_variable_names,
                            nickname=large_nickname)

    # Inputs A
    large_terms = Terms(term_type='ZZ',
                        connections=large_Z_connections)
    large_magnitudes = [1.0]

    # Expected output
    large_exp_bsf_array = [np.array([[[0, 0], [3, 0]],
                                     [[0, 0], [0, 3]]],
                                    dtype=np.int64)]
    large_exp_commuted_terms = [np.array([[[0, 1], [0, 3]],
                                          [[0, 2], [0, 3]],
                                          [[1, 0], [3, 0]],
                                          [[2, 0], [3, 0]]],
                                         dtype=np.int64)]
    large_exp_maps = [sp.csr_array([[0.0, -1.0],
                                    [0.0, -1.0],
                                    [-1.0, 0.0],
                                    [-1.0, 0.0]]),
                      sp.csr_array([[0.0, 0.0],
                                    [0.0, 0.0],
                                    [0.0, 0.0],
                                    [0.0, 0.0]])]

    # Setup large
    large_linkedbsfset = LinkedBSFSet(input_terms=[large_terms],
                                      magnitudes=large_magnitudes,
                                      hamiltonian=large_ham)

    # Retrieve attributes
    for large_bsf, large_exp in zip(large_linkedbsfset.bsf_array, large_exp_bsf_array):
        assert np.array_equal(large_bsf, large_exp)
    assert np.array_equal(large_linkedbsfset.magnitudes, large_magnitudes)
    assert large_linkedbsfset.nickname == "LinkedBSFSet object"
    assert large_linkedbsfset._int_num == 2
    assert large_linkedbsfset._int_type == np.uint64

    # Test commute
    large_commuted_terms, large_maps = large_linkedbsfset.commute()
    assert np.array_equal(large_commuted_terms, large_exp_commuted_terms[0])
    for large_map, large_exp_map in zip(large_maps, large_exp_maps):
        assert np.array_equal(large_map.todense(), large_exp_map.todense())

def test_many_body_commute():
    # Test a commutation with many bodies
    connections_XXXX = np.zeros((4,4,4,4), dtype=bool)
    connections_XXXX[0,1,2,3] = True
    terms_XXXX = Terms(term_type='XXXX',
                    connections=connections_XXXX)
    
    
    bsf_set_XXXX = BSFSetBase(input_terms=[terms_XXXX],
                                 magnitudes=[1.0])
    

    # Hamiltonian inputs
    many_list_of_bsf = [bsf_set_XXXX]
    many_variable_names = ["XXXX"]
    many_nickname = "Many body"

    many_ham = Hamiltonian(list_of_bsf=many_list_of_bsf,
                            variable_names=many_variable_names,
                            nickname=many_nickname)

    # Inputs A
    connections_YYY = np.zeros((4,4,4), dtype=bool)
    connections_YYY[0,1,2] = True
    terms_YYY = Terms(term_type='YYY',
                    connections=connections_YYY)
    many_magnitudes = [1.0]

    # Expected output
    many_exp_commuted_terms = [np.array([[[8], [7]]],
                                         dtype=np.int8)]
    many_exp_maps = [sp.csr_array([[-1.0]])]

    # Setup 
    many_linkedbsfset = LinkedBSFSet(input_terms=[terms_YYY],
                                      magnitudes=many_magnitudes,
                                      hamiltonian=many_ham)

    # Test commute
    many_commuted_terms, many_maps = many_linkedbsfset.commute()
    assert np.array_equal(many_commuted_terms, many_exp_commuted_terms[0])
    for many_map, many_exp_map in zip(many_maps, many_exp_maps):
        assert np.array_equal(many_map.todense(), many_exp_map.todense())

def test_large_many_body_commute():
    # Test a commutation with many bodies
    connections_XYZXYZ = np.zeros((6,6,6,6,6,6), dtype=bool)
    connections_XYZXYZ[0,1,2,3,4,5] = True
    terms_XYZXYZ = Terms(term_type='XYZXYZ',
                    connections=connections_XYZXYZ)
    
    
    bsf_set_XYZXYZ = BSFSetBase(input_terms=[terms_XYZXYZ],
                                 magnitudes=[1.0])
    

    # Hamiltonian inputs
    large_many_list_of_bsf = [bsf_set_XYZXYZ]
    large_many_variable_names = ["XYZXYZ"]
    large_many_nickname = "Large many body"

    large_many_ham = Hamiltonian(list_of_bsf=large_many_list_of_bsf,
                            variable_names=large_many_variable_names,
                            nickname=large_many_nickname)

    # Inputs A
    connections_ZYXYZX = np.zeros((6,6,6,6,6,6), dtype=bool)
    connections_ZYXYZX[0,1,2,3,4,5] = True
    terms_ZYXYZX = Terms(term_type='ZYXYZX',
                    connections=connections_ZYXYZX)
    large_many_magnitudes = [1.0]

    # Expected output
    large_many_exp_commuted_terms = [np.array([[[53], [45]]],
                                         dtype=np.int8)]
    large_many_exp_maps = [sp.csr_array([[-1.0]])]

    # Setup 
    large_many_linkedbsfset = LinkedBSFSet(input_terms=[terms_ZYXYZX],
                                      magnitudes=large_many_magnitudes,
                                      hamiltonian=large_many_ham)

    # Test commute
    large_many_commuted_terms, large_many_maps = large_many_linkedbsfset.commute()
    assert np.array_equal(large_many_commuted_terms, large_many_exp_commuted_terms[0])
    for large_many_map, large_many_exp_map in zip(large_many_maps, large_many_exp_maps):
        assert np.array_equal(large_many_map.todense(), large_many_exp_map.todense())


def test_even_set_bad_input():
    # Currently no attributes different to parent class
    pass


def test_even_set():
    terms_X = Terms(term_type='X',
                    connections=np.array([1, 1]).astype(bool))
    terms_ZZ = Terms(term_type='ZZ',
                     connections=np.array([[0, 1],
                                          [0, 0]]).astype(bool))

    bsf_set_X = BSFSetBase(input_terms=[terms_X],
                           magnitudes=[1.0])
    bsf_set_Z = BSFSetBase(input_terms=[terms_ZZ],
                           magnitudes=[1.0])

    # inputs A
    inputs_A = [terms_X]
    magnitudes_A = [1.0]
    ham = Hamiltonian(list_of_bsf=[bsf_set_X, bsf_set_Z],
                      variable_names=["X", "ZZ"])

    # Expected odd (right) set attributes
    exp_left_map = [sp.csc_array([[0.0, 0.0],
                                  [0.0, 0.0]]),
                    sp.csc_array([[-1.0, 0],
                                  [0, -1.0]])]
    exp_bsf_array = [np.array([[[1], [3]],
                               [[2], [3]]],
                              dtype=np.int8)]

    # Create even set and generate the odd
    even_set_A = EvenSet(input_terms=inputs_A,
                         magnitudes=magnitudes_A,
                         hamiltonian=ham)
    even_set_A.generate_odd()

    # Check attributes
    for exp_map, map_A in zip(exp_left_map, even_set_A.right_set.left_map):
        assert np.array_equal(exp_map.todense(), map_A.todense())
    assert np.array_equal(exp_bsf_array, even_set_A.right_set.bsf_array)


def test_odd_set_bad_input():
    terms_X = Terms(term_type='X',
                    connections=np.array([1, 1]).astype(bool))
    terms_ZZ = Terms(term_type='ZZ',
                     connections=np.array([[0, 1],
                                          [0, 0]]).astype(bool))

    bsf_set_X = BSFSetBase(input_terms=[terms_X],
                           magnitudes=[1.0])
    bsf_set_Z = BSFSetBase(input_terms=[terms_ZZ],
                           magnitudes=[1.0])

    # Good inputs
    good_input_array = [Terms(term_type='YZ',
                              connections=np.array([[0, 1],
                                                    [1, 0]]).astype(bool))]
    good_magnitudes = [1.0]
    good_hamiltonian = Hamiltonian(list_of_bsf=[bsf_set_X, bsf_set_Z],
                                   variable_names=["X", "ZZ"])
    good_left_set = EvenSet(input_terms=[terms_X],
                            magnitudes=[1.0],
                            hamiltonian=good_hamiltonian)
    good_nickname = "test"
    good_left_map = [sp.csc_array([[0.0, 0.0],
                                   [0.0, 0.0]]),
                     sp.csc_array([[-1.0, 0],
                                   [0, -1.0]])]
    good_right_map = [sp.csr_array([[1.0, 1.0],
                                   [-1.0, -1.0]]),
                      sp.csr_array([[0.0, 0],
                                   [0, 0.0]])]

    # Initialise with bad inputs
    # left_map
    with pytest.raises(TypeError):  # not list
        OddSet(input_terms=good_input_array,
               magnitudes=good_magnitudes,
               hamiltonian=good_hamiltonian,
               left_set=good_left_set,
               left_map="bla",
               nickname=good_nickname)

    with pytest.raises(ValueError):  # wrong length
        OddSet(input_terms=good_input_array,
               magnitudes=good_magnitudes,
               hamiltonian=good_hamiltonian,
               left_set=good_left_set,
               left_map=[sp.csc_array([[0.0, 0.0],
                                       [0.0, 0.0]])],
               nickname=good_nickname)

    with pytest.raises(TypeError):  # non csr_array, csc_array type
        OddSet(input_terms=good_input_array,
               magnitudes=good_magnitudes,
               hamiltonian=good_hamiltonian,
               left_set=good_left_set,
               left_map=["bla", "bla"],
               nickname=good_nickname)

    with pytest.raises(TypeError):  # non floating point
        OddSet(input_terms=good_input_array,
               magnitudes=good_magnitudes,
               hamiltonian=good_hamiltonian,
               left_set=good_left_set,
               left_map=[sp.csc_array([[False, False],
                                       [False, False]], dtype=bool),
                         sp.csc_array([[False, False],
                                       [False, False]], dtype=bool)],
               nickname=good_nickname)

    with pytest.raises(ValueError):  # wrong number of rows
        OddSet(input_terms=good_input_array,
               magnitudes=good_magnitudes,
               hamiltonian=good_hamiltonian,
               left_set=good_left_set,
               left_map=[sp.csc_array([[0.0, 0.0],
                                       [0.0, 0.0],
                                       [0.0, 0.0]]),
                         sp.csc_array([[-1.0, 0.0],
                                       [0.0, -1.0]])],
               nickname=good_nickname)

    with pytest.raises(ValueError):  # wrong number of columns
        OddSet(input_terms=good_input_array,
               magnitudes=good_magnitudes,
               hamiltonian=good_hamiltonian,
               left_set=good_left_set,
               left_map=[sp.csc_array([[0.0, 0.0],
                                       [0.0, 0.0]]),
                         sp.csc_array([[-1.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0]])],
               nickname=good_nickname)

    # Create good OddSet
    good_odd_set = OddSet(input_terms=good_input_array,
                          magnitudes=good_magnitudes,
                          hamiltonian=good_hamiltonian,
                          left_set=good_left_set,
                          left_map=good_left_map,
                          nickname=good_nickname)

    # Set good right_set
    good_right_set = EvenSet(input_terms=[Terms(term_type='ZZ',
                                                connections=np.array([[0, 1],
                                                                      [0, 0]]).astype(bool)),
                                          Terms(term_type='YY',
                                                connections=np.array([[0, 1],
                                                                      [0, 0]]).astype(bool))],
                             magnitudes=[1.0, 1.0],
                             hamiltonian=good_hamiltonian,
                             left_set=good_odd_set,
                             nickname=good_nickname)
    good_odd_set.right_set = good_right_set

    # right_map
    with pytest.raises(TypeError):  # not list
        good_odd_set.right_map = "bla"

    with pytest.raises(ValueError):  # wrong length
        good_odd_set.right_map = [sp.csc_array([[0.0, 0.0],
                                                [0.0, 0.0]])]

    with pytest.raises(TypeError):  # non csr_array, csc_array type
        good_odd_set.right_map = ["bla", "bla"]

    with pytest.raises(TypeError):  # non floating point
        good_odd_set.right_map = [sp.csc_array([[False, False],
                                                [False, False]], dtype=bool),
                                  sp.csc_array([[False, False],
                                                [False, False]], dtype=bool)]

    with pytest.raises(ValueError):  # wrong number of rows
        good_odd_set.right_map = [sp.csc_array([[0.0, 0.0],
                                                [0.0, 0.0],
                                                [0.0, 0.0]]),
                                  sp.csc_array([[-1.0, 0.0],
                                                [0.0, -1.0]])]

    with pytest.raises(ValueError):  # wrong number of columns
        good_odd_set.right_map = [sp.csc_array([[0.0, 0.0],
                                               [0.0, 0.0]]),
                                  sp.csc_array([[-1.0, 0.0, 0.0],
                                               [0.0, -1.0, 0.0]])]

    # set good right map
    good_odd_set.right_map = good_right_map

    # Check immutable
    with pytest.raises(ValueError):  # left_map change
        good_odd_set.left_map = good_left_map

    with pytest.raises(ValueError):  # right_map change
        good_odd_set.right_map = good_right_map


def test_odd_set():
    terms_X = Terms(term_type='X',
                    connections=np.array([1, 1]).astype(bool))
    terms_ZZ = Terms(term_type='ZZ',
                     connections=np.array([[0, 1],
                                          [0, 0]]).astype(bool))

    bsf_set_X = BSFSetBase(input_terms=[terms_X],
                           magnitudes=[1.0])
    bsf_set_Z = BSFSetBase(input_terms=[terms_ZZ],
                           magnitudes=[1.0])

    # inputs A
    inputs_A = [np.array([[[1], [3]],
                          [[2], [3]]],
                         dtype=np.int8)]
    magnitudes_A = [1.0]
    ham = Hamiltonian(list_of_bsf=[bsf_set_X, bsf_set_Z],
                      variable_names=["X", "ZZ"])

    # Expected attributes
    exp_right_map = [sp.csr_array([[1.0, 1.0],
                                   [-1.0, -1.0]]),
                     sp.csr_array([[0.0, 0.0],
                                   [0.0, 0.0]])]
    exp_right_bsf_array = [np.array([[[0], [3]],
                                     [[3], [3]]],
                                    dtype=np.int8)]

    # Create left set
    left_set_A = EvenSet(input_terms=[terms_X],
                         magnitudes=[1.0],
                         hamiltonian=ham)
    # create odd set and generate even
    odd_set_A = OddSet(input_terms=inputs_A,
                       magnitudes=magnitudes_A,
                       hamiltonian=ham,
                       left_set=left_set_A,
                       left_map=[sp.csc_array([[0.0, 0.0],
                                               [0.0, 0.0]]),
                                 sp.csc_array([[-1.0, 0],
                                               [0, -1.0]])])
    odd_set_A.generate_even()

    # Check attributes
    for exp_map, map_A in zip(exp_right_map, odd_set_A.right_map):
        assert np.array_equal(exp_map.todense(), map_A.todense())
    assert np.array_equal(exp_right_bsf_array, odd_set_A.right_set.bsf_array)
