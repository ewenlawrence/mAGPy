#    Copyright 2024 Ewen Lawrence

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# pylint: disable=no-name-in-module
# gmpy2 has error with name for popcount
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
# Other pylint errors will be fixed when the commute is rewritten
# pylint: disable=duplicate-code
# disable duplicate code for now, as there are shared attributes here
# in a code cleanup this can be refactored


"""Module for creating linked sets that make up the meta graph
"""
from typing import List, Union, Optional

from numpy import ndarray, floating
from numpy import array, issubdtype, unique, zeros, concatenate, sort
from numpy import append as numpy_append
from numpy.typing import ArrayLike
from scipy.sparse import csr_array, csc_array
from gmpy2 import popcount

from magpy.terms import Terms
from magpy.hamiltonian import Hamiltonian
from magpy.bsf_set import BSFSetBase
from magpy.decorators import immutable

class LinkedBSFSet(BSFSetBase):
    """Class for linked sets of bsf operators (parent class:BSFSetBase)
    """

    def __init__(self,
                 input_terms: List[Union[Terms, ndarray]],
                 magnitudes: ArrayLike,
                 hamiltonian: Hamiltonian,
                 left_set: Optional["LinkedBSFSet"] = None,
                 nickname: Optional[str] = None):
        """Create LinkedBSFSet object

        Parameters
        ----------
        input_terms : List[Union[Terms, ndarray]]
            input for the operators, should contain either Terms object that
            generate the bsf arrays, or the explicit bsf array in ndarray form
        magnitudes : ArrayLike
            magnitudes associated with each input term
        hamiltonian : Hamiltonian
            hamiltonian of the system
        left_set : LinkedBSFSet, optional
            linked set to the left, by default None
        nickname : str, optional
            nickname of class, by default "LinkedBSFSet object"
        """
        # Default values
        if nickname is None:
            nickname = "LinkedBSFSet object"

        # Call parent class __init__
        super().__init__(input_terms=input_terms,
                         magnitudes=magnitudes,
                         nickname=nickname)

        # Set values (using setter method)
        self.hamiltonian = hamiltonian
        if left_set is None:
            self._left_set = left_set
        else:
            self.left_set = left_set

        self._right_set = None

        # Figure out how many unique terms in object
        count = 0
        for bsf in self.bsf_array:
            count += len(bsf)
        self._num_terms = count

    def __repr__(self) -> str:
        return f"LinkedBSFSet({self.bsf_array}, {self.magnitudes}, "\
            f"{self.nickname}, {self.hamiltonian}, {self.left_set}, "\
            f"{self.right_set})"

    def __str__(self) -> str:
        return f"{self.nickname} with {self._num_terms} different "\
            f"terms and hamiltonian called {self.hamiltonian.nickname}"

    @property
    def hamiltonian(self):
        """Hamiltonian associated with linked set"""
        return self._hamiltonian

    @hamiltonian.setter
    @immutable("hamiltonian")
    def hamiltonian(self, value):
        # Check input of hamiltonian
        if not isinstance(value, Hamiltonian):
            raise TypeError("Argument 'hamiltonian' must be of type "
                            "Hamiltonian")
        self._hamiltonian = value

    @property
    def left_set(self):
        """Linked set to the left"""
        return self._left_set

    @left_set.setter
    @immutable("left_set")
    def left_set(self, value):
        # Check input of left_set
        if not isinstance(value, LinkedBSFSet):
            raise TypeError("Argument 'left_set' must be of type or child "
                            "of LinkedBSFSet")
        if value.hamiltonian != self.hamiltonian:
            raise ValueError("Argument 'left_set' must have same "
                             "hamiltonian")
        self._left_set = value

    @property
    def right_set(self):
        """Linked set to the right"""
        try:
            return self._right_set
        except AttributeError:
            return None

    @right_set.setter
    @immutable("right_set")
    def right_set(self, value):
        # Check input of right_set
        if not isinstance(value, LinkedBSFSet):
            raise TypeError("Argument 'right_set' must be of type or child "
                            "of LinkedBSFSet")
        if value.hamiltonian != self.hamiltonian:
            raise ValueError("Argument 'right_set' must have same "
                             "hamiltonian")
        self._right_set = value

    @property
    def num_terms(self):
        """Number of terms in set"""
        return self._num_terms

    def commute(self) -> tuple[ndarray]:
        """Commute the set with the Hamiltonian and generate set to the right"""
        # This algorithm could be very slow due to nested for loops etc.
        # This is likely the main area to improve performance

        # Setup lists to return
        commuted_terms = None
        maps = []

        # Intermediate list required
        wrong_shape_maps = []

        # Loop over each hamiltonian variable
        for ham_var_bsf_set in self.hamiltonian.list_of_bsf:

            if self.left_set is None:
                left_check = []
            else:
                left_check = self.left_set.bsf_array[0].tolist()

            # Setup list for var
            var_commuted_terms = []
            var_commuted_magnitudes = []
            var_from = []

            # Loop over each hamiltonian term magnitude
            for ham_term_array, ham_mag in zip(ham_var_bsf_set.bsf_array,
                                               ham_var_bsf_set.magnitudes):

                # loop over each individual bsf term in ham
                for ham_term in ham_term_array:

                    # Loop over each self term
                    # There is a bit of a mess due to their being
                    # magnitudes as well
                    for tmp_bsf_array in self.bsf_array:
                        for from_index, term in enumerate(tmp_bsf_array):
                            # Find out how many flips occur
                            f_locs = []
                            f_count = 0
                            for bit_num in range(self._int_num):
                                tmp = (ham_term[0][bit_num] & term[1][bit_num])\
                                    ^ (ham_term[1][bit_num] & term[0][bit_num])
                                f_locs.append(tmp)
                                f_count += popcount(int(tmp))

                            # If even number of flips, commutator is zero
                            if f_count % 2 == 0:
                                continue

                            # Commute the term and figure out sign
                            new_term_x = []
                            new_term_z = []
                            p_count = 0
                            for bit_num in range(self._int_num):
                                tmp_x = ham_term[0][bit_num] ^ term[0][bit_num]
                                tmp_z = ham_term[1][bit_num] ^ term[1][bit_num]

                                # append the terms
                                new_term_x.append(tmp_x)
                                new_term_z.append(tmp_z)

                                # how many plus signs
                                # (may be able to simplify this term)
                                p_loc = ((ham_term[0][bit_num] &
                                          term[1][bit_num]) ^
                                         (f_locs[bit_num] & (tmp_x & tmp_z))) \
                                    & f_locs[bit_num]
                                p_count += popcount(int(p_loc))

                            # figure out the phase/sign
                            phase = (2 * p_count - f_count) % 4

                            # build new term
                            new_term = [new_term_x, new_term_z]

                            # Check if already generated in left set
                            if new_term in left_check:
                                continue

                            # Add to terms and magnitudes
                            var_commuted_terms.append(new_term)
                            if phase > 1:
                                var_commuted_magnitudes.append(-2 * ham_mag)
                            else:
                                var_commuted_magnitudes.append(2 * ham_mag)

                            # Track where it came from
                            var_from.append(from_index)

            # Ensure new terms are created
            if len(var_commuted_terms) == 0:
                grouped_map_values = []
                grouped_col_indexes = []
                grouped_row_indexes = [0, 0]
            else:
                # Add onto the overall list (duplicates allowed atm)
                var_commuted_terms = array(var_commuted_terms,
                                           dtype=self._int_type)
                if commuted_terms is None:
                    tmp_duplicate_commuted_terms = var_commuted_terms
                    starting_index = 0
                else:
                    tmp_duplicate_commuted_terms = concatenate((
                        commuted_terms, var_commuted_terms))
                    starting_index = len(commuted_terms)

                # BUG does not preserve order of terms,
                # so previous maps get broken
                # Now group any terms that are the same
                commuted_terms, sorting_idx, indexes = unique(
                    tmp_duplicate_commuted_terms,
                    axis=0,
                    return_index=True,
                    return_inverse=True
                )
                # re adjust the ordering
                commuted_terms = commuted_terms[sort(sorting_idx)]
                new_indexes = array([sorting_idx[ind] for ind in indexes])
                tmp_num_terms = new_indexes.max() + 1

                # setup arrays
                tmp_grouped_map_values = [[] for i in range(tmp_num_terms)]
                tmp_grouped_col_indexes = [[] for i in range(tmp_num_terms)]

                # Only apply to the new terms
                for new_index, magnitude, from_index in \
                    zip(new_indexes[starting_index:],
                        var_commuted_magnitudes,
                        var_from):
                    tmp_grouped_map_values[new_index].append(magnitude)
                    tmp_grouped_col_indexes[new_index].append(from_index)

                # Currently this doesn't work, as it can cause looping when
                # right cancels but left doesn't
                # can be implemented at a later date
                # # Check if total of values is zero anywhere
                # zero_indices = []
                # for cur_index, value_array in \
                    # enumerate(tmp_grouped_map_values):
                #     if numpy_sum(value_array) == 0:
                #         tmp_grouped_map_values[cur_index] = []
                #         tmp_grouped_col_indexes[cur_index] = []

                #         # Track any new terms that are zero
                #         if cur_index > starting_index:
                #             zero_indices.append(cur_index)

                # # Pop off the values that are zero
                # for zero_index in zero_indices[::-1]:
                #     commuted_terms.pop(zero_index)
                #     # This is also needed as used to get rows
                #     tmp_grouped_map_values.pop(zero_index)

                # track the row count
                grouped_row_indexes = zeros(
                    len(tmp_grouped_map_values)+1, dtype=int)
                count = 0
                for i, row_vals in enumerate(tmp_grouped_map_values):
                    count += len(row_vals)
                    grouped_row_indexes[i+1] = count

                # Flatten the maps
                grouped_map_values = []
                for axis in tmp_grouped_map_values:
                    if len(axis) > 0:
                        grouped_map_values.extend(axis)
                grouped_col_indexes = []
                for axis in tmp_grouped_col_indexes:
                    if len(axis) > 0:
                        grouped_col_indexes.extend(axis)

            # Collect the csr maps data (but still can be leading zeros missing)
            wrong_shape_maps.append((grouped_map_values,
                                     grouped_col_indexes,
                                     grouped_row_indexes))

        # Add extra zero rows to each of the csr representations of maps
        if commuted_terms is None:
            total_number_terms = 0
        else:
            total_number_terms = len(commuted_terms)

        # If no new terms, return None
        if total_number_terms == 0:
            return None, None

        for values, indices, rows in wrong_shape_maps:

            final_count = rows[-1]
            current_rows = len(rows)-1

            if current_rows < total_number_terms:
                extra_zeros = [final_count for i in range(
                    total_number_terms-current_rows)]

                new_rows = numpy_append(rows, extra_zeros, axis=0)
            else:
                new_rows = rows

            maps.append(csr_array((values, indices, new_rows),
                                  shape=[total_number_terms,
                                         self._num_terms],
                                  dtype=float))

        # Return the commuted terms and the maps for each variable
        return array(commuted_terms, dtype=self._int_type), maps


class EvenSet(LinkedBSFSet):
    """Class for even set (parent class:LinkedBSFSet)
    """
    def __init__(self,
                 input_terms: List[Union[Terms, ndarray]],
                 magnitudes: ArrayLike,
                 hamiltonian: Hamiltonian,
                 left_set: Optional[LinkedBSFSet] = None,
                 nickname: Optional[str] = None):
        """Create EvenSet object

        Parameters
        ----------
        input_terms : List[Union[Terms, ndarray]]
            input for the operators, should contain either Terms object that
            generate the bsf arrays, or the explicit bsf array in ndarray form
        magnitudes : ArrayLike
            magnitudes associated with each input term
        hamiltonian : Hamiltonian
            hamiltonian of the system
        left_set : LinkedBSFSet, optional
            linked set to the left, by default None
        nickname : str, optional
            nickname of class, by default "EvenSet object"
        """
        # Default values
        if nickname is None:
            nickname = "EvenSet object"

        # Call parent class __init__
        super().__init__(input_terms=input_terms,
                         magnitudes=magnitudes,
                         hamiltonian=hamiltonian,
                         left_set=left_set,
                         nickname=nickname)

    def __repr__(self) -> str:
        return f"EvenSet({self.bsf_array}, {self.magnitudes}, "\
            f"{self.nickname}, {self.hamiltonian}, {self.left_set}, "\
            f"{self.right_set})"

    def generate_odd(self):
        """Generate the next odd set"""
        # Call parent function to generate the map and next set
        odd_input_terms, maps_to_odd = self.commute()
        if odd_input_terms is None:
            self._right_set = None
        else:
            # Adjust the map such that it goes odd -> even
            maps_from_odd = []
            for map_i in maps_to_odd:
                maps_from_odd.append(-1 * map_i.T)

            # Create OddSet object and assign to right_set
            self.right_set = OddSet(input_terms=[odd_input_terms],
                                    magnitudes=[1.0],
                                    hamiltonian=self.hamiltonian,
                                    left_map=maps_from_odd,
                                    left_set=self,)
        return self.right_set


class OddSet(LinkedBSFSet):
    """Class for OddSet (parent class: LinkedBSFSet)"""

    def __init__(self,
                 input_terms: List[Union[Terms, ndarray]],
                 magnitudes: ArrayLike,
                 hamiltonian: Hamiltonian,
                 left_set: LinkedBSFSet,
                 left_map: List[Union[csr_array, csc_array]],
                 nickname: Optional[str] = None):
        """Create OddSet

        Parameters
        ----------
        input_terms : List[Union[Terms, ndarray]]
            input for the operators, should contain either Terms object that
            generate the bsf arrays, or the explicit bsf array in ndarray form
        magnitudes : ArrayLike
            magnitudes associated with each input term
        hamiltonian : Hamiltonian
            hamiltonian of the system
        left_set : LinkedBSFSet, optional
            linked set to the left, by default None
        left_map : List[Union[csr_array, csc_array]]
            mapping to left set
        nickname : str, optional
            nickname of class, by default "OddSet object"
        """
        # Default values
        if nickname is None:
            nickname = "OddSet object"

        # Call parent class __init__
        super().__init__(input_terms=input_terms,
                         magnitudes=magnitudes,
                         hamiltonian=hamiltonian,
                         left_set=left_set,
                         nickname=nickname)

        # Set values (using setter method)
        self.left_map = left_map

    def __repr__(self) -> str:
        return f"OddSet({self.bsf_array}, {self.magnitudes}, "\
            f"{self.nickname}, {self.hamiltonian}, {self.left_set}, "\
            f"{self.right_set}, {self.left_map}, {self.right_map})"

    @property
    def left_map(self):
        """Map to the left set"""
        return self._left_map

    @left_map.setter
    @immutable("left_map")
    def left_map(self, value):
        # Check input of left_map
        if not isinstance(value, list):
            raise TypeError("Argument 'left_map' must be a list.")
        if len(value) != self.hamiltonian.num_variables:
            raise ValueError("Argument 'left_map' must have length equal "
                             "to number of variables in Hamiltonian")
        for tmp_var_left_map in value:
            if not isinstance(tmp_var_left_map, (csr_array, csc_array)):
                raise TypeError("Argument 'left map' must have elements "
                                "of type csr_array or csc_array")
            if not issubdtype(tmp_var_left_map.dtype, floating):
                raise TypeError("Argument 'left_map' data type must be "
                                "subclass of numpy.floating.")
            if tmp_var_left_map.shape[0] != self.left_set.num_terms:
                raise ValueError("Argument 'left_map' must have the number "
                                 "of rows equal to the number of "
                                 "terms in left set")
            if tmp_var_left_map.shape[1] != self._num_terms:
                raise ValueError("Argument 'left_map' must have the number "
                                 "of columns equal to the number of "
                                 "terms of itself")
        self._left_map = value

    @property
    def right_map(self):
        """Map to the right set"""
        try:
            return self._right_map
        except AttributeError:
            return None

    @right_map.setter
    @immutable("right_map")
    def right_map(self, value):
        # Check input of left_map
        if not isinstance(value, list):
            raise TypeError("Argument 'right_map' must be a list.")
        if len(value) != self.hamiltonian.num_variables:
            raise ValueError("Argument 'right_map' must have length equal "
                             "to number of variables in Hamiltonian")
        for tmp_var_right_map in value:
            if not isinstance(tmp_var_right_map, (csr_array, csc_array)):
                raise TypeError("Argument 'right map' must have elements "
                                "of type csr_array or csc_array")
            if not issubdtype(tmp_var_right_map.dtype, floating):
                raise TypeError("Argument 'right_map' data type must be "
                                "subclass of numpy.floating.")
            if tmp_var_right_map.shape[0] != self.right_set.num_terms:
                raise ValueError("Argument 'right_map' must have the number"
                                 " of rows equal to the number of "
                                 "magnitudes/terms in right set")
            if tmp_var_right_map.shape[1] != self._num_terms:
                raise ValueError("Argument 'right_map' must have the number"
                                 " of columns equal to the number of "
                                 "magnitudes/terms of itself")
        self._right_map = value

    def generate_even(self):
        """Generate the next even set"""
        # Call parent function to generate the map and next set
        even_input_terms, map_to_even = self.commute()
        if even_input_terms is None:
            self._right_set = None
        else:
            # Create EvenSet object and assign to right_set
            self.right_set = EvenSet(input_terms=[even_input_terms],
                                     magnitudes=[1.0],
                                     hamiltonian=self.hamiltonian,
                                     left_set=self,)
            # Assign map to right_map
            self.right_map = map_to_even

        return self.right_set
