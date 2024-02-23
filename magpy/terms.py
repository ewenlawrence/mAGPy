from typing import Optional
import warnings

from numpy import ndarray, uint8, uint16, uint32, uint64
from numpy import array, nonzero, zeros, unique
from numpy.typing import ArrayLike


class Terms():

    def __init__(self,
                 term_type: str,
                 connections: ArrayLike,
                 nickname: Optional[str] = None):
        """Create a Terms object"""

        # Default values
        if nickname is None:
            nickname = "Terms object"

        # Initial values (for type checking ordering)
        self._term_size = None

        # Set values (using setter method)
        self.term_type = term_type
        self.connections = connections
        self.nickname = nickname

    def __repr__(self) -> str:
        return f"Terms({self._term_type, self._connections, self._nickname})"

    def __str__(self) -> str:
        return f"{self._nickname} using {self._term_type} operator"

    @property
    def term_type(self) -> str:
        """Pauli string representing operator"""
        return self._term_type

    @term_type.setter
    def term_type(self, value):
        # Check input of term_type
        if not isinstance(value, str):
            raise TypeError("Argument 'term_type' must be str.")
        tmp_term_type = str(value).upper()
        if not set(tmp_term_type) <= set(['X', 'Y', 'Z', 'I']):
            raise ValueError("Argument 'term_type' must only contain X,Y,Z,I.")
        self._term_type = tmp_term_type
        if self._term_size is None:
            self._term_size = len(tmp_term_type)
        else:
            tmp_term_size = len(tmp_term_type)
            if tmp_term_size != self._connections.ndim:
                raise ValueError("Argument 'term_type' length must be same as "\
                                 "dimension of 'connections'.")
            else:
                self._term_size = tmp_term_size

    @property
    def connections(self) -> ndarray:
        """Array of connections of terms"""
        return self._connections

    @connections.setter
    def connections(self, value):
        # Check input of connections
        try:
            tmp_connections = array(value)
        except Exception as e:
            raise TypeError("Argument 'connections' must be arraylike.") from e
        if tmp_connections.dtype != bool:
            try:
                tmp_connections.astype(bool)
                warnings.warn("Casting argument 'connections' to bool",
                              UserWarning)
            except Exception as e:
                raise TypeError("Argument 'connections' must contain dtype "\
                    "bool.") from e
        if tmp_connections.ndim != self._term_size:
            raise ValueError("Argument 'connections' must be same dimension "\
                "as 'term_type' length.")
        if len(set(tmp_connections.shape)) > 1:
            raise ValueError("Argument 'connections' must be have all axis "\
                "same length.")
        tmp_non_zero_locs = array(nonzero(tmp_connections))
        # Note: this check is likely not a good implementation, and could be
        # improved for runtime
        for index_array in tmp_non_zero_locs.T:
            if unique(index_array).shape[0] != tmp_connections.ndim:
                raise ValueError("Argument 'connections' must only have non "\
                    "zero entries where all indexes are different")
        self._non_zero_locs = tmp_non_zero_locs
        self._connections = tmp_connections

    @property
    def nickname(self) -> str:
        """Nickname of terms object"""
        return self._nickname

    @nickname.setter
    def nickname(self, value):
        # Check input of nickname
        if not isinstance(value, str):
            raise TypeError("Argument 'nickname' must be str.")
        self._nickname = value

    def generate_bsf(self) -> ndarray:
        """Generate the BSF array for use in BSFSET"""

        # Figure out best int size for memory optimisation
        # Speed comparison should be made between using everything int8
        dimension = self._connections.shape[0]
        int_type = None
        pos_int_sizes = [8, 16, 32]
        pos_int_types = [uint8, uint16, uint32]
        for pos_int_size, pos_int_type in zip(pos_int_sizes, pos_int_types):
            if dimension <= pos_int_size:
                int_size = pos_int_size
                int_type = pos_int_type
                break
        if int_type is None:
            int_size = 64
            int_type = uint64

        # How many ints are required (there will likely be some hanging zeros)
        num_ints = -(dimension // -int_size)

        # Create array of zero bsfs, for each non zeros
        bsf_array = zeros((len(self._non_zero_locs[0]), 2, num_ints), dtype=int_type)

        # Update each value with the operator, by adding each binary number
        for term, loc_array in zip(self._term_type, self._non_zero_locs):
            # Loop over each location
            for index, loc in enumerate(loc_array):
                # Find which bit represents the location
                bit_loc = loc // int_size
                bit_power = loc % int_size
                # Add the term
                if term == 'X':
                    bsf_array[index, 0, bit_loc] += 2 ** bit_power
                elif term == 'Z':
                    bsf_array[index, 1, bit_loc] += 2 ** bit_power
                elif term == 'Y':
                    bsf_array[index, 0, bit_loc] += 2 ** bit_power
                    bsf_array[index, 1, bit_loc] += 2 ** bit_power

        return bsf_array
    