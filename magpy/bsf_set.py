from typing import List, Union, Optional

from magpy.terms import Terms

from numpy import ndarray, integer, floating
from numpy import array, issubdtype
from numpy.typing import ArrayLike


class BSFSetBase:

    def __init__(self,
                 input_terms: List[Union[Terms, ndarray]],
                 magnitudes: ArrayLike,
                 nickname: Optional[str] = None):
        """Create a BSFSetBase object"""

        # Default values
        if nickname is None:
            nickname = "BSFSetBase object"

        # Initial values (for type checking ordering)
        self._num_magnitudes = None

        # Set values (using setter method)
        self.bsf_array = input_terms
        self.magnitudes = magnitudes
        self.nickname = nickname

    def __repr__(self) -> str:
        return f"BSFSetBase({self.bsf_array}, {self.magnitudes}, "\
            f"{self.nickname})"

    def __str__(self) -> str:
        return f"{self.nickname} with {self._num_magnitudes} different "\
            "magnitudes"

    @property
    def bsf_array(self):
        """BSF array representing the operators"""
        return self._bsf_array

    @bsf_array.setter
    def bsf_array(self, value):
        # Check input of bsf_array
        if not isinstance(value, List):
            raise TypeError("Argument 'input_terms' must be a list")
        if self._num_magnitudes is None:
            self._num_magnitudes = len(value)
        else:
            tmp_num_magnitudes = len(value)
            if tmp_num_magnitudes != self._magnitudes.shape[0]:
                raise ValueError("Argument 'input_terms' must be same length "
                                 "as number of 'magnitudes'")
            self._num_magnitudes = tmp_num_magnitudes
        tmp_bsf_array = []
        self._int_type = None
        self._int_num = None
        for index, term in enumerate(value):
            if isinstance(term, Terms):
                tmp_bsf_array.append(term.generate_bsf())
            elif isinstance(term, ndarray):
                if not issubdtype(term.dtype, integer):
                    raise ValueError("If argument 'input_terms' contains "
                                     "ndarray, the data type must be a subclass of "
                                     "numpy.integer")
                if term.ndim != 3:
                    raise ValueError("If argument 'input_terms' contains "
                                     "ndarray, they must have ndim = 3")
                if term.shape[1] != 2:
                    raise ValueError("If argument 'input_terms' contains "
                                     "ndarray, they must have shape = (num_bsfs, 2, "
                                     "int_num)")
                tmp_bsf_array.append(term)
            else:
                raise TypeError("Argument 'input_terms' must only contain "
                                "Terms or ndarray")
            if self._int_type is None:
                self._int_type = tmp_bsf_array[index].dtype
            if tmp_bsf_array[index].dtype != self._int_type:
                raise TypeError("Argument 'input_terms' must have or generate "
                                "bsf arrays with all the same integer type")
            if self._int_num is None:
                self._int_num = tmp_bsf_array[index].shape[2]
            if tmp_bsf_array[index].shape[2] != self._int_num:
                raise TypeError("Argument 'input_terms' must have or generate "
                                "bsf arrays with the same number of integers representing "
                                "a bsf term")
        self._bsf_array = tmp_bsf_array

    @property
    def magnitudes(self):
        """Magnitudes of each of the operators"""
        return self._magnitudes

    @magnitudes.setter
    def magnitudes(self, value):
        # Check input of magnitudes
        try:
            tmp_magnitudes = array(value)
        except Exception as e:
            raise TypeError("Argument 'magnitudes' must be arraylike.") from e
        if not issubdtype(tmp_magnitudes.dtype, floating):
            raise TypeError("Argument 'magnitudes' data type must be subclass "
                            "of numpy.floating.")
        if tmp_magnitudes.ndim > 1:
            raise ValueError("Argument 'magnitudes' must be a 1D arraylike")
        if tmp_magnitudes.shape[0] != self._num_magnitudes:
            raise ValueError("Argument 'magnitudes' must be same length as "
                             "number of 'input_terms'")
        self._magnitudes = tmp_magnitudes

    @property
    def nickname(self):
        """Nickname of BSFBaseSet class"""
        return self._nickname

    @nickname.setter
    def nickname(self, value):
        # Check input of nickname
        if not isinstance(value, str):
            raise TypeError("Argument 'nickname' must be str.")
        self._nickname = value


class LinkedBSFSet(BSFSetBase):
    pass


class EvenSet(LinkedBSFSet):
    pass


class OddSet(LinkedBSFSet):
    pass
