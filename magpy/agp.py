from magpy.hamiltonian import Hamiltonian
from magpy.bsf_set import OddSet

from typing import List

import warnings
from numpy import issubdtype, array
from numpy import integer, ndarray, floating


class AGP:

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 lambda_index: integer,
                 lambda_values: ndarray,
                 constant_values: ndarray,
                 operators: List[OddSet],
                 coefficients: ndarray,
                 exact: bool,
                 nickname: str = None):

        # Default values
        if nickname is None:
            nickname = "AGP object"

        # Set attributes
        self.hamiltonian = hamiltonian
        self.lambda_index = lambda_index
        self.lambda_values = lambda_values
        self.constant_values = constant_values
        self.operators = operators
        self.coefficients = coefficients
        self.exact = exact
        self.nickname = nickname

    def __repr__(self) -> str:
        return "TODO"

    def __str__(self) -> str:
        return "TODO"

    @property
    def hamiltonian(self):
        """Hamiltonian associated with meta graph"""
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, value):
        # Make immutable by only setting if attribute is empty
        try:
            self._hamiltonian
            raise ValueError("Argument 'hamiltonian' is immutable and "
                             "already has a value set")
        except ValueError as e:
            raise e
        except AttributeError:
            # Check input of hamiltonian
            if not isinstance(value, Hamiltonian):
                raise TypeError("Argument 'hamiltonian' must be of type "
                                "Hamiltonian")
            self._hamiltonian = value

    @property
    def lambda_index(self):
        """Index of lambda being varied in AGP"""
        return self._lambda_index

    @lambda_index.setter
    def lambda_index(self, value):
        # Make immutable by only setting if attribute is empty
        try:
            self._lambda_index
            raise ValueError("Argument 'lambda_index' is immutable and "
                             "already has a value set")
        except ValueError as e:
            raise e
        except AttributeError:
            # Check input of lambda_index
            if not issubdtype(type(value), integer):
                raise TypeError("Argument 'lambda_index' must be an integer")
            if value >= len(self._hamiltonian.variable_names):
                raise ValueError("Argument 'lambda_index' must be a valid index "
                                 "for the hamiltonian lambda names")
            self._lambda_index = value

    def _parse_lambda_values(self, value):
        try:
            tmp_lambda_values = array(value)
        except Exception as e:
            raise TypeError(
                "Argument 'lambda_values' must be arraylike.") from e
        if tmp_lambda_values.dtype != float:
            try:
                tmp_lambda_values.astype(float)
                warnings.warn("Casting argument 'lambda_values' to float",
                              UserWarning)
            except Exception as e:
                raise TypeError("Argument 'lambda_values' must contain dtype "
                                "float.") from e
        if tmp_lambda_values.ndim != 1:
            raise ValueError("Argument 'lambda_values' must be 1D array")

        return tmp_lambda_values

    @property
    def lambda_values(self):
        """Array of values the lambda takes"""
        return self._lambda_values

    @lambda_values.setter
    def lambda_values(self, value):
        # Make immutable by only setting if attribute is empty
        try:
            self._lambda_values
            raise ValueError("Argument 'lambda_values' is immutable and "
                             "already has a value set")
        except ValueError as e:
            raise e
        except AttributeError:
            self._lambda_values = self._parse_lambda_values(value=value)

    def _parse_constant_values(self, value):
        try:
            tmp_constant_values = array(value)
        except Exception as e:
            raise TypeError(
                "Argument 'constant_values' must be arraylike.") from e
        if tmp_constant_values.dtype != float:
            try:
                tmp_constant_values.astype(float)
                warnings.warn("Casting argument 'constant_values' to float",
                              UserWarning)
            except Exception as e:
                raise TypeError("Argument 'constant_values' must contain dtype "
                                "float.") from e
        if tmp_constant_values.ndim != 1:
            raise ValueError("Argument 'constant_values' must be 1D array")
        if len(tmp_constant_values) != len(self.hamiltonian.variable_names) - 1:
            raise ValueError("Argument 'constant_values' must be same length "
                             "as the number of Hamiltonian variables minus 1")

        return tmp_constant_values

    @property
    def constant_values(self):
        """Values of the constant variables"""
        return self._constant_values

    @constant_values.setter
    def constant_values(self, value):
        # Make immutable by only setting if attribute is empty
        try:
            self._constant_values
            raise ValueError("Argument 'constant_values' is immutable and "
                             "already has a value set")
        except ValueError as e:
            raise e
        except AttributeError:
            self._constant_values = self._parse_constant_values(value=value)

    @property
    def operators(self):
        """Operators of the AGP"""
        return self._operators

    @operators.setter
    def operators(self, value):
        # Make immutable by only setting if attribute is empty
        try:
            self._operators
            raise ValueError("Argument 'operators' is immutable and "
                             "already has a value set")
        except ValueError as e:
            raise e
        except AttributeError:
            if not isinstance(value, list):
                raise TypeError("Argument 'operators' should be a list")
            for op in value:
                if not isinstance(op, OddSet):
                    raise TypeError("Argument 'operators' should contain "
                                    "OddSet")
            self._num_operators = len(value)
            self._operators = value

    @property
    def coefficients(self):
        """Coefficients of the AGP operators"""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        # Make immutable by only setting if attribute is empty
        try:
            self._coefficients
            raise ValueError("Argument 'coefficients' is immutable and "
                             "already has a value set")
        except ValueError as e:
            raise e
        except AttributeError:
            if not isinstance(value, ndarray):
                raise TypeError(
                    "Argument 'coefficients' should be type ndarray")
            if not issubdtype(value.dtype, floating):
                raise TypeError("Argument 'coefficients' should have have dtype "
                                "that is subtype of floating")
            self._coefficients = value

    @property
    def exact(self):
        """Whether the AGP is exact"""
        return self._exact

    @exact.setter
    def exact(self, value):
        # Make immutable by only setting if attribute is empty
        try:
            self._exact
            raise ValueError("Argument 'exact' is immutable and "
                             "already has a value set")
        except ValueError as e:
            raise e
        except AttributeError:
            if not isinstance(value, bool):
                raise TypeError("Argument 'exact' should be of type bool")
            self._exact = value

    @property
    def nickname(self):
        """Nickname of MetaGraph class"""
        return self._nickname

    @nickname.setter
    def nickname(self, value):
        # Check input of nickname
        if not isinstance(value, str):
            raise TypeError("Argument 'nickname' must be str.")
        self._nickname = value
