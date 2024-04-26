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

from typing import List

import warnings
from numpy import issubdtype, array
from numpy import integer, ndarray, floating

from magpy.hamiltonian import Hamiltonian
from magpy.linked_set import OddSet
from magpy.decorators import add_nickname, immutable


@add_nickname("AGP object")
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

        # Set attributes
        self.hamiltonian = hamiltonian
        self.lambda_index = lambda_index
        self.lambda_values = lambda_values
        self.constant_values = constant_values
        self.operators = operators
        self.coefficients = coefficients
        self.exact = exact
        if not nickname is None:
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
    @immutable("hamiltonian")
    def hamiltonian(self, value):
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
    @immutable("lambda_index")
    def lambda_index(self, value):
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
    @immutable("lambda_values")
    def lambda_values(self, value):
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
    @immutable("constant_values")
    def constant_values(self, value):
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
    @immutable("coefficients")
    def coefficients(self, value):
        # check the input of coefficients
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
    @immutable("exact")
    def exact(self, value):
        # Type check the value fo exact
        if not isinstance(value, bool):
            raise TypeError("Argument 'exact' should be of type bool")
        self._exact = value
