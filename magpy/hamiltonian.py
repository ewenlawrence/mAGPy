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

"""Module for handling Hamiltonians for the use of commutation
"""
from typing import List, Optional

from magpy.bsf_set import BSFSetBase
from magpy.decorators import add_nickname

@add_nickname("Hamiltonian object")
class Hamiltonian:
    """Class of the hamiltonian
    """
    def __init__(self,
                 list_of_bsf: List[BSFSetBase],
                 variable_names: List[str],
                 nickname: Optional[str] = None):
        """Create Hamiltonian object

        Parameters
        ----------
        list_of_bsf : List[BSFSetBase]
            list of BSFSetBase for each of the variables
        variable_names : List[str]
            associated names of each of the variables in the list_of_bsf
        nickname : Optional[str], optional
            nickname for the class, by default "Hamiltonian object"
        """

        # Initial values (for type checking ordering)
        self._num_variables = None

        # Set values (using setter method)
        self.list_of_bsf = list_of_bsf
        self.variable_names = variable_names
        if not nickname is None:
            self.nickname = nickname

    def __repr__(self) -> str:
        return f"Hamiltonian({self.list_of_bsf}, {self.variable_names}, "\
            f"{self.nickname})"

    def __str__(self) -> str:
        return f"{self.nickname} with {self._num_variables} different "\
            "variables"

    @property
    def list_of_bsf(self):
        """List of bsf sets that make the Hamiltonian"""
        return self._list_of_bsf

    @list_of_bsf.setter
    def list_of_bsf(self, value):
        # Check input of list_of_bsf
        if not isinstance(value, list):
            raise TypeError("Argument 'list_of_bsf' must be a list")
        if not all(isinstance(bsf, BSFSetBase) for bsf in value):
            raise TypeError("Argument 'list_of_bsf' must contain only "
                            "BSFSetBase objects")
        if self._num_variables is None:
            self._num_variables = len(value)
        else:
            if len(value) != self._num_variables:
                raise ValueError("Argument 'list_of_bsf' must be same length "
                                 "as the number of variables")
        self._list_of_bsf = value

    @property
    def variable_names(self):
        """List of variables the Hamiltonian depends on"""
        return self._variable_names

    @variable_names.setter
    def variable_names(self, value):
        # Check input of variable_names
        if not isinstance(value, list):
            raise TypeError("Argument 'variable_names' must be a list")
        if not all(isinstance(bsf, str) for bsf in value):
            raise TypeError("Argument 'variable_names' must contain only str")
        if len(value) != self._num_variables:
            raise ValueError("Argument 'variable_names' must be same "
                             "length as the number of bsf sets")
        self._variable_names = value
