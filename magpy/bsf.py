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

"""Module for user friendly BSF manipulation"""
from numpy.typing import ArrayLike


class BSF:
    """Class for handling individual BSF terms, 
    for easy readability for the user"""

    def __init__(self, x: ArrayLike, z: ArrayLike, plus: bool = True):
        """Create a single BSF term

        Parameters
        ----------
        x : ArrayLike
            array corresponding to the x part of bsf
        z : ArrayLike
            array corresponding to the z part of bsf
        plus : bool, optional
            sign of the bsf (True = +, False = -), by default True
        """

        self._plus = plus
        self._x = x
        self._z = z

    # Convert from pauli string
    def _from_pauli_to_bsf(self):
        #TODO
        pass

    # Convert to pauli string
    def _from_bsf_to_pauli(self):
        #TODO
        pass

    # Commute with another bsf
    # Return 0 if commutes, return bsf (with appropriate sign) if not
    def commute(self, other:"BSF"):
        """Commute with another bsf"""
        #TODO

    # Addition of bsf
    def add(self, other:"BSF"):
        """Add with another bsf"""
        #TODO
