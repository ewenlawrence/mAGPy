from typing import Optional

from numpy import ndarray, array
from numpy.typing import ArrayLike


class Terms():

    def __init__(self,
                 term_type: str,
                 connections: ArrayLike,
                 nickname: Optional[str]=None):
        """Create a Terms object"""
        
        # Default values
        if nickname is None:
            nickname = "Terms object"

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
        try:
            tmp_term_type = str(value).upper()
        except Exception as e:
            raise TypeError("Argument 'term_type' must be str.") from e
        if set(tmp_term_type) > set('X', 'Y', 'Z', 'I'):
            raise ValueError("Argument 'term_type' must only contain X,Y,Z,I.")
        self._term_type = tmp_term_type
        self._term_size = len(tmp_term_type)
        
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
        if not isinstance(tmp_connections.dtype, int):
            raise TypeError("Argument 'connections' must contain dtype int.")
        if tmp_connections.ndim != self._term_size:
            raise ValueError("Argument 'connections' must be same dimension as \
                'term_type' length.")
        if len(set(tmp_connections.shape())) > 1:
            raise ValueError("Argument 'connections' must be have all axis \
                same length.")
        self._connections = tmp_connections

    @property
    def nickname(self) -> str:
        """Nickname of terms object"""
        return self._nickname
        
    @nickname.setter
    def nickname(self, value):
        # Check input of nickname
        try:
            tmp_nickname = str(value)
        except Exception as e:
            raise TypeError("Argument 'nickname' must be str.") from e
        self._nickname = tmp_nickname
    
    def generate_bsf(self) -> ndarray:
        """Generate the BSF array for use in BSFSET"""
        pass