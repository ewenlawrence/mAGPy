from typing import List

from numpy import zeros, tile

from terms import Terms

class BSFSetBase:
    
    def __init__(self, 
                 term_array: List[float,Terms]):
        # ADD IN DEFAULT VALUES AND TYPE CHECKING
        
        # Behaviour should be either term_array is provided, or explicit
        # magnitudes and bsf_array
        
        # Figure out how many coefficient types
        self._num_coefficients = len(term_array)
        
        # process the term_array
        tmp_magnitudes = []
        tmp_bsf_array = []
        for index, (factor, terms_obj) in enumerate(term_array):
            tmp_bsf = terms_obj.generate_bsf()
            tmp_bsf_array.extend(tmp_bsf)
            tmp_coefficient = zeros(self._num_coefficients, dtype=float)
            tmp_coefficient[index] = factor
            tmp_magnitudes.extend(tile(tmp_coefficient,(len(tmp_bsf))))
            
            
        self._bsf_array = None
    
    def __repr__(self) -> str:
        pass
    
    def __str__(self) -> str:
        pass
    
    # Convert pauli to bsf    
    def _from_pauli(self):
        pass
    
    # Convert bsf to pauli
    def _to_pauli(self):
        pass

class HamiltonianSet(BSFSetBase):
    pass

class AGPSet(BSFSetBase):
    pass