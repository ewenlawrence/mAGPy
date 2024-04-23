from magpy.bsf_set import EvenSet
from magpy.hamiltonian import Hamiltonian
from magpy.agp import AGP

import warnings

from numpy import concatenate, issubdtype, array, shape, zeros, pad
from numpy.linalg import solve as numpy_solve
from numpy import integer
from scipy.linalg import block_diag
from scipy.sparse import csr_array


class MetaGraph:

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 lambda_index: integer,
                 nickname: str = None):

        # Default values
        if nickname is None:
            nickname = "MetaGraph object"
        self._exact = False

        # Setup initially empty attributes
        self._left_lam_maps = None
        self._right_lam_maps = None
        self._left_const_maps = None
        self._right_const_maps = None
        self._odd_sets = None
        self._even_sets = None

        self._current_max_set = 0  # Number of odd sets currently computed

        # Initialise for type checking
        self._lambda_index = None

        # Set attributes
        self.hamiltonian = hamiltonian
        self.lambda_index = lambda_index
        self.nickname = nickname

    def __repr__(self):
        return "TODO"

    def __str__(self):
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
        """Index of variable being varied in AGP"""
        return self._lambda_index

    @lambda_index.setter
    def lambda_index(self, value):
        # Check input of lambda_index
        if not issubdtype(type(value), integer):
            raise TypeError("Argument 'lambda_index' must be an integer")
        if value >= len(self._hamiltonian.variable_names):
            raise ValueError("Argument 'lambda_index' must be a valid index "
                             "for the hamiltonian variable names")
        # reset the class if it is different
        if value != self._lambda_index:
            self._lambda_index = value

            self._left_lam_maps = []
            self._right_lam_maps = []
            self._left_const_maps = []
            self._right_const_maps = []
            self._exact = False

            self._current_max_set = 0

            # Create the first even set
            tmp_bsf_set = self.hamiltonian.list_of_bsf[self._lambda_index]
            tmp_bsf_array = concatenate(tmp_bsf_set.bsf_array)
            even_set_0 = EvenSet(input_terms=[tmp_bsf_array],
                                 magnitudes=[1.0],
                                 hamiltonian=self._hamiltonian)

            # Setup sets
            self._odd_sets = []
            self._even_sets = [even_set_0]

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

    def _parse_max_odd(self, value):
        if not issubdtype(type(value), integer):
            raise TypeError("Argument 'max_odd' must be int")
        if value < 1:
            raise ValueError("Argument 'max_odd' must be greater then 0")
        return value

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

    def _compute_odd_even_step(self):
        # Compute the next odd and even sets, collecting the left and right maps
        new_odd = self._even_sets[-1].generate_odd()
        if new_odd is None:
            self._exact = True
            return

        # If still more left add the left map and continue
        self._odd_sets.append(new_odd)
        tmp_left_map = self._odd_sets[-1]._left_map
        self._left_lam_maps.append(tmp_left_map[self.lambda_index])
        tmp_left_map.pop(self.lambda_index)
        self._left_const_maps.append(tmp_left_map)

        # Right map
        self._even_sets.append(self._odd_sets[-1].generate_even())
        tmp_right_map = self._odd_sets[-1]._right_map
        self._right_lam_maps.append(tmp_right_map[self.lambda_index])
        tmp_right_map.pop(self.lambda_index)
        self._right_const_maps.append(tmp_right_map)

        self._current_max_set += 1

    def _generate_maps(self, max_odd):
        # Generate the maps upto a maximum number of odd commutations

        # Ensure max_odd is valid
        self._parse_max_odd(value=max_odd)

        # Repeat upto max_odd or exact AGP is found
        while self._current_max_set < max_odd and not self._exact:
            self._compute_odd_even_step()

    def _create_dense(self, diag, off_diag):
        # Convert from list of sparse block to dense form
        lengths = [shape(block)[0] for block in diag]
        diag = [mat.toarray() for mat in diag]
        dense_matrix = block_diag(*diag)
        start = 0
        for i in range(len(lengths)-1):
            off_array = off_diag[i].toarray()
            middle = start + lengths[i]
            end = middle + lengths[i+1]
            dense_matrix[start:middle, middle:end] = off_array
            dense_matrix[middle:end, start:middle] = off_array.T
            start = middle
        return dense_matrix

    def _solve_numpy(self, hessian, initial, args):
        # Dense solver using numpy.linalg.solve()
        print(hessian)
        print(initial)
        return numpy_solve(hessian, initial)

    def compute_AGP(self, lambda_values, constant_values, max_odd=None,
                    solver='numpy', solver_args=None):
        # Solve the Hessian for a list of values of the variable and return an
        # AGP object

        # Parse lambda_values and constant_values
        lambda_values = self._parse_lambda_values(value=lambda_values)
        constant_values = self._parse_constant_values(value=constant_values)

        # Ensure max_odd is valid
        if max_odd is None:
            max_odd = self._current_max_set
        else:
            self._parse_max_odd(value=max_odd)

        # Generate any extra terms
        if max_odd > self._current_max_set:
            self._generate_maps(max_odd=max_odd)

        # Setup the constant values
        left_constant_maps = []
        right_constant_maps = []
        for left_maps, right_maps in zip(self._left_const_maps[:max_odd],
                                         self._right_const_maps[:max_odd]):
            tmp_left = constant_values[0] * left_maps[0]
            tmp_right = constant_values[0] * right_maps[0]
            for const_val, left_mat, right_mat in zip(constant_values[1:],
                                                      left_maps[1:],
                                                      right_maps[1:]):
                tmp_left += const_val * left_mat
                tmp_right += const_val * right_mat
            left_constant_maps.append(tmp_left)
            right_constant_maps.append(tmp_right)

        # Setup initial condition (can depend on lambda)
        bsf_set_0 = self.hamiltonian.list_of_bsf[self.lambda_index]
        magnitudes_0 = []
        for bsf_array, mag in zip(bsf_set_0.bsf_array, bsf_set_0.magnitudes):
            tmp_array = [mag for i in range(len(bsf_array))]
            magnitudes_0.extend(tmp_array)
        initial_const = -1 * left_constant_maps[0].T @ array(magnitudes_0).T
        initial_lambda = -1*self._left_lam_maps[0].T @ array(magnitudes_0).T

        # Build polynomial in variable (axis = power of lambda)
        base_diag = [[], [], []]
        for index, (left_const, right_const, left_lam, right_lam) \
            in enumerate(zip(left_constant_maps[:max_odd],
                             right_constant_maps[:max_odd],
                             self._left_lam_maps[:max_odd],
                             self._right_lam_maps[:max_odd])):
            # left-left and right-right
            base_diag[0].append(left_const.T @ left_const +
                                right_const.T @ right_const)
            base_diag[1].append(left_const.T @ left_lam +
                                left_lam.T @ left_const + right_const.T @ right_lam +
                                right_lam.T @ right_const)
            base_diag[2].append(left_lam.T @ left_lam +
                                right_lam.T @ right_lam)

        base_off_diag = [[], [], []]
        for left_const, right_const, left_lam, right_lam \
            in zip(left_constant_maps[1:max_odd],
                   right_constant_maps[:max_odd-1],
                   self._left_lam_maps[1:max_odd],
                   self._right_lam_maps[:max_odd-1]):
            # left-right
            base_off_diag[0].append(left_const.T @ right_const)
            base_off_diag[1].append(left_const.T @ right_lam +
                                    left_lam.T @ right_const)
            base_off_diag[2].append(left_lam.T @ right_lam)

        # Solve the matrix equation using chosen method
        if solver == 'numpy':
            solver_func = self._solve_numpy
            dense = True
        else:
            raise ValueError("Argument 'solver' must be a valid solver")

        # If using a dense method, convert to a dense matrix
        if dense:
            base_hessian = []
            for diag_lambda_pow, off_diag_lambda_pow in zip(base_diag,
                                                            base_off_diag):
                base_hessian.append(self._create_dense(diag_lambda_pow,
                                                       off_diag_lambda_pow))

        alphas = []
        # Multiply in the value into the matrix equation and solve
        for lam in lambda_values:
            initial = initial_const + lam * initial_lambda
            initial = pad(initial, (0, len(base_hessian[0])-len(initial)),
                          mode="constant", constant_values=0)
            hessian = base_hessian[0] + lam * \
                base_hessian[1] + (lam**2)*base_hessian[2]
            alphas.append(solver_func(hessian=hessian,
                                      initial=initial,
                                      args=solver_args))

        # Create an AGP object
        return AGP(hamiltonian=self.hamiltonian,
                   lambda_index=self.lambda_index,
                   lambda_values=lambda_values,
                   constant_values=constant_values,
                   operators=self._odd_sets[:self._current_max_set],
                   coefficients=array(alphas),
                   exact=self._exact)