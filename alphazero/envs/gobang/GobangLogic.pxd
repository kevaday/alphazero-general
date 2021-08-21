# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

import numpy as np
cimport numpy as np

DTYPE = np.intc
ctypedef np.int32_t DTYPE_t


cdef class Board:
    cdef public int n
    cdef public int n_in_row
    cdef public np.ndarray pieces
    
    cpdef list get_legal_moves(self)
    cpdef bint has_legal_moves(self)
    cpdef tuple get_win_state(self)
    cpdef void execute_move(self, tuple move, int color)

