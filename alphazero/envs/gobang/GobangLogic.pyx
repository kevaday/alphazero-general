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


cdef class Board:
    """
    Author: MBoss. Modified by Kevaday for Cython
    Date: Jan 17, 2018.
    Board class.
    Board data:
      1=white, -1=black, 0=empty
      first dim is column , 2nd is row:
         pieces[1][7] is the square in column 2,
         at the opposite end of the board in row 8.
    Squares are stored and manipulated as (x,y) tuples.
    x is the column, y is the row.
    """
    
    def __init__(self, int n, int n_in_row, _pieces=None):
        """Set up initial board configuration."""
        self.n = n
        self.n_in_row = n_in_row

        if _pieces is not None:
            self.pieces = _pieces
        else:
            # Create the empty board array.
            self.pieces = np.zeros((n, n), dtype=np.intc)

    # add [][] indexer syntax to the Board
    def __getitem__(self, tuple index): 
        return self.pieces[index]
    
    def __setitem__(self, tuple index, int value):
        self.pieces[index] = value

    cpdef list get_legal_moves(self):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        """
        cdef tuple pos
        return [tuple(reversed(pos)) for pos in zip(*np.where(self.pieces == 0))]

    cpdef bint has_legal_moves(self):
        """Returns True if has legal move else False
        """
        return 0 in self.pieces

    cpdef tuple get_win_state(self):
        cdef Py_ssize_t n, w, h, i, j, k, l
        n = self.n_in_row

        for w in range(self.n):
            for h in range(self.n):
                if (w in range(self.n - n + 1) and self[w, h] != 0 and
                        len(set([self[i, h] for i in range(w, w + n)])) == 1):
                    return True, self[w, h]
                if (h in range(self.n - n + 1) and self[w, h] != 0 and
                        len(set([self[w, j] for j in range(h, h + n)])) == 1):
                    return True, self[w, h]
                if (w in range(self.n - n + 1) and h in range(self.n - n + 1) and self[w, 
                    h] != 0 and
                        len(set([self[w + k, h + k] for k in range(n)])) == 1):
                    return True, self[w, h]
                if (w in range(self.n - n + 1) and h in range(n - 1, self.n) and self[w, 
                    h] != 0 and
                        len(set([self[w + l, h - l] for l in range(n)])) == 1):
                    return True, self[w, h]

        if self.has_legal_moves():
            return False, 0
        return True, 0

    cpdef void execute_move(self, tuple move, int color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color of the piece to play (1=white,-1=black)
        """
        assert self[move] == 0, f'invalid move {move}'
        self[move] = color
