# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True
# cython: profile=True

import numpy as np


cdef class Board():
    """
    Connect4 Board.
    """

    cdef int height
    cdef int width
    cdef int length
    cdef int win_length
    cdef public int[:,:] pieces

    def __init__(self, int height, int width, int win_length):
        """Set up initial board configuration."""
        self.height = height
        self.width = width
        self.win_length = win_length

        self.pieces = np.zeros((self.height, self.width), dtype=np.intc)

    def __getstate__(self):
        return self.height, self.width, self.win_length, np.asarray(self.pieces)

    def __setstate__(self, state):
        self.height, self.width, self.win_length, pieces = state
        self.pieces = np.asarray(pieces)

    def add_stone(self, int column, int player):
        """Create copy of board containing new stone."""
        cdef Py_ssize_t r
        for r in range(self.height):
            if self.pieces[(self.height-1)-r,column] == 0:
                self.pieces[(self.height-1)-r,column] = player
                return

        raise ValueError("Can't play column %s on board %s" % (column, self))

    def get_valid_moves(self):
        """Any zero value in top row is a valid move"""
        cdef Py_ssize_t c
        cdef int[:] valid = np.zeros((self.width), dtype=np.intc)
        for c in range(self.width):
            if self.pieces[0,c] == 0:
                valid[c] = 1

        return valid

    def get_win_state(self):
        cdef int player
        cdef int total
        cdef int good
        cdef Py_ssize_t r, c, x
        for player in [1, -1]:
            #check row wins
            for r in range(self.height):
                total = 0
                for c in range(self.width):
                    if self.pieces[r,c] == player:
                        total += 1
                    else:
                        total = 0
                    if total == self.win_length:
                        return (True, player)
            #check column wins
            for c in range(self.width):
                total = 0
                for r in range(self.height):
                    if self.pieces[r,c] == player:
                        total += 1
                    else:
                        total = 0
                    if total == self.win_length:
                        return (True, player)
            #check diagonal
            for r in range(self.height - self.win_length + 1):
                for c in range(self.width - self.win_length + 1):
                    good = True
                    for x in range(self.win_length):
                        if self.pieces[r+x,c+x] != player:
                            good = False
                            break
                    if good:
                        return (True, player)
                for c in range(self.win_length - 1, self.width):
                    good = True
                    for x in range(self.win_length):
                        if self.pieces[r+x,c-x] != player:
                            good = False
                            break
                    if good:
                        return (True, player)

        # draw has very little value.
        if sum(self.get_valid_moves()) == 0:
            return (True, 0)

        # Game is not ended yet.
        return (False, 0)

    def __str__(self):
        return str(np.asarray(self.pieces))
