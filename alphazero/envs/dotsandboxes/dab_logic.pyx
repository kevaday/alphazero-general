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
    cdef public int[:,:] pieces
    cdef public int possible_boxes
    def __init__(self, int height, int width, _pieces=None):
        """Set up initial board configuration."""
        self.height = height
        self.width = width
        self.possible_boxes = 0
        cdef Py_ssize_t x, y

        if _pieces is not None:
            self.pieces = _pieces
        else:
            self.pieces = np.zeros((self.height, self.width), dtype=np.intc)
            
            for y in range(self.height):
                for x in range(self.width):
                    if (x + y) % 2 == 0:
                        self.pieces[y][x] = 2

        for y in range(self.height):
            for x in range(self.width):
                if (x + y) % 2 != 0 and x != 0 and y <= self.height-3 and x != self.width-1:
                    self.possible_boxes += 1
        """
        1, 0, 1, 0, 1
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1
        """

    def __getstate__(self):
        return self.height, self.width, np.asarray(self.pieces)

    def __setstate__(self, state):
        self.height, self.width, pieces = state
        self.pieces = np.asarray(pieces)

    def add_piece(self, int y_, int x_, int player):
        now_boxes = self.check_boxes()
        self.pieces[y_][x_] = player

        cdef Py_ssize_t x, y
        if self.check_boxes() > now_boxes:
            for y in range(self.height):
                for x in range(self.width):
                    if (x + y) % 2 != 0 and y % 2 == 0 and x != 0 and y <= self.height-3 and x != self.width-1:
                        if self.pieces[y][x] != 0 and \
                            self.pieces[y+1][x-1] != 0 and \
                            self.pieces[y+1][x+1] != 0 and \
                            self.pieces[y+2][x] != 0:
                            if (y_ == y and x_ == x) or \
                                (y_ == y+1 and x_ == x-1) or \
                                (y_ == y+1 and x_ == x+1) or \
                                (y_ == y+2 and x_ == x):
                                self.pieces[y+1][x] = player * 3

    def check_boxes(self):
        cdef Py_ssize_t x, y
        cdef int boxes = 0
        for y in range(self.height):
            for x in range(self.width):
                if (x + y) % 2 != 0 and y % 2 == 0 and x != 0 and y <= self.height-3 and x != self.width-1:
                    if self.pieces[y][x] != 0 and \
                        self.pieces[y+1][x-1] != 0 and \
                        self.pieces[y+1][x+1] != 0 and \
                        self.pieces[y+2][x] != 0:
                        boxes += 1
        return boxes

    def get_valids(self):
        cdef Py_ssize_t x, y
        cdef list valids = []
        for y in range(self.height):
            for x in range(self.width):
                if (x + y) % 2 != 0 and self.pieces[y][x] == 0:
                    valids.append((y, x))
        return valids

    #These 2 functions are HORRIBLE!
    #Im trying to figure out how to optimise them
    def map_to_num(self, p):
        cdef Py_ssize_t x, y
        cdef int i = 0
        for y in range(self.height):
            for x in range(self.width):
                if (x + y) % 2:
                    if p[0] == y and p[1] == x:
                        return i
                    i+=1
    def reverse_map(self, n):
        cdef Py_ssize_t x, y
        cdef int i = 0
        for y in range(self.height):
            for x in range(self.width):
                if (x + y) % 2:
                    if i == n:
                        return (y, x)
                    i+=1