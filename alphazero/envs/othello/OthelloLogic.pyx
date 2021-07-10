# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
"""
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
"""

cimport cython
import numpy as np

# list of all 8 directions on the board, as (x,y) offsets
cdef list __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

cdef class Board:
    cdef public int n
    cdef public int[:,:] pieces

    def __init__(self, int n, _pieces=None):
        """Set up initial board configuration."""

        self.n = n

        if _pieces is not None:
            self.pieces = _pieces
        else:
            # Create the empty board array.
            self.pieces = np.zeros((self.n, self.n), dtype=np.intc)

            # Set up the initial 4 pieces.
            self.pieces[self.n//2-1,self.n//2] = 1
            self.pieces[self.n//2,self.n//2-1] = 1
            self.pieces[self.n//2-1,self.n//2-1] = -1
            self.pieces[self.n//2,self.n//2] = -1

    def __getstate__(self):
        return self.n, np.asarray(self.pieces)

    def __setstate__(self, state):
        self.n, pieces = state
        self.pieces = np.asarray(pieces)

    # add [][] indexer syntax to the Board
    def __getitem__(self, Py_ssize_t index):
        return self.pieces[index]

    def count_diff(self, int color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        cdef int count = 0
        cdef Py_ssize_t x, y

        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[x,y] == color:
                    count += 1
                if self.pieces[x,y] == -color:
                    count -= 1

        return count

    cpdef set get_legal_moves(self, int color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        cdef set moves = set()  # stores the legal moves.

        # Get all the squares with pieces_raw of the given color.
        cdef Py_ssize_t x, y
        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[x,y]==color:
                    moves.update(self.get_moves_for_square((x,y)))
        return moves

    cpdef bint has_legal_moves(self, int color):
        cdef Py_ssize_t x, y
        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[x,y] == color:
                    if len(self.get_moves_for_square((x,y)))>0:
                        return True
        return False

    cpdef list get_moves_for_square(self, (Py_ssize_t, Py_ssize_t) square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        cdef Py_ssize_t x, y
        (x,y) = square

        # determine the color of the piece.
        cdef int color = self.pieces[x,y]

        # skip empty source squares.
        if color==0:
            return []

        # search all possible directions.
        cdef list moves = []
        cdef (Py_ssize_t, Py_ssize_t) move
        for direction in __directions:
            move = self._discover_move(square, direction)
            if move != (-1, -1):
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, (Py_ssize_t, Py_ssize_t) move, int color):
        """Perform the given move on the board; flips pieces_raw as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        cdef list flips = [flip for direction in __directions
                      for flip in self._get_flips(move, direction, color)]
        assert len(list(flips))>0
        cdef Py_ssize_t x, y
        for x, y in flips:
            #print(self.pieces_raw[x,y],color)
            self.pieces[x,y] = color

    cdef (Py_ssize_t, Py_ssize_t) _discover_move(self, (Py_ssize_t, Py_ssize_t) origin, (Py_ssize_t, Py_ssize_t) direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        cdef Py_ssize_t x, y
        x, y = origin
        cdef int color = self.pieces[x,y]
        cdef list flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self.pieces[x,y] == 0:
                if flips != []:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return (-1, -1)
            elif self.pieces[x,y] == color:
                return (-1, -1)
            elif self.pieces[x,y] == -1*color:
                # print("Flip",x,y)
                flips.append((x, y))
        return (-1, -1)

    cdef list _get_flips(self, (Py_ssize_t, Py_ssize_t) origin, (Py_ssize_t, Py_ssize_t) direction, int color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        cdef list flips = [origin]
        cdef Py_ssize_t x, y

        for x, y in Board._increment_move(origin, direction, self.n):
            #print(x,y)
            if self.pieces[x,y] == 0:
                return []
            if self.pieces[x,y] == -color:
                flips.append((x, y))
            elif self.pieces[x,y] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod
    cdef list _increment_move((Py_ssize_t, Py_ssize_t) move, (Py_ssize_t, Py_ssize_t) direction, int n):
        # print(move)
        """ Generator expression for incrementing moves """
        #move = list(map(sum, zip(move, direction)))
        cdef list moves = []
        move = (move[0]+direction[0], move[1]+direction[1])
        #while all(map(lambda x: 0 <= x < n, move)): 
        while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            moves.append(move)
            #move=list(map(sum,zip(move,direction)))
            move = (move[0]+direction[0],move[1]+direction[1])
        return moves

