import numpy as np

# list of all 8 directions on the board, as (x,y) offsets
cdef list __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]
cdef (Py_ssize_t, Py_ssize_t) null_point = (-1, -1)


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
						self.pieces[self.n//2-1,self.n//2] = -1
						self.pieces[self.n//2,self.n//2-1] = -1
						self.pieces[self.n//2-1,self.n//2-1] = 1
						self.pieces[self.n//2,self.n//2] = 1
		def __getstate__(self):
			return self.n, np.asarray(self.pieces)

		def __setstate__(self, state):
			self.n, pieces = state
			self.pieces = np.asarray(pieces)

		def get_total(self, int color):
			return np.count_nonzero(np.asarray(self.pieces) == color)
		cdef list get_flips(self, (Py_ssize_t, Py_ssize_t) origin, (Py_ssize_t, Py_ssize_t) direction, int color):
			"""
			Get all the flips possible in a given direction from a given starting origin
			"""
			cdef Py_ssize_t x, y
			x, y = origin
			cdef list flips = []

			x += direction[0]
			y += direction[1]
			while x < self.n and y < self.n:
				if self.pieces[x,y] == -color:
					flips.append((x,y))
					x += direction[0]
					y += direction[1]
				elif self.pieces[x,y] == color:
					return flips
				else:
					return []
			return []

		def get_legal(self, int color):
			"""
			Use number of flips possible to generate legal moves
			"""
			cdef Py_ssize_t x, y
			cdef list moves = []
			for y in range(self.n):
				for x in range(self.n):
					if self.pieces[x,y] == 0:
						for d in __directions:
							if len(self.get_flips((x,y), d, color)) > 0:
								moves.append((y,x))
			return moves

		def execute_move(self, (Py_ssize_t, Py_ssize_t) move, int color):
			cdef list flips = []
			cdef list sub_flips = []
			cdef Py_ssize_t x, y
			for d in __directions:
				sub_flips = self.get_flips((move[1], move[0]), d, color)
				for i in sub_flips:
					flips.append(i)

			self.pieces[move[1], move[0]] = color
			for x, y in flips:
				self.pieces[x,y] = color
