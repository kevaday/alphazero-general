import numpy as np


cdef class Board():
    """
    Connect4 Board.
    """

    cdef public int size
    cdef public int[:,:,:,:] pieces

    def __init__(self, _pieces=None):
        """Set up initial board configuration."""
        self.size = 3
        if _pieces is not None:
            self.pieces = _pieces
        else:
            self.pieces = np.zeros((self.size, self.size, self.size, self.size), dtype=np.intc)
    def __getstate__(self):
            return self.size, np.asarray(self.pieces)

    def __setstate__(self, state):
        self.size, pieces = state
        self.pieces = np.asarray(pieces)
    def place_piece(self, int major_board_y, int major_board_x, 
                    int minor_board_y, int minor_board_x,
                    int player):

        self.pieces[major_board_y][major_board_x][minor_board_y][minor_board_x] = player

    def check_minor_win(self, int player, int y, int x):
        cdef int y_, x_ = 0
        cdef int win = True
        #Check Rows
        for y_ in range(self.size):
            win = True
            for x_ in range(self.size):
                if self.pieces[y][x][y_][x_] != player:
                    win = False
            if win:
                return 1
        #Cols
        for x_ in range(self.size):
            win = True
            for y_ in range(self.size):
                if self.pieces[y][x][y_][x_] != player:
                    win = False
            if win:
                return 1
        #Diag
        win = True
        for y_ in range(self.size):
            if self.pieces[y][x][y_][y_] != player:
                win = False
        if win:
            return 1
        #Anti Diag
        win = True
        for y_ in range(self.size):
            if self.pieces[y][x][y_][self.size-y_-1] != player:
                win = False
        if win:
            return 1
        return 0

    cdef int check_major_win(self, int player):
        cdef int y_, x_ = 0
        cdef int win = True
        for y_ in range(self.size):
            win = True
            for x_ in range(self.size):
                if self.check_minor_win(player, y_, x_) == 0:
                    win = False
            if win:
                return 1
        #Cols
        for x_ in range(self.size):
            win = True
            for y_ in range(self.size):
                if self.check_minor_win(player, y_, x_) == 0:
                    win = False
            if win:
                return 1
        #Diag
        win = True
        for y_ in range(self.size):
            if self.check_minor_win(player, y_, y_) == 0:
                win = False
        if win:
            return 1
        #Anti Diag
        win = True
        for y_ in range(self.size):
            if self.check_minor_win(player, y_, self.size-y_-1) == 0:
                win = False
        if win:
            return 1

        return 0

    def check_win(self, int player):
        return self.check_major_win(player)

    def get_valid_moves(self, int y, int x):
        cdef list valids = []
        cdef int x_, y_ = 0
        if self.check_minor_win(1, y, x) == 1 or self.check_minor_win(-1, y, x) == 1:
            return self.get_all_open()
        for y_ in range(self.size):
            for x_ in range(self.size):
                if self.pieces[y][x][y_][x_] == 0:
                    valids.append((y, x, y_, x_))
        return valids
    def get_all_open(self):
        cdef list valids = []
        cdef int x_, y_ = 0
        for y in range(self.size):
            for x in range(self.size):
                for y_ in range(self.size):
                    for x_ in range(self.size):
                        if self.pieces[y][x][y_][x_] == 0 and not (self.check_minor_win(1, y, x) == 1 or self.check_minor_win(-1, y, x) == 1):
                            valids.append((y, x, y_, x_))
        return valids
    def point_to_num(self, int y, int x, int y_, int x_):
        return 27*y + 3*x + 9*y_ + x_
    def num_to_point(self, int n):
        return (n // 27, (n // 3) % 3, (n-(n//27 * 27)-(((n//3)%3)*3)) // 9, (n-(n//27 * 27)-(((n//3)%3)*3)) % 3)

                