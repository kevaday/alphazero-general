class Board:
    """
    Author: MBoss
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
    def __init__(self, n, n_in_row):
        """Set up initial board configuration."""
        self.n = n
        self.n_in_row = n_in_row

        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        """
        moves = set()  # stores the legal moves.

        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.add((x, y))
        return list(moves)

    def has_legal_moves(self):
        """Returns True if has legal move else False
        """
        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def get_win_state(self):
        n = self.n_in_row

        for w in range(self.n):
            for h in range(self.n):
                if (w in range(self.n - n + 1) and self[w][h] != 0 and
                        len(set(self[i][h] for i in range(w, w + n))) == 1):
                    return True, self[w][h]
                if (h in range(self.n - n + 1) and self[w][h] != 0 and
                        len(set(self[w][j] for j in range(h, h + n))) == 1):
                    return True, self[w][h]
                if (w in range(self.n - n + 1) and h in range(self.n - n + 1) and self[w][
                    h] != 0 and
                        len(set(self[w + k][h + k] for k in range(n))) == 1):
                    return True, self[w][h]
                if (w in range(self.n - n + 1) and h in range(n - 1, self.n) and self[w][
                    h] != 0 and
                        len(set(self[w + l][h - l] for l in range(n))) == 1):
                    return True, self[w][h]

        if self.has_legal_moves():
            return False, 0
        return True, 0

    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        (x, y) = move
        assert self[x][y] == 0, f'invalid move {move}'
        self[x][y] = color

