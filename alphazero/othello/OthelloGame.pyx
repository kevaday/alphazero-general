# cython: language_level=3

from alphazero.othello.OthelloLogic import Board
from alphazero.Game import Game

import numpy as np

NUM_PLAYERS = 2
NUM_CHANNELS = 1


class OthelloGame(Game):
    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.asarray(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return self.n * self.n + 1

    def getObservationSize(self):
        return (NUM_CHANNELS, *self.getBoardSize())

    def getPlayers(self):
        return list(range(NUM_PLAYERS))

    def _player_range(self, player):
        return 1 if player == self.getPlayers()[0] else -1

    def getNextState(self, board, player, action, copy=True):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n)
        b.pieces = np.copy(board) if copy else board

        if action == self.n * self.n + 1:
            return b, self.getNextPlayer(player)

        move = (action // self.n, action % self.n)
        b.execute_move(move, player)
        return np.asarray(b.pieces), self.getNextPlayer(player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = board

        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)

        for x, y in legalMoves:
            valids[self.n*x+y] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        b = Board(self.n)
        b.pieces = board
        player = self._player_range(player)

        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player, copy=True):
        # return state if player==1, else return -state if player==-1
        return board * self._player_range(player)

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)


def display(board):
    n = board.shape[0]

    for y in range(n):
        print(y, "|", end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|", end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1:
                print("b ", end="")
            elif piece == 1:
                print("W ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")

    print("   -----------------------")
